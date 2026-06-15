import torch 
from torch import nn 
import torch.utils.data as torchdata
from dataloader import PatchDataset
import torch.nn.functional as F 

from visualizer.hook import start_visualizer, send_tensor


class LTE(nn.Module): 
  """
  可学习纹理提取器
  """

  def __init__(self, feat_ch=64):
    super().__init__() 
    self.body = nn.Sequential(
        nn.Conv2d(1, feat_ch, 3, 1, 1), 
        nn.ReLU(),  # nn.ReLU(inplace=True) to optimize 
        nn.Conv2d(feat_ch, feat_ch, 3, 1, 1), 
        nn.ReLU(), 
        nn.Conv2d(feat_ch, feat_ch, 3, 1, 1), 
      )

    self.scale = nn.Sequential(
      nn.Conv2d(1, feat_ch, 3, 1, 1), 
      nn.ReLU(),  # nn.ReLU(inplace=True) to optimize 
      nn.Conv2d(feat_ch, feat_ch, 3, 1, 1), 
      nn.ReLU(), 
    )
    self.scale_final = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)


  
  def forward(self, lr, ref, target_size=None):  # lr is 低质量图像
    _, _, H, W = lr.shape 
    _, _, H1, W1 = ref.shape 
    scale_h = H / H1
    scale_w = W / W1 
    lr_up = F.interpolate(lr, size=(H1, W1), mode="bicubic", align_corners=False)
    ref_down = F.interpolate(ref, scale_factor=(scale_h, scale_w), mode="bicubic", align_corners=False)
    ref_down_up = F.interpolate(ref_down, size=(H1, W1), mode="bicubic", align_corners=False)

    if target_size:
      Q = self.scale(lr_up) 
      Q = F.adaptive_avg_pool2d(Q, target_size) 
      Q = self.scale_final(Q)
    else: 
      Q = self.body(lr_up)
    K = self.body(ref_down_up) 
    
    V = self.body(ref) 

    return Q, K, V

class RelianceEmbedding(nn.Module): 
  """
  相关性嵌入
  """
  def __init__(self, kernel_size=3):
    super().__init__()
    self.ks = kernel_size
    self.pd = self.ks // 2

  def forward(self, Q, K):  # Q: (B, C, Hl, Wl), K: (B, C, Hr, Wr)

    # patch num = (H + 2*pd - ks) // stride + 1
    # if ks = 3 and pd = 1, stride = 1, then number of patch will be HxW(have to be same)
    # let N_q = the number of patchs = Hl x Wl, N_r = Hr x Wr, N_p(atch) = C x ks x ks
    
    outQ = F.unfold(Q, kernel_size=self.ks, padding=self.pd) # outQ: (B, N_p, N_q)
    outK = F.unfold(K, kernel_size=self.ks, padding=self.pd) # outK: (B, N_p, N_r)

    outQ = F.normalize(outQ, p=2, dim=1) # L2 normal
    outK = F.normalize(outK, p=2, dim=1) 

    rel = torch.einsum("bdi,bdj->bij", outQ, outK) # rel: (B, N_q, N_r)

    H = torch.argmax(rel, dim=2)  # H: (B, N_q), hi \in (0, N_r)
    S, _ = torch.max(rel, dim=2)   # S: (B, N_q), si \in (0, 1)

    return H, S

class HardAttention(nn.Module): 
  def forward(self, H, V): 

    B, C, Hr, Wr = V.shape 
    N_q = H.shape[1] # N_q should = N_r (temporary)
    N_r = Hr * Wr

    V_flat = V.reshape(B, C, N_r) 
    H_idx = H.unsqueeze(1).expand(B, C, N_q)   # 在dim=1上复制C份, C_idx中的元素属于 0 ~ N_r

    T = torch.gather(V_flat, dim=2, index=H_idx) # T: (B, C, N_q), ti's range = vi's range
    return T 

class Backbone(nn.Module):
    """
    示例用的骨干网
    输入: LR: Hl x Wl 
    输出: 特征图 (feat_ch, Hlr, Wlr)
    """
    def __init__(self, in_ch=1, feat_ch=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
        )

    def forward(self, lr):
        return self.body(lr)   # (B, C, H, W)
    
class SoftAttention(nn.Module): 
  """
  F: (B, C, Hl, Wl)
  T: (B, C, N_q) 
  S: (B, N_q)
  """
  def __init__(self, feat_ch=64): 
    super().__init__() 
    self.fusion_conv = nn.Sequential(
      nn.Conv2d(2*feat_ch, feat_ch, 3, 1, 1), 
      nn.ReLU()
    )

  def forward(self, F, T, S): 
    B, C, H, W = F.shape
    T = T.view(B, C, H, W) 
    S = S.view(B, 1, H, W)

    fused = torch.cat([F, T], dim=1) # concat them with channel dimension (B, 2*C, H, W)
    fused = self.fusion_conv(fused)  # fuse them (B, C, H, W)

    result = F + fused * S # F+: resnet; fused * S: 门控; result: (B, C, H, W)

    return result 
  

class OutputLayer(nn.Module): 
  """
  算出F + fused * S后上采样
  """
  def __init__(self, in_ch=1, feat_ch=64): 
    super().__init__()
    self.conv = nn.Conv2d(feat_ch, in_ch, 3, 1, 1) 
  def forward(self, x, Hr, Wr): 
    x_up = F.interpolate(x, size=(Hr, Wr), mode="bicubic", align_corners=False)
    x_output = self.conv(x_up) 
    return x_output



class TTSR(nn.Module): 
  """
  典型尺寸: sample: (B, C, H, W) = 4, 1, 32, 128; ref: (B, C, H, W) = 4, 1, 128, 128
  """
  def __init__(self, in_ch=1, feat_ch=64):
    super().__init__()
    self.name = "TTSR-basic"

    self.LTE = LTE(feat_ch)
    self.RE = RelianceEmbedding(kernel_size=3)
    self.HA = HardAttention()
    self.SA = SoftAttention(feat_ch)
    self.BB = Backbone(in_ch, feat_ch)
    self.OL = OutputLayer(in_ch, feat_ch) 
  def forward(self, lr, ref): 
    Hl, Wl, Hr, Wr = lr.shape[2], lr.shape[3], ref.shape[2], ref.shape[3]

    Q, K, V = self.LTE(lr, ref, target_size=(Hl, Wl)) # K, V: (B, C, Hr, Wr), Q = 
    H, S = self.RE(Q, K)
    T = self.HA(H, V)  
    F = self.BB(lr)
    
    output = self.SA(F, T, S) 

    # send_tensor("out", output[0][0], f"TTSR top output")

    output = self.OL(output, Hr, Wr)
    
    return output





