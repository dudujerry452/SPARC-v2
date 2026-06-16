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

class LTEVGG(nn.Module): 
    def __init__(self, in_ch=1): 
      super().__init__() 
      self.conv1 = nn.Sequential(
         nn.Conv2d(in_ch, 64, 3, 1, 1), 
         nn.ReLU(), 
         nn.Conv2d(64, 64, 3, 1, 1), 
         nn.ReLU(), 
      )

      self.conv2 = nn.Sequential(
         nn.MaxPool2d(2,2), 
         nn.Conv2d(64, 128, 3, 1, 1), 
         nn.ReLU(), 
         nn.Conv2d(128, 128, 3, 1, 1), 
         nn.ReLU(), 
      )

      self.conv3 = nn.Sequential(
         nn.MaxPool2d(2,2), 
         nn.Conv2d(128, 256, 3, 1, 1), 
         nn.ReLU(), 
         nn.Conv2d(256, 256, 3, 1, 1), 
         nn.ReLU(), 
      )

    def forward(self, x): 
        x1 = self.conv1(x) 
        x2 = self.conv2(x1) 
        x3 = self.conv3(x2)

        return x1, x2, x3 

class SearchTransfer(nn.Module): 
   """
    combination of RE and HA, input Q, K, V and get S, T
    Q: (B, C, Hlv3, Wlv3)
    K: (B, C, Hlv3, Wlv3) 
    V: (B, C, Hlv1-3, Hlv1-3)
   """
   def forward(self, Q, K, V_v1, V_v2, V_v3):  
      
      B, _, H, W = Q.shape

      q_patchs = F.unfold(Q, kernel_size=3, padding=1) # (B, N_patch, N_q = Hlv3 * Wlv3)
      k_patchs = F.unfold(K, kernel_size=3, padding=1) 

      q_patchs = F.normalize(q_patchs, p=2, dim=1) 
      k_patchs = F.normalize(k_patchs, p=2, dim=1) 

      rel = torch.einsum("bdi,bdj->bij", q_patchs, k_patchs) # (B, N_q, N_q)

      S, H_idx = torch.max(rel, dim=2) # (B, N_q)

      V_v3_patchs = F.unfold(V_v3, kernel_size=3, padding=1)
      V_v2_patchs = F.unfold(V_v2, stride=2, kernel_size=6, padding=2) # 首先确定/2, 然后+4-6 = -2 = -1*2
      V_v1_patchs = F.unfold(V_v1, stride=4, kernel_size=12, padding=4)  # (B, N_p, N_q)

      
      idx = H_idx.unsqueeze(1).expand(-1, V_v3_patchs.shape[1], -1)
      T_v3 = torch.gather(V_v3_patchs, dim=2, index=idx)  # T_v3[B][:][i] = V_v3_patchs[B][:][hi]
      idx = H_idx.unsqueeze(1).expand(-1, V_v2_patchs.shape[1], -1)
      T_v2 = torch.gather(V_v2_patchs, dim=2, index=idx)  # output = [b, c, input[b, c, i]]
      idx = H_idx.unsqueeze(1).expand(-1, V_v1_patchs.shape[1], -1)
      T_v1 = torch.gather(V_v1_patchs, dim=2, index=idx) 

      # fold back
      _, _, H3, W3 = V_v3.shape
      _, _, H2, W2 = V_v2.shape
      _, _, H1, W1 = V_v1.shape

      T_v3 = F.fold(T_v3, output_size=(H3, W3), kernel_size=3, padding=1) / 9
      T_v2 = F.fold(T_v2, output_size=(H2, W2), kernel_size=6, padding=2, stride=2) / 9
      T_v1 = F.fold(T_v1, output_size=(H1, W1), kernel_size=12, padding=4, stride=4) / 9 

      S = S.view(B, 1, H, W) 

      return S, T_v1, T_v2, T_v3





class ResBlock(nn.Module): 
    def __init__(self, feat_ch=64, res_scale=1.0): 
      super().__init__()
      self.scale = res_scale
      self.body = nn.Sequential(
         nn.Conv2d(feat_ch, feat_ch, 3, 1, 1), 
         nn.ReLU(), 
         nn.Conv2d(feat_ch, feat_ch, 3, 1, 1), 
      )
    def forward(self, x): 
       res = self.body(x) 
       return x + res * self.scale
    
class SFE(nn.Module): 
    def __init__(self, in_ch=1, feat_ch=64, res_scale=1.0): 
        super().__init__() 
        self.scale = res_scale 
        self.conv_head = nn.Conv2d(in_ch, feat_ch, 3, 1, 1) 
        self.resblk = nn.Sequential(
            ResBlock(feat_ch), 
            ResBlock(feat_ch), 
            ResBlock(feat_ch) 
        )
        self.conv_tail = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1) 
    def forward(self, x): 
        x = self.conv_head(x) 
        res = self.resblk(x) 
        res = self.conv_tail(res) 
        return x + res * self.scale 
   

class SoftAttention(nn.Module): 
  """
  F: (B, C, Hl, Wl)
  T: (B, C, N_q) 
  S: (B, N_q)
  """
  def __init__(self, feat_ch=64): 
    super().__init__() 
    self.fusion_conv = nn.Sequential(
      nn.Conv2d(feat_ch+ 256, feat_ch, 3, 1, 1),  # 256 from VGG19
      nn.ReLU()
    )

  def forward(self, F, T, S): 
    B, C, H, W = F.shape
    # print(F.shape, T.shape, S.shape)

    fused = torch.cat([F, T], dim=1) # concat them with channel dimension (B, 2*C, H, W)
    fused = self.fusion_conv(fused)  # fuse them (B, C, H, W)

    result = F + fused * S # F+: resnet; fused * S: 门控; result: (B, C, H, W)

    return result 
  
class PixelShuffleH(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, Cr, H, W = x.shape
        r = self.upscale_factor
        C = Cr // r

        # (B, C, r, H, W)
        x = x.reshape(B, C, r, H, W)

        # (B, C, H, r, W) -> (B, C, H*r, W)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, C, H * r, W)

        return x
class PixelShuffleW(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, Cr, H, W = x.shape
        r = self.upscale_factor
        C = Cr // r

        # (B, C, r, H, W)
        x = x.reshape(B, C, r, H, W)

        # (B, C, H, W, r) -> (B, C, H, W*r)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(B, C, H, W * r)

        return x



class OutputLayer(nn.Module): 
  """
  算出F + fused * S后上采样
  """
  def __init__(self, in_ch=1, feat_ch=64, scale=(4, 1)): 
    super().__init__()
    self.conv = nn.Conv2d(feat_ch, in_ch * scale[0] * scale[1], 3, 1, 1) 
    self.shuffleh = PixelShuffleH(scale[0]) 
    self.shufflew = PixelShuffleW(scale[1]) 
    self.scale = scale
  def forward(self, x):
    x = self.conv(x) 

    if self.scale[0] != 1: 
       x = self.shuffleh(x) 
    if self.scale[1] != 1: 
       x = self.shufflew(x) 
    return x



class TTSR(nn.Module): 
  """
  典型尺寸: sample: (B, C, H, W) = 4, 1, 32, 128; ref: (B, C, H, W) = 4, 1, 128, 128
  """
  def __init__(self, in_ch=1, feat_ch=64):
    super().__init__()
    self.name = "TTSR-basic"

    self.LTEVGG = LTEVGG(in_ch)
    self.SearchTransfer = SearchTransfer()
    self.SA = SoftAttention(feat_ch)
    self.Backbone = SFE(in_ch, feat_ch)
    self.OL = OutputLayer(in_ch, feat_ch, scale=(4,1)) 
  def forward(self, lr, ref): 

    B, _, H, W = lr.shape 
    _, _, H1, W1 = ref.shape 
    scale_h = H / H1
    scale_w = W / W1 

    lr_up = F.interpolate(lr, size=(H1, W1), mode="bicubic", align_corners=False)
    ref_down = F.interpolate(ref, scale_factor=(scale_h, scale_w), mode="bicubic", align_corners=False)
    ref_down_up = F.interpolate(ref_down, size=(H1, W1), mode="bicubic", align_corners=False)

    lrsr_v1, lrsr_v2, lrsr_v3 = self.LTEVGG(lr_up) # v1, v2, v3: /1, /2, /4
    refsr_v1, refsr_v2, refsr_v3 = self.LTEVGG(ref_down_up) 
    ref_v1, ref_v2, ref_v3 = self.LTEVGG(ref) 

    Q, K, V = lrsr_v3, refsr_v3, ref_v3 # (B, C, H1/4, W1/4)

    S, T_v1, T_v2, T_v3 = self.SearchTransfer(Q, K, ref_v1, ref_v2, ref_v3)

    T = T_v3 

    S = F.interpolate(S, size=(lr.shape[2], lr.shape[3]), align_corners=False, mode="bicubic")
    T = F.interpolate(T, size=(lr.shape[2], lr.shape[3]), align_corners=False, mode="bicubic")


    F_ = self.Backbone(lr) # (B, C, H, W)
    
    output = self.SA(F_, T, S) 

    # send_tensor("out", output[0][0], f"TTSR top output")

    output = self.OL(output)
    
    return output





