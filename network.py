import torch
from torch import nn
import torch.utils.data as torchdata
from dataloader import PatchDataset
import torch.nn.functional as F
import math

from visualizer.hook import start_visualizer, send_tensor


class LTEVGG(nn.Module):
    def __init__(self, in_ch=1):
      super().__init__()
      self.conv1 = nn.Sequential(
         nn.Conv3d(in_ch, 64, (3,3,3), 1, 1),
         nn.ReLU(),
         nn.Conv3d(64, 64, (3,3,3), 1, 1),
         nn.ReLU(),
      )

      self.conv2 = nn.Sequential(
         nn.MaxPool3d((1,2,1), (1,2,1)),
         nn.Conv3d(64, 128, (3,3,3), 1, 1),
         nn.ReLU(),
         nn.Conv3d(128, 128, (3,3,3), 1, 1),
         nn.ReLU(),
      )

      self.conv3 = nn.Sequential(
         nn.MaxPool3d((1,2,1), (1,2,1)),
         nn.Conv3d(128, 256, (3,3,3), 1, 1),
         nn.ReLU(),
         nn.Conv3d(256, 256, (3,3,3), 1, 1),
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
    Q: (B, C, T, Hlv3, Wlv3)
    K: (B, C, T, Hlv3, Wlv3)
    V: (B, C, T, Hlv1-3, Wlv1-3)
   """
   def forward(self, Q, K, V_v1, V_v2, V_v3):

      B, _, T, H, W = Q.shape

      # frame-wise 2D unfold for Q, K
      Q_2d = Q.view(B * T, Q.shape[1], H, W)
      K_2d = K.view(B * T, K.shape[1], H, W)

      q_patchs = F.unfold(Q_2d, kernel_size=3, padding=1) # (B*T, C*9, H*W)
      k_patchs = F.unfold(K_2d, kernel_size=3, padding=1)

      # split T dimension: (B, T, C*9, H*W)
      q_patchs = q_patchs.view(B, T, -1, H * W)
      k_patchs = k_patchs.view(B, T, -1, H * W)

      q_patchs = F.normalize(q_patchs, p=2, dim=2)
      k_patchs = F.normalize(k_patchs, p=2, dim=2)

      rel = torch.einsum("btci,btcj->btij", q_patchs, k_patchs) # (B, T, H*W, H*W)

      S, H_idx = torch.max(rel, dim=3) # (B, T, H*W)

      # frame-wise 2D unfold for V
      V_v3_2d = V_v3.view(B * T, V_v3.shape[1], V_v3.shape[3], V_v3.shape[4])
      V_v2_2d = V_v2.view(B * T, V_v2.shape[1], V_v2.shape[3], V_v2.shape[4])
      V_v1_2d = V_v1.view(B * T, V_v1.shape[1], V_v1.shape[3], V_v1.shape[4])

      V_v3_patchs = F.unfold(V_v3_2d, kernel_size=3, padding=1)
      V_v2_patchs = F.unfold(V_v2_2d, stride=(2,1), kernel_size=(6,3), padding=(2,1))
      V_v1_patchs = F.unfold(V_v1_2d, stride=(4,1), kernel_size=(12,3), padding=(4,1))

      # H_idx: (B, T, H*W) -> (B*T, 1, H*W)
      H_idx_bt = H_idx.view(B * T, 1, H * W).expand(-1, V_v3_patchs.shape[1], -1)
      T_v3 = torch.gather(V_v3_patchs, dim=2, index=H_idx_bt)  # T_v3[B][:][i] = V_v3_patchs[B][:][hi]
      H_idx_bt = H_idx.view(B * T, 1, H * W).expand(-1, V_v2_patchs.shape[1], -1)
      T_v2 = torch.gather(V_v2_patchs, dim=2, index=H_idx_bt)  # output = [b, c, input[b, c, i]]
      H_idx_bt = H_idx.view(B * T, 1, H * W).expand(-1, V_v1_patchs.shape[1], -1)
      T_v1 = torch.gather(V_v1_patchs, dim=2, index=H_idx_bt)

      # fold back per frame
      _, _, _, H3, W3 = V_v3.shape
      _, _, _, H2, W2 = V_v2.shape
      _, _, _, H1, W1 = V_v1.shape

      T_v3 = F.fold(T_v3, output_size=(H3, W3), kernel_size=3, padding=1) / 9
      T_v2 = F.fold(T_v2, output_size=(H2, W2), kernel_size=(6,3), padding=(2,1), stride=(2,1)) / 9
      T_v1 = F.fold(T_v1, output_size=(H1, W1), kernel_size=(12,3), padding=(4,1), stride=(4,1)) / 9

      # reshape back to 3D
      T_v3 = T_v3.view(B, V_v3.shape[1], T, H3, W3)
      T_v2 = T_v2.view(B, V_v2.shape[1], T, H2, W2)
      T_v1 = T_v1.view(B, V_v1.shape[1], T, H1, W1)

      S = S.view(B, 1, T, H, W)

      return S, T_v1, T_v2, T_v3





class ResBlock(nn.Module):
    def __init__(self, feat_ch=64, res_scale=1.0):
      super().__init__()
      self.scale = res_scale
      self.body = nn.Sequential(
         nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1),
         nn.ReLU(),
         nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1),
      )
    def forward(self, x):
       res = self.body(x)
       return x + res * self.scale

class SFE(nn.Module):
    def __init__(self, in_ch=1, feat_ch=64, res_scale=1.0):
        super().__init__()
        self.scale = res_scale
        self.conv_head = nn.Conv3d(in_ch, feat_ch, (3,3,3), 1, 1)
        self.resblk = nn.Sequential(
            ResBlock(feat_ch),
            ResBlock(feat_ch),
            ResBlock(feat_ch)
        )
        self.conv_tail = nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1)
    def forward(self, x):
        x = self.conv_head(x)
        res = self.resblk(x)
        res = self.conv_tail(res)
        return x + res * self.scale


class PixelShuffleH3D(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, Cr, D, H, W = x.shape
        r = self.upscale_factor
        C = Cr // r

        # (B, C, r, D, H, W)
        x = x.reshape(B, C, r, D, H, W)

        # (B, C, D, H*r, W)
        x = x.permute(0, 1, 3, 4, 2, 5)
        x = x.reshape(B, C, D, H * r, W)

        return x
class PixelShuffleW3D(nn.Module):
    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, Cr, D, H, W = x.shape
        r = self.upscale_factor
        C = Cr // r

        # (B, C, r, D, H, W)
        x = x.reshape(B, C, r, D, H, W)

        # (B, C, D, H, W*r)
        x = x.permute(0, 1, 3, 4, 5, 2)
        x = x.reshape(B, C, D, H, W * r)

        return x



class MainNet(nn.Module):
  """
  拼接三个尺度的V, 以及lr, S
  """

  def __init__(self, feat_ch=64, out_ch=1, scale=(2,1)):
    super().__init__()
    self.SFE = SFE(1, 64)

    self.scale = scale
    up_dims = scale[0] * scale[1]
    self.conv =  nn.ModuleList([nn.Conv3d(feat_ch + 256, feat_ch, (3,3,3), 1, 1),
                    nn.Conv3d(feat_ch + 128, feat_ch, (3,3,3), 1, 1) ,
                    nn.Conv3d(feat_ch + 64, feat_ch, (3,3,3), 1, 1)])
    self.up_conv =  nn.ModuleList([nn.Conv3d(feat_ch, feat_ch * up_dims, (3,3,3), 1, 1 ),
                        nn.Conv3d(feat_ch, feat_ch * up_dims, (3,3,3), 1, 1 ),
                        nn.Conv3d(feat_ch, feat_ch * up_dims, (3,3,3), 1, 1 )])
    self.final_conv = nn.Conv3d(feat_ch, out_ch, (3,3,3), 1, 1)

    if self.scale[0] != 1:
      self.shuffle_h = nn.ModuleList([PixelShuffleH3D(scale[0]),
                            PixelShuffleH3D(scale[0])])
    else:
      self.shuffle_h = None
    if self.scale[1] != 1:
      self.shuffle_w = nn.ModuleList([PixelShuffleW3D(scale[1]),
                        PixelShuffleW3D(scale[1])])
    else:
      self.shuffle_w = None

    self.resblks = nn.ModuleList([
       nn.Sequential(ResBlock(feat_ch), ResBlock(feat_ch), ResBlock(feat_ch)),
       nn.Sequential(ResBlock(feat_ch), ResBlock(feat_ch), ResBlock(feat_ch)),
       nn.Sequential(ResBlock(feat_ch), ResBlock(feat_ch), ResBlock(feat_ch))
    ])
    self.res_tails = nn.ModuleList([
       nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1),
       nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1),
       nn.Conv3d(feat_ch, feat_ch, (3,3,3), 1, 1)
    ])

    self.merge_conv = nn.Conv3d(feat_ch*3, feat_ch, (3,3,3), 1, 1)

  def stage(self, T, S, F_, idx):
    T = F.interpolate(T, size=(F_.shape[2], F_.shape[3], F_.shape[4]), mode="trilinear", align_corners=False)
    S = F.interpolate(S, size=(F_.shape[2], F_.shape[3], F_.shape[4]), mode="trilinear", align_corners=False)

    fused = torch.cat([F_, T], dim=1) # channel: 256 + 64
    F_ = (self.conv[idx](fused))*S + F_

    if idx != 2:
      # upsample 2x
      F_ = self.up_conv[idx](F_)
      if self.shuffle_h:
          F_ = self.shuffle_h[idx](F_)
      if self.shuffle_w:
          F_ = self.shuffle_w[idx](F_)

    res = F_
    for rb in self.resblks[idx]:
      res = rb(res)
    res = self.res_tails[idx](res)
    F_ = F_ + res

    return F_



  def forward(self, lr, S, T_v1, T_v2, T_v3):

    F_ = self.SFE(lr)  # F 1x

    F_1 = self.stage(T_v3, S, F_, 0)
    F_2 = self.stage(T_v2, S, F_1, 1)
    F_3 = self.stage(T_v1, S, F_2, 2)

    F_1up = F.interpolate(F_1, size=F_3.shape[2:], mode='trilinear')
    F_2up = F.interpolate(F_2, size=F_3.shape[2:], mode='trilinear')

    multi = torch.cat([F_1up, F_2up, F_3], dim=1)
    # print(multi.shape)
    F_ = self.merge_conv(multi)

    F_ = self.final_conv(F_)
    return F_




class TTSR(nn.Module):
  """
  典型尺寸: sample: (B, C, T, H, W) = 4, 1, 16, 32, 128; ref: (B, C, T, H, W) = 4, 1, 16, 128, 128
  """
  def __init__(self, in_ch=1, feat_ch=64):
    super().__init__()
    self.name = "TTSR-basic"

    self.LTEVGG = LTEVGG(in_ch)
    self.SearchTransfer = SearchTransfer()
    self.Backbone = SFE(in_ch, feat_ch)
    self.mainnet = MainNet(feat_ch=64, out_ch=1, scale=(2,1))  # hard-coded
  def forward(self, lr, ref):

    B, _, T, H, W = lr.shape
    _, _, T1, H1, W1 = ref.shape
    scale_h = H / H1
    scale_w = W / W1

    lr_up = F.interpolate(lr, size=(T1, H1, W1), mode="trilinear", align_corners=False)
    ref_down = F.interpolate(ref, scale_factor=(1, scale_h, scale_w), mode="trilinear", align_corners=False)
    ref_down_up = F.interpolate(ref_down, size=(T1, H1, W1), mode="trilinear", align_corners=False)

    lrsr_v1, lrsr_v2, lrsr_v3 = self.LTEVGG(lr_up) # v1, v2, v3: /1, /2, /4
    refsr_v1, refsr_v2, refsr_v3 = self.LTEVGG(ref_down_up)
    ref_v1, ref_v2, ref_v3 = self.LTEVGG(ref)

    Q, K, V = lrsr_v3, refsr_v3, ref_v3 # (B, C, T, H1/4, W1/4)

    S, T_v1, T_v2, T_v3 = self.SearchTransfer(Q, K, ref_v1, ref_v2, ref_v3)

    output = self.mainnet(lr, S, T_v1, T_v2, T_v3 )

    return output




