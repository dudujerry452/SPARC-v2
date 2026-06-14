import torch 
from torch import nn 
import torch.utils.data as torchdata
from dataloader import PatchDataset
import torch.nn.functional as F 


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
  
  def forward(self, x):  # lr is 低质量图像
    return self.body(x)


class TTSR(nn.Module): 
  """
  典型尺寸: sample: (B, C, H, W) = 4, 1, 32, 128; ref: (B, C, H, W) = 4, 1, 128, 128
  """
  def __init__(self):
    super().__init__()
    self.LTE = LTE() 
  def forward(self, lr, ref): 
    _, _, H, W = lr.shape 
    _, _, H1, W1 = ref.shape 
    scale_h = H / H1
    scale_w = W / W1 
    lr_up = F.interpolate(lr, size=(H1, W1), mode="bicubic", align_corners=False)
    ref_down = F.interpolate(ref, scale_factor=(scale_h, scale_w), mode="bicubic", align_corners=False)
    ref_down_up = F.interpolate(ref_down, size=(H1, W1), mode="bicubic", align_corners=False)

    Q = self.LTE(lr_up) 
    K = self.LTE(ref_down_up) 
    V = self.LTE(ref) 

    return Q, K, V


  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PatchDataset(
   "../zzydata/dataset_st/samples", 
   "../zzydata/dataset_st/labels", 
   2)

dataloader = torchdata.DataLoader(
   dataset, 
   batch_size=4, 
   shuffle=True, 
   num_workers=0
)

net = TTSR().to(device)

for idx, (sample, label) in enumerate(dataloader): 
  sample = sample.to(device)
  label = label.to(device)
  Q, K, V = net(sample, label) # 这里label当作ref用
  print(Q.shape, K.shape, V.shape)




