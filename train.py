import torch 
from torch import nn 
import torch.utils.data as torchdata
from dataloader import PatchDataset
from tifftool.load_tiff import write_tiff
import torch.nn.functional as F 
from network import TTSR
import os



dataset = PatchDataset(
   "../zzydata/dataset_st/samples", 
   "../zzydata/dataset_st/labels", 
   1)

dataloader = torchdata.DataLoader(
   dataset, 
   batch_size=4, 
   shuffle=True, 
   num_workers=0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = TTSR(in_ch=1, feat_ch=64).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) 
criterion = nn.L1Loss() 

os.makedirs("checkpoint", exist_ok=True)

for epoch in range(10): 
  for idx, (sample, label) in enumerate(dataloader): 
    sample = sample.to(device) 
    label = label.to(device) 

    sr = net(sample, label) # use label as ref  
    loss = criterion(sr, label) 

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if idx % 10 == 0: 
      print(f"epoch {epoch} idx {idx} loss = {loss}")
  if epoch % 5 == 0: 
    torch.save(net.state_dict(), f"checkpoint/e{epoch}.pth")



