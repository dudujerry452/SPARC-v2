import torch
from network import TTSR
import torch.utils.data as torchdata
from dataloader import PatchDataset
from tifftool.load_tiff import write_tiff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "checkpoint/e5.pth"

net = TTSR(in_ch=1, feat_ch=64).to(device)
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

print(f"load {model_path} success")

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

with torch.no_grad():
    

  for sample, label in dataloader: 
    sample = sample.to(device) 
    label = label.to(device) 

    test_res = net(sample, label) 

    write_tiff(test_res[0][0].detach().cpu().numpy(), "test1.tif")
    break
