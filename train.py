import torch 
from torch import nn 
import torch.utils.data as torchdata
from dataloader import PatchDataset
from tifftool.load_tiff import write_tiff
import torch.nn.functional as F 
from network import TTSR
from visualizer.hook import start_visualizer, send_tensor
import os, argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TTSR checkpoint on a dataset.")
    parser.add_argument("--samples", required=True, help="Path to LR sample images")
    parser.add_argument("--labels", required=True, help="Path to HR label images")
    parser.add_argument("--checkpoint_folder", default=".", help="Path to HR label images")
    parser.add_argument("--epoch", type=int, default=10, help="Batch size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-dataset", type=int, default=1, help="Maximum number of dataset(tif pair) to evaluate")
    parser.add_argument("--patch-t", type=int, default=16, help="Temporal patch size")
    parser.add_argument("--patch-y", type=int, default=32, help="Height patch size")
    parser.add_argument("--patch-x", type=int, default=128, help="Width patch size")
    parser.add_argument("--visualize", action="store_true", help="enable html visualize")
    return parser.parse_args()

args = parse_args()

dataset = PatchDataset(
   args.samples,
   args.labels,
   args.num_dataset,
   patch_t=args.patch_t,
   patch_y=args.patch_y,
   patch_x=args.patch_x,
   use_video=True)

dataloader = torchdata.DataLoader(
   dataset, 
   batch_size=args.batch_size, 
   shuffle=True, 
   num_workers=0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.visualize: 
  start_visualizer(port=5000)

net = TTSR(in_ch=1, feat_ch=64).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4) 
criterion = nn.L1Loss() 

os.makedirs(args.checkpoint_folder, exist_ok=True)

for epoch in range(args.epoch): 
  for idx, (sample, label) in enumerate(dataloader): 
    sample = sample.to(device) 
    label = label.to(device) 

    sr = net(sample, label) # use label as ref
    # send_tensor('sr', sr[0], title=f'epoch{epoch} idx{idx} sr')
    loss = criterion(sr, label) 

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if idx % 10 == 0: 
      print(f"epoch {epoch} idx {idx} loss = {loss}")
  torch.save(net.state_dict(), os.path.join(args.checkpoint_folder, f"{net.name}_epoch{epoch}.pth"))


