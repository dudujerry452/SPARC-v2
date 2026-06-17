import torch
from torch import nn
import torch.utils.data as torchdata
from dataloader import PatchDataset
from tifftool.load_tiff import write_tiff
import torch.nn.functional as F
from network import TTSR
from denoise_network import UNet3D
from upsample_utils import upsample_matrix
from visualizer.hook import start_visualizer, send_tensor
from tqdm import tqdm
import time
import os, argparse, glob, re


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D TTSR with denoise-based pseudo-reference.")
    parser.add_argument("--samples", required=True, help="Path to LR sample images")
    parser.add_argument("--labels", required=True, help="Path to HR label images")
    parser.add_argument("--groundtruth", type=str, default=None, help="Path to 3D ground truth images (optional, for validation only)")
    parser.add_argument("--checkpoint_folder", default=".", help="Path to save checkpoints")
    parser.add_argument("--epoch", type=int, default=10, help="Number of TTSR training epochs")
    parser.add_argument("--denoise-epoch", type=int, default=5, help="Number of denoise pre-training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for TTSR training")
    parser.add_argument("--denoise-batch-size", type=int, default=4, help="Batch size for denoise training")
    parser.add_argument("--num-dataset", type=int, default=1, help="Maximum number of dataset (tif pair) to load")
    parser.add_argument("--patch-t", type=int, default=16, help="Temporal patch size")
    parser.add_argument("--patch-y", type=int, default=32, help="Height patch size")
    parser.add_argument("--patch-x", type=int, default=128, help="Width patch size")
    parser.add_argument("--denoise-weight", type=str, default=None, help="Path to pretrained denoise model (skip denoise training if provided)")
    parser.add_argument("--visualize", action="store_true", help="enable html visualize")
    parser.add_argument("--continue-train", action="store_true", help="enable html visualize")
    return parser.parse_args()


args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.visualize:
  start_visualizer(port=5000)

os.makedirs(args.checkpoint_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage 1: train / load denoise model
# ---------------------------------------------------------------------------
denoise_model = UNet3D(in_channels=1, out_channels=1, f_maps=16).to(device)
denoise_weight_path = args.denoise_weight if args.denoise_weight else os.path.join(args.checkpoint_folder, "denoise_init.pth")

if args.denoise_weight and os.path.exists(args.denoise_weight):
    denoise_model.load_state_dict(torch.load(args.denoise_weight, map_location=device, weights_only=False))
    print(f"Loaded pretrained denoise model from {args.denoise_weight}")
elif os.path.exists(denoise_weight_path):
    denoise_model.load_state_dict(torch.load(denoise_weight_path, map_location=device, weights_only=False))
    print(f"Loaded pretrained denoise model from {denoise_weight_path}")
else:
    print(f"Pre-training denoise model for {args.denoise_epoch} epochs...")

    denoise_dataset = PatchDataset(
       args.samples,
       args.labels,
       args.num_dataset,
       patch_t=args.patch_t,
       patch_y=args.patch_y,
       patch_x=args.patch_x,
       use_video=True)

    denoise_loader = torchdata.DataLoader(
       denoise_dataset,
       batch_size=args.denoise_batch_size,
       shuffle=True,
       num_workers=0
    )

    optimizer_d = torch.optim.Adam(denoise_model.parameters(), lr=1e-4)
    criterion_d = nn.L1Loss()

    for epoch in range(args.denoise_epoch):
        denoise_model.train()
        pbar = tqdm(enumerate(denoise_loader), total=len(denoise_loader),
                    desc=f"denoise epoch {epoch}/{args.denoise_epoch}")
        start_time = time.time()
        for idx, batch in pbar:
            sample = batch[0].to(device)

            noisy_even = sample[:, :, 0::2, :, :]
            noisy_odd  = sample[:, :, 1::2, :, :]

            denoised, _ = denoise_model(noisy_even)

            # denoise target: odd frames
            loss = criterion_d(denoised, noisy_odd)

            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

            if idx % 10 == 0:
                elapsed = time.time() - start_time
                it_per_s = (idx + 1) / elapsed if elapsed > 0 else 0
                remain = (len(denoise_loader) - idx - 1) / it_per_s if it_per_s > 0 else 0
                pbar.set_postfix(loss=f"{loss.item():.4f}", remain=f"{remain:.0f}s")

    torch.save(denoise_model.state_dict(), denoise_weight_path)
    print(f"Denoise model saved to {denoise_weight_path}")

# ---------------------------------------------------------------------------
# Stage 2: train TTSR with frozen denoise-generated reference
# ---------------------------------------------------------------------------
print("Training TTSR with frozen denoise model...")

denoise_model.eval()
for param in denoise_model.parameters():
    param.requires_grad = False

dataset = PatchDataset(
   args.samples,
   args.labels,
   args.groundtruth,
   args.num_dataset,
   patch_t=args.patch_t,
   patch_y=args.patch_y,
   patch_x=args.patch_x,
   use_video=True,
   use_range=(0.0, 1.0)
   )

dataloader = torchdata.DataLoader(
   dataset,
   batch_size=args.batch_size,
   shuffle=True,
   num_workers=0
)

net = TTSR(in_ch=1, feat_ch=64).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.L1Loss()

if args.continue_train:
    checkpoint_paths = glob.glob(os.path.join(args.checkpoint_folder, net.name + "_epoch*.pth"))
    if checkpoint_paths:
        def extract_epoch(path):
            match = re.search(r"_epoch(\d+)\.pth$", os.path.basename(path))
            return int(match.group(1)) if match else -1

        latest = max(checkpoint_paths, key=extract_epoch)
        start_epoch = extract_epoch(latest)+1
        print(f"continue from checkpoint {latest}, epoch {start_epoch}")
        net.load_state_dict(torch.load(latest, map_location=device, weights_only=False))
    else:
        start_epoch = 0
else:
    start_epoch = 0

for epoch in range(start_epoch, args.epoch):
    net.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"TTSR epoch {epoch}/{args.epoch}")
    start_time = time.time()

    for idx, batch in pbar:
        sample = batch[0].to(device)
        label = batch[1].to(device)
        gt = batch[2].to(device) if len(batch) > 2 else None

        T = sample.size(2)
        # ref for TTSR: broadcast 2D label to T frames
        ref = label.unsqueeze(2).expand(-1, -1, T, -1, -1)

        with torch.no_grad():
            denoised, _ = denoise_model(sample)
            # loss target: denoised + upsample_matrix (sequence pseudo-label)
            target_list = []
            for b in range(denoised.size(0)):
                target_b = upsample_matrix(denoised[b:b+1], label[b, 0, :, :])
                target_list.append(target_b)
            target = torch.cat(target_list, dim=0).to(device)

        sr = net(sample, ref)
        loss = criterion(sr, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            it_per_s = (idx + 1) / elapsed if elapsed > 0 else 0
            remain = (len(dataloader) - idx - 1) / it_per_s if it_per_s > 0 else 0
            pbar.set_postfix(loss=f"{loss.item():.4f}", remain=f"{remain:.0f}s")

        if idx % 100 == 0:
            write_tiff(ref.detach().cpu().numpy(), os.path.join(args.checkpoint_folder, f"ref_epoch{epoch}_batch{idx}.tif"))
            write_tiff(target.detach().cpu().numpy(), os.path.join(args.checkpoint_folder, f"target_epoch{epoch}_batch{idx}.tif"))
            if gt is not None:
                write_tiff(gt.detach().cpu().numpy(), os.path.join(args.checkpoint_folder, f"gt_epoch{epoch}_batch{idx}.tif"))

    torch.save(net.state_dict(), os.path.join(args.checkpoint_folder, f"{net.name}_epoch{epoch}.pth"))

print("Training complete.")
