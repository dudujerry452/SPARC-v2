import argparse
import csv
import os

import numpy as np
import torch
import torch.utils.data as torchdata

import sys

from tifftool.load_tiff import write_tiff
from dataloader import PatchDataset
from tifftool.metrics import evaluate_batch
from network import TTSR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TTSR checkpoint on a dataset.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--samples", required=True, help="Path to LR sample images")
    parser.add_argument("--labels", required=True, help="Path to HR label images")
    parser.add_argument("--output_folder", default=".", help="Path to HR label images")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path to save per-batch metrics")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TTSR(in_ch=1, feat_ch=64).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    net.eval()

    dataset = PatchDataset(args.samples, args.labels, 1)

    if args.num_samples is not None:
        indices = list(range(min(args.num_samples, len(dataset))))
        subset = torchdata.Subset(dataset, indices)
    else:
        subset = dataset

    dataloader = torchdata.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    metric_keys = ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "SSIE", "Pearson"]
    batch_metrics = []

    with torch.no_grad():
        for sample, label in dataloader:
            sample = sample.to(device)
            label = label.to(device)

            sr = net(sample, label)
            metrics = evaluate_batch(sr, label)
            batch_metrics.append(metrics)

        for sample, label in dataloader: 
            sample = sample.to(device)
            label = label.to(device)

            sr = net(sample, label)

            basename = os.path.join(args.output_folder, net.name) 
            write_tiff(sample.detach().cpu().numpy(), basename+"_sample.tif")
            write_tiff(label.detach().cpu().numpy(), basename+"_label.tif")
            write_tiff(sr.detach().cpu().numpy(), basename+"_sr.tif")

            break 

    if not batch_metrics:
        print("No samples evaluated.")
        return

    # Aggregate
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in batch_metrics]
        # Ignore inf for PSNR mean/std to avoid nan
        finite_values = [v for v in values if np.isfinite(v)]
        aggregated[key] = {
            "mean": float(np.mean(finite_values)) if finite_values else float("inf"),
            "std": float(np.std(finite_values)) if finite_values else 0.0,
        }

    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print(f"Labels: {args.labels}")
    print(f"Evaluated batches: {len(batch_metrics)}")
    print("-" * 40)
    for key in metric_keys:
        mean = aggregated[key]["mean"]
        std = aggregated[key]["std"]
        print(f"{key:10s}: {mean:.6f} +/- {std:.6f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_keys)
            writer.writeheader()
            writer.writerows(batch_metrics)
        print(f"Per-batch metrics saved to {args.output_csv}")


if __name__ == "__main__":
    main()
