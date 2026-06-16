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
    parser.add_argument("--metric", action="store_true", help="display metric info")
    parser.add_argument("--sample_id", type=int, default=21, help="the sample id write to tiff")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TTSR(in_ch=1, feat_ch=64).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    net.eval()

    dataset = PatchDataset(args.samples, args.labels, 1, use_video=True, use_random=False)

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

    metric_keys = ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "SSIE", "Pearson", "Mixed"]
    batch_metrics = []

    with torch.no_grad():

        if args.metric is True: 
            for sample, label in dataloader:

                sample = sample[:, :, 0:1, :, :].squeeze(axis=1) # 暂时丢弃3D数据, 只使用其第一帧
                sample = sample.to(device)
                label = label.to(device)

                sr = net(sample, label)
                metrics = evaluate_batch(sr, label)
                batch_metrics.append(metrics)

        for idx, (sample, label) in enumerate(dataloader): 
            
            if idx == args.sample_id: 

                label = label.to(device)
                sample = sample.to(device)
                srs = []
                
                for idx in range(sample.shape[2]): 
                    frame = sample[:, :, idx:(idx+1), :, :].squeeze(axis=1)
                    sr = net(frame, label)
                    srs.append(sr.detach().cpu().numpy()) 

                srs = np.stack(srs) 

                basename = os.path.join(args.output_folder, net.name) 
                write_tiff(sample.detach().cpu().numpy(), basename+"_sample.tif")
                write_tiff(label.detach().cpu().numpy(), basename+"_label.tif")
                write_tiff(srs, basename+"_sr.tif")
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
