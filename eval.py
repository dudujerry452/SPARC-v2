import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

from tifftool.load_tiff import load_tiff, write_tiff
from dataloader import PatchDataset
from network import TTSR
from denoise_network import UNet3D
from tifftool.metrics import evaluate_batch, Mixed


def parse_args():
    parser = argparse.ArgumentParser(description="Inference 3D TTSR on full images or patches.")
    parser.add_argument("--model", required=True, help="Path to TTSR checkpoint (.pth)")
    parser.add_argument("--denoise-weight", required=True, help="Path to denoise model checkpoint (.pth)")
    parser.add_argument("--samples", required=True, help="Path to LR sample images")
    parser.add_argument("--labels", required=True, help="Path to HR label images")
    parser.add_argument("--groundtruth", type=str, default=None, help="Path to 3D ground truth images (optional)")
    parser.add_argument("--output_folder", default="./inference_output", help="Output folder")
    parser.add_argument("--num-dataset", type=int, default=1, help="Number of image pairs to process")
    parser.add_argument("--patch-t", type=int, default=16, help="Temporal patch size")
    parser.add_argument("--patch-y", type=int, default=32, help="Height patch size")
    parser.add_argument("--patch-x", type=int, default=128, help="Width patch size")
    parser.add_argument("--overlap-factor", type=float, default=0.5, help="Overlap factor for whole-image mode")
    parser.add_argument("--whole-image", action="store_true", help="Enable whole-image inference mode")
    parser.add_argument("--eval-frames", type=int, default=None, help="Only evaluate first N frames in whole-image mode")
    parser.add_argument("--sample-id", type=int, default=0, help="Which image pair to process in patch mode")
    return parser.parse_args()


def load_image_pair(raw_path, label_path):
    raw = load_tiff(raw_path).astype(np.float32)
    label = load_tiff(label_path).astype(np.float32)
    return raw, label


def make_coordinates(T, H, W, patch_t, patch_y, patch_x, overlap_factor):
    gap_t = int(patch_t * (1 - overlap_factor))
    gap_y = int(patch_y * (1 - overlap_factor))
    gap_x = int(patch_x * (1 - overlap_factor))

    cut_t = (patch_t - gap_t) // 2
    cut_y = (patch_y - gap_y) // 2
    cut_x = (patch_x - gap_x) // 2

    num_t = math.ceil((T - patch_t + gap_t) / gap_t)
    num_h = math.ceil((H - patch_y + gap_y) / gap_y)
    num_w = math.ceil((W - patch_x + gap_x) / gap_x)

    coords = []
    for z in range(num_t):
        for y in range(num_h):
            for x in range(num_w):
                coord = {}

                # read position
                if z != num_t - 1:
                    init_s = gap_t * z
                    end_s = init_s + patch_t
                else:
                    init_s = T - patch_t
                    end_s = T

                if y != num_h - 1:
                    init_h = gap_y * y
                    end_h = init_h + patch_y
                else:
                    init_h = H - patch_y
                    end_h = H

                if x != num_w - 1:
                    init_w = gap_x * x
                    end_w = init_w + patch_x
                else:
                    init_w = W - patch_x
                    end_w = W

                coord['init_s'] = init_s
                coord['end_s'] = end_s
                coord['init_h'] = init_h
                coord['end_h'] = end_h
                coord['init_w'] = init_w
                coord['end_w'] = end_w

                # write position and crop position
                if z == 0:
                    coord['stack_start_s'] = 0
                    coord['stack_end_s'] = patch_t - cut_t
                    coord['patch_start_s'] = 0
                    coord['patch_end_s'] = patch_t - cut_t
                elif z == num_t - 1:
                    coord['stack_start_s'] = T - patch_t + cut_t
                    coord['stack_end_s'] = T
                    coord['patch_start_s'] = cut_t
                    coord['patch_end_s'] = patch_t
                else:
                    coord['stack_start_s'] = gap_t * z + cut_t
                    coord['stack_end_s'] = gap_t * z + patch_t - cut_t
                    coord['patch_start_s'] = cut_t
                    coord['patch_end_s'] = patch_t - cut_t

                if y == 0:
                    coord['stack_start_h'] = 0
                    coord['stack_end_h'] = patch_y - cut_y
                    coord['patch_start_h'] = 0
                    coord['patch_end_h'] = patch_y - cut_y
                elif y == num_h - 1:
                    coord['stack_start_h'] = H - patch_y + cut_y
                    coord['stack_end_h'] = H
                    coord['patch_start_h'] = cut_y
                    coord['patch_end_h'] = patch_y
                else:
                    coord['stack_start_h'] = gap_y * y + cut_y
                    coord['stack_end_h'] = gap_y * y + patch_y - cut_y
                    coord['patch_start_h'] = cut_y
                    coord['patch_end_h'] = patch_y - cut_y

                if x == 0:
                    coord['stack_start_w'] = 0
                    coord['stack_end_w'] = patch_x - cut_x
                    coord['patch_start_w'] = 0
                    coord['patch_end_w'] = patch_x - cut_x
                elif x == num_w - 1:
                    coord['stack_start_w'] = W - patch_x + cut_x
                    coord['stack_end_w'] = W
                    coord['patch_start_w'] = cut_x
                    coord['patch_end_w'] = patch_x
                else:
                    coord['stack_start_w'] = gap_x * x + cut_x
                    coord['stack_end_w'] = gap_x * x + patch_x - cut_x
                    coord['patch_start_w'] = cut_x
                    coord['patch_end_w'] = patch_x - cut_x

                coords.append(coord)

    return coords


def inference_whole_image(raw, label, net, denoise_model, patch_t, patch_y, patch_x, overlap_factor, device):
    T, H, W = raw.shape
    H4, W4 = label.shape  # label is (H*4, W)
    assert H4 == H * 4

    coords = make_coordinates(T, H, W, patch_t, patch_y, patch_x, overlap_factor)

    sr_full = np.zeros((T, H * 4, W4), dtype=np.float32)
    weight = np.zeros((T, H * 4, W4), dtype=np.float32)

    denoise_model.eval()
    net.eval()

    with torch.no_grad():
        for coord in tqdm(coords, desc="whole image inference"):
            patch = raw[coord['init_s']:coord['end_s'],
                        coord['init_h']:coord['end_h'],
                        coord['init_w']:coord['end_w']]
            patch_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(patch, 0), 0)).to(device)  # (1, 1, T, H, W)

            # ref: crop corresponding HR label patch and broadcast to T
            ref_patch = label[coord['init_h'] * 4:coord['end_h'] * 4,
                              coord['init_w']:coord['end_w']]
            ref_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(ref_patch, 0), 0)).to(device)  # (1, 1, H*4, W)
            Tp = patch_tensor.size(2)
            ref = ref_tensor.expand(1, 1, Tp, -1, -1)

            denoised_patch = denoise_model(patch_tensor)
            sr_patch = net(denoised_patch, ref).squeeze(0).cpu().numpy()  # (1, T, H*4, W)

            pss, pes = coord['patch_start_s'], coord['patch_end_s']
            psh, peh = coord['patch_start_h'] * 4, coord['patch_end_h'] * 4
            psw, pew = coord['patch_start_w'], coord['patch_end_w']

            sss, ses = coord['stack_start_s'], coord['stack_end_s']
            ssh, seh = coord['stack_start_h'] * 4, coord['stack_end_h'] * 4
            ssw, sew = coord['stack_start_w'], coord['stack_end_w']

            sr_crop = sr_patch[0, pss:pes, psh:peh, psw:pew]
            sr_full[sss:ses, ssh:seh, ssw:sew] += sr_crop
            weight[sss:ses, ssh:seh, ssw:sew] += 1.0

    # avoid division by zero
    sr_full = np.divide(sr_full, weight, out=np.zeros_like(sr_full), where=weight > 0)
    return sr_full


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TTSR(in_ch=1, feat_ch=64).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=False))
    net.eval()

    denoise_model = UNet3D(in_channels=1, out_channels=1, f_maps=16).to(device)
    denoise_model.load_state_dict(torch.load(args.denoise_weight, map_location=device, weights_only=False))
    denoise_model.eval()
    for param in denoise_model.parameters():
        param.requires_grad = False

    if args.whole_image:
        # load raw/label pairs directly
        files_raw = sorted([f for f in os.listdir(args.samples) if f.endswith(".tif")])
        files_label = sorted([f for f in os.listdir(args.labels) if f.endswith("_label.tif")])

        file_pair = []
        for f in files_raw:
            label_name = os.path.splitext(f)[0] + "_label.tif"
            if label_name in files_label:
                file_pair.append((f, label_name))

        num = min(args.num_dataset, len(file_pair)) if args.num_dataset > 0 else len(file_pair)

        for i in range(num):
            raw_name, label_name = file_pair[i]
            raw_path = os.path.join(args.samples, raw_name)
            label_path = os.path.join(args.labels, label_name)

            print(f"Processing {raw_name} ...")
            raw, label = load_image_pair(raw_path, label_path)

            if args.eval_frames is not None:
                raw = raw[:args.eval_frames, :, :]
                print(f"  Using first {raw.shape[0]} frames")

            sr = inference_whole_image(raw, label, net, denoise_model,
                                       args.patch_t, args.patch_y, args.patch_x,
                                       args.overlap_factor, device)

            basename = os.path.join(args.output_folder, raw_name.replace(".tif", "") + "_sr")
            write_tiff(np.expand_dims(sr, 0), basename + ".tif")
            print(f"Saved {basename}.tif")

            if args.groundtruth is not None:
                gt_name = os.path.splitext(raw_name)[0] + ".tif"
                gt_path = os.path.join(args.groundtruth, gt_name)
                if os.path.exists(gt_path):
                    gt = load_tiff(gt_path).astype(np.float32)
                    if args.eval_frames is not None:
                        gt = gt[:args.eval_frames, :, :]
                    gt_tensor = torch.from_numpy(np.expand_dims(gt, 0)).unsqueeze(0).to(device)
                    sr_tensor = torch.from_numpy(np.expand_dims(sr, 0)).unsqueeze(0).to(device)
                    metrics = evaluate_batch(sr_tensor, gt_tensor)
                    print(f"  Mixed (vs GT): {metrics['Mixed']:.6f}")
                else:
                    print(f"  Groundtruth not found at {gt_path}")

    else:
        # patch mode: same as eval sample writing
        dataset = PatchDataset(args.samples, args.labels, args.groundtruth, args.num_dataset,
                               patch_t=args.patch_t, patch_y=args.patch_y, patch_x=args.patch_x,
                               use_video=True, use_random=False)
        sample, label, gt = dataset[args.sample_id]
        sample = sample.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        T = sample.size(2)
        ref = label.unsqueeze(2).expand(-1, -1, T, -1, -1)

        with torch.no_grad():
            denoised = denoise_model(sample)
            sr = net(denoised, ref)

        print(f"Patch {args.sample_id} Mixed metric:")
        if gt is not None:
            gt = gt.unsqueeze(0).to(device)
            metrics = evaluate_batch(sr, gt)
            print(f"  Mixed (vs GT): {metrics['Mixed']:.6f}")
        else:
            print("  No groundtruth provided")

        basename = os.path.join(args.output_folder, "patch_sr")
        write_tiff(sr.detach().cpu().numpy(), basename + ".tif")
        write_tiff(sample.detach().cpu().numpy(), basename.replace("_sr", "_sample") + ".tif")
        write_tiff(label.detach().cpu().numpy(), basename.replace("_sr", "_label") + ".tif")
        if gt is not None:
            write_tiff(gt.detach().cpu().numpy(), basename.replace("_sr", "_gt") + ".tif")
        print(f"Saved patch results to {args.output_folder}")


if __name__ == "__main__":
    main()
