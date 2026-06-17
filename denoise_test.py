import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from tifftool.load_tiff import load_tiff, write_tiff
from denoise_network import UNet3D


def parse_args():
    parser = argparse.ArgumentParser(description="Test 3D U-Net denoising on full images.")
    parser.add_argument("--denoise-weight", required=True, help="Path to denoise model checkpoint (.pth)")
    parser.add_argument("--samples", required=True, help="Path to LR sample images")
    parser.add_argument("--output_folder", default="~/tmp/result_denoise", help="Output folder")
    parser.add_argument("--num-dataset", type=int, default=1, help="Number of image pairs to process")
    parser.add_argument("--patch-t", type=int, default=16, help="Temporal patch size")
    parser.add_argument("--patch-y", type=int, default=32, help="Height patch size")
    parser.add_argument("--patch-x", type=int, default=128, help="Width patch size")
    parser.add_argument("--overlap-factor", type=float, default=0.5, help="Overlap factor for sliding window")
    parser.add_argument("--eval-frames", type=int, default=None, help="Only process first N frames")
    return parser.parse_args()


def make_coordinates(T, H, W, patch_t, patch_y, patch_x, overlap_factor):
    gap_t = int(patch_t * (1 - overlap_factor))
    gap_y = int(patch_y * (1 - overlap_factor))
    gap_x = int(patch_x * (1 - overlap_factor))

    cut_t = (patch_t - gap_t) // 2
    cut_y = (patch_y - gap_y) // 2
    cut_x = (patch_x - gap_x) // 2

    num_t = int(np.ceil((T - patch_t + gap_t) / gap_t))
    num_h = int(np.ceil((H - patch_y + gap_y) / gap_y))
    num_w = int(np.ceil((W - patch_x + gap_x) / gap_x))

    coords = []
    for z in range(num_t):
        for y in range(num_h):
            for x in range(num_w):
                coord = {}

                if z != num_t - 1:
                    coord['init_s'] = gap_t * z
                    coord['end_s'] = coord['init_s'] + patch_t
                else:
                    coord['init_s'] = T - patch_t
                    coord['end_s'] = T

                if y != num_h - 1:
                    coord['init_h'] = gap_y * y
                    coord['end_h'] = coord['init_h'] + patch_y
                else:
                    coord['init_h'] = H - patch_y
                    coord['end_h'] = H

                if x != num_w - 1:
                    coord['init_w'] = gap_x * x
                    coord['end_w'] = coord['init_w'] + patch_x
                else:
                    coord['init_w'] = W - patch_x
                    coord['end_w'] = W

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


def denoise_whole_image(raw, denoise_model, patch_t, patch_y, patch_x, overlap_factor, device):
    T, H, W = raw.shape
    coords = make_coordinates(T, H, W, patch_t, patch_y, patch_x, overlap_factor)

    denoised_full = np.zeros((T, H, W), dtype=np.float32)
    weight = np.zeros((T, H, W), dtype=np.float32)

    denoise_model.eval()

    with torch.no_grad():
        for coord in tqdm(coords, desc="denoise whole image"):
            patch = raw[coord['init_s']:coord['end_s'],
                        coord['init_h']:coord['end_h'],
                        coord['init_w']:coord['end_w']]
            patch_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(patch, 0), 0)).to(device)

            denoised_patch = denoise_model(patch_tensor).squeeze(0).cpu().numpy()

            pss, pes = coord['patch_start_s'], coord['patch_end_s']
            psh, peh = coord['patch_start_h'], coord['patch_end_h']
            psw, pew = coord['patch_start_w'], coord['patch_end_w']

            sss, ses = coord['stack_start_s'], coord['stack_end_s']
            ssh, seh = coord['stack_start_h'], coord['stack_end_h']
            ssw, sew = coord['stack_start_w'], coord['stack_end_w']

            crop = denoised_patch[0, pss:pes, psh:peh, psw:pew]
            denoised_full[sss:ses, ssh:seh, ssw:sew] += crop
            weight[sss:ses, ssh:seh, ssw:sew] += 1.0

    denoised_full = np.divide(denoised_full, weight, out=np.zeros_like(denoised_full), where=weight > 0)
    return denoised_full


def main():
    args = parse_args()
    args.output_folder = os.path.expanduser(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    denoise_model = UNet3D(in_channels=1, out_channels=1, f_maps=16).to(device)
    denoise_model.load_state_dict(torch.load(args.denoise_weight, map_location=device, weights_only=False))
    denoise_model.eval()
    for param in denoise_model.parameters():
        param.requires_grad = False

    files_raw = sorted([f for f in os.listdir(args.samples) if f.endswith(".tif")])
    num = min(args.num_dataset, len(files_raw)) if args.num_dataset > 0 else len(files_raw)

    for i in range(num):
        raw_name = files_raw[i]
        raw_path = os.path.join(args.samples, raw_name)

        print(f"Processing {raw_name} ...")
        raw = load_tiff(raw_path).astype(np.float32)

        if args.eval_frames is not None:
            raw = raw[:args.eval_frames, :, :]
            print(f"  Using first {raw.shape[0]} frames")

        denoised = denoise_whole_image(raw, denoise_model,
                                       args.patch_t, args.patch_y, args.patch_x,
                                       args.overlap_factor, device)

        basename = os.path.join(args.output_folder, raw_name.replace(".tif", ""))
        write_tiff(np.expand_dims(np.expand_dims(raw, 0), 0), basename + "_sample.tif")
        write_tiff(np.expand_dims(np.expand_dims(denoised, 0), 0), basename + "_denoised.tif")
        print(f"Saved {basename}_sample.tif and {basename}_denoised.tif")


if __name__ == "__main__":
    main()
