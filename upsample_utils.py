import numpy as np
import torch
from scipy.linalg import toeplitz


def upsample_matrix(frames, reference):
    """
    Custom upsampling using Toeplitz matrix.

    Args:
        frames: torch.Tensor of shape (B, 1, T, H, W), denoised LR frames.
        reference: torch.Tensor of shape (H*4, W) HR reference used for structure guidance.

    Returns:
        torch.Tensor of shape (B, 1, T, H*4, W) upsampled target.
    """
    frames = frames.detach().cpu().numpy().astype(np.float32)

    ref_min = reference.min()
    ref_max = reference.max()
    ratios = (reference - ref_min) / (ref_max - ref_min + 1e-8)
    ratios = ratios.detach().cpu().numpy().astype(np.float32)

    B, _, T, H, W_x = frames.shape
    rows = H * 4

    epsilon = 1e-8
    d = np.arange(1, rows + 1)
    w = 1.0 / (d + epsilon)
    W = toeplitz(w)
    W[W < 0.4] = 0.0

    output = np.zeros((B, 1, T, reference.shape[0], reference.shape[1]), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            image = frames[b, 0, t, :, :]
            row_base = np.arange(reference.shape[0]) // 4

            X_up_new = np.zeros_like(ratios)
            for i in range(H * 4):
                row = row_base[i]
                if i < 4:
                    idx = np.arange(0, 8)
                    img_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1])
                elif i < rows - 4:
                    idx_start = 4 * (row - 1) + 0
                    idx = np.arange(idx_start, idx_start + 12)
                    img_rows = np.array([row - 1] * 4 + [row] * 4 + [row + 1] * 4)
                else:
                    idx = np.arange(rows - 8, rows)
                    img_rows = np.array([int(rows / 4 - 2)] * 4 + [int(rows / 4 - 1)] * 4)

                weights = W[idx, i]
                # weights = weights / (weights.sum() + epsilon)  # normalize to avoid signal amplification/diffusion
                img_vals = image[img_rows, :]
                X_up_new[i, :] = np.sum(weights[:, None] * img_vals, axis=0) * ratios[i, :]

            output[b, 0, t, :, :] = X_up_new

    output = torch.from_numpy(output).type(torch.float32)
    return output
