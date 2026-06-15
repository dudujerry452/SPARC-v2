import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_hw_array(tensor):
    """Convert a torch/ numpy tensor to a 2D (H, W) numpy array."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    # Squeeze leading singleton dimensions until we get (H, W)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"Expected a (H, W) tensor after squeezing, got shape {arr.shape}")

    return arr


def _min_max_normalize(arr):
    """Normalize array to [0, 1] using min-max scaling."""
    arr = arr.astype(np.float32)
    amin = arr.min()
    amax = arr.max()
    if amax - amin < 1e-8:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)


def tensor_to_image(tensor, title=None, cmap='gray'):
    """Render a 2D tensor as a Matplotlib figure and return a base64 PNG URI."""
    arr = _to_hw_array(tensor)
    arr = _min_max_normalize(arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis('off')
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'
