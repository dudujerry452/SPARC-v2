import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr


def _to_numpy(x):
    """Convert torch.Tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _batch_to_samples(pred, target):
    """Convert batched tensors to list of single-sample arrays."""
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} does not match target shape {target.shape}")
    if pred.ndim == 5:
        return [pred[i] for i in range(pred.shape[0])], [target[i] for i in range(target.shape[0])]
    elif pred.ndim == 4:
        return [pred], [target]
    else:
        raise ValueError(f"Expected 4D or 5D input, got {pred.ndim}D")


def _mean_metric(metric_fn, pred, target, **kwargs):
    """Apply a single-sample metric to each sample and return the mean."""
    preds, targets = _batch_to_samples(pred, target)
    values = [metric_fn(p, t, **kwargs) for p, t in zip(preds, targets)]
    return float(np.mean(values))


def _data_range(pred, target):
    """Infer data range from the inputs."""
    pred = _to_numpy(pred)
    target = _to_numpy(target)
    return float(max(np.max(pred), np.max(target)))


def MAE(pred, target):
    """Mean Absolute Error."""
    return _mean_metric(lambda p, t: np.mean(np.abs(p - t)), pred, target)


def MSE(pred, target):
    """Mean Squared Error."""
    return _mean_metric(lambda p, t: np.mean((p - t) ** 2), pred, target)


def RMSE(pred, target):
    """Root Mean Squared Error."""
    return float(np.sqrt(MSE(pred, target)))


def PSNR(pred, target, data_range=None):
    """Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = _data_range(pred, target)

    def _psnr(p, t):
        mse = np.mean((p - t) ** 2)
        if mse == 0:
            return float("inf")
        return 10.0 * np.log10(data_range ** 2 / mse)

    return _mean_metric(_psnr, pred, target)


def SSIM(pred, target, data_range=None):
    """Structural Similarity Index."""
    if data_range is None:
        data_range = _data_range(pred, target)

    def _ssim(p, t):
        if p.ndim == 4 and p.shape[0] == 1:
            p = p[0]
            t = t[0]
        if p.ndim == 3:
            return np.mean([ssim(p[i], t[i], data_range=data_range) for i in range(p.shape[0])])
        return ssim(p, t, data_range=data_range)

    return _mean_metric(_ssim, pred, target)


def SSIE(pred, target, data_range=None):
    """Structural Similarity Index Error = 1 - SSIM."""
    return 1.0 - SSIM(pred, target, data_range=data_range)


def Pearson(pred, target):
    """Pearson correlation coefficient."""
    def _pearson(p, t):
        p = p.flatten()
        t = t.flatten()
        return pearsonr(p, t)[0]

    return _mean_metric(_pearson, pred, target)



def Mixed(pred, target, alp=0.85):
    """ alp * (1-SSIM) + (1-alp) * MAE"""
    return alp * SSIE(pred, target) + (1-alp) * MAE(pred, target )


def evaluate_batch(pred, target):
    """
    Evaluate a batch of predictions against targets.

    Args:
        pred: torch.Tensor or numpy array of shape (B, C, T, H, W) or (C, T, H, W).
        target: same shape as pred.

    Returns:
        dict with MAE, MSE, RMSE, PSNR, SSIM, SSIE, Pearson.
    """
    data_range = _data_range(pred, target)
    return {
        "MAE": MAE(pred, target),
        "MSE": MSE(pred, target),
        "RMSE": RMSE(pred, target),
        "PSNR": PSNR(pred, target, data_range=data_range),
        "SSIM": SSIM(pred, target, data_range=data_range),
        "SSIE": SSIE(pred, target, data_range=data_range),
        "Pearson": Pearson(pred, target),
        "Mixed": Mixed(pred, target)
    }
