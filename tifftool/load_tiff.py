import argparse
import numpy as np
import tifffile
import os
import sys
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr


def load_tiff(path):
    """Load a TIFF file and return. might be different type"""
    img = tifffile.imread(path)
    # img = img.astype(np.int16)
    return img


def compute_tiff_summary(img):
    values = img.astype(np.float64, copy=False).ravel()
    values = values[np.isfinite(values)]

    summary = {
        "shape": tuple(img.shape),
        "dtype": str(img.dtype),
        "valid_count": int(values.size),
    }

    if values.size == 0:
        summary["status"] = "empty"
        return summary

    is_integer = np.issubdtype(img.dtype, np.integer)

    # compute numeric values (store as native python types)
    min_val = values.min()
    max_val = values.max()
    mean_val = float(values.mean())
    std_val = float(values.std())

    summary.update({
        "is_integer": bool(is_integer),
        "min": int(min_val) if is_integer else float(min_val),
        "max": int(max_val) if is_integer else float(max_val),
        "mean": mean_val,
        "std": std_val,
    })

    return summary


def print_tiff_summary(summary):
    if summary.get("status") == "empty":
        print("=" * 64)
        print(" TIFF summary ".center(64, "="))
        print("=" * 64)
        print(f"{'shape':>12}: {summary.get('shape')}")
        print(f"{'dtype':>12}: {summary.get('dtype')}")
        print(f"{'status':>12}: the content is empty!")
        return

    is_integer = summary.get("is_integer", False)

    def fmt(v):
        return f"{int(v)}" if is_integer else f"{v:.4f}"

    print(" TIFF summary ".center(64, "="))
    print(f"{'shape':>12}: {summary.get('shape')}")
    print(f"{'dtype':>12}: {summary.get('dtype')}")
    print(f"{'min':>12}: {fmt(summary.get('min'))}")
    print(f"{'max':>12}: {fmt(summary.get('max'))}")
    print(f"{'mean':>12}: {summary.get('mean'):.4f}")
    print(f"{'std':>12}: {summary.get('std'):.4f}")


def print_ascii_histogram(img, bins=20, width=40):
    values = img.astype(np.float64, copy=False).ravel()
    values = values[np.isfinite(values)]
    is_integer = np.issubdtype(img.dtype, np.integer)

    if values.size == 0:
        print("\n[Histogram] the content is empty! ")
        return

    counts, edges = np.histogram(values, bins=bins)
    max_count = counts.max()

    print(" histogram ".center(64, "-"))

    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        if is_integer:
            label = f"[{int(np.floor(left)):>8d}, {int(np.ceil(right)):>8d})"
        else:
            label = f"[{left:>10.4f}, {right:>10.4f})"
        print(f"{label} | {bar:<{width}} | {count}")

    hist = {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "bins": int(bins),
        "width": int(width),
        "is_integer": bool(is_integer),
        "total": int(values.size),
    }

    return hist



def check_compliance(tiff, summary, drange=1500):
  """
  正确的tiff: 动态范围0~1500, mean在70~100, 数据类型是uint16
  """
  s = summary 
  msgs = []
  flag = False

  nan_count = int(np.isnan(tiff.astype(np.float64, copy=False)).sum())
  if nan_count > 0:
          msgs.append(f"[fixable] failed: found {nan_count} NaN values, will replace with 0. ")
          flag = True

  # normalize dtype from summary (summary stores dtype as string)
  try:
      dtype = np.dtype(s["dtype"])
  except Exception:
      dtype = None

  # prepare a working copy for modifications when writeit=True
  t = tiff

  t = tiff.astype(np.float64, copy=True)
  # convert NaN -> 0 and infinities -> 0
  t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
  # set negative values to 0
  t[t < 0] = 0

  # Check for unusually small max (possible offset or wrong dtype interpretation)
  if s["max"] < 50:
      if dtype is not None and np.issubdtype(dtype, np.integer):
          msg = "[can't fix] failed: the data is int but too small! "
          raise RuntimeError(msg)
      msgs.append(f"[fixable] failed: tiff's max {s['max']} is too small(speculate its a float)! ")
      flag = True

      # scale to requested dynamic range
      t = t * (drange / float(s["max"])) if s["max"] != 0 else t

  if s["min"] > 32000:
      msgs.append(f"[fixable] failed: tiff's min {s['min']} proved that its base value is 32768! ")
      flag = True
      t = t - 32768

  if s["mean"] > 1000:
      msg = f"[can't fix] failed: tiff's mean {s['mean']} is not regular! "
      raise RuntimeError(msg)

  if dtype is None or not np.issubdtype(dtype, np.uint16):
      msgs.append(f"[fixable] failed: the type {s['dtype']} is not uint16! ")
      flag = True
      # ensure non-negative and finite before casting
      if not np.issubdtype(t.dtype, np.integer):
          t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
          t[t < 0] = 0
      t = t.astype(np.uint16)

  ret = "\n".join(msgs)
  return (flag, ret, t)

def load_correct_tiff(tiff_path):
    tiff = load_tiff(tiff_path) 
    s = compute_tiff_summary(tiff)
    _, _, ret = check_compliance(tiff, s)
    return ret 

def write_tiff(tiff, tiff_path): 
    tifffile.imwrite(tiff_path, tiff)
  
def add_noise(tiff, snr, gau_ratio, rng=None): 

    if rng is None: 
        rng = np.random.default_rng()

    tiff = tiff.astype(np.float64) 
    ret = np.zeros_like(tiff, dtype=np.uint16)
    snr = 10.0 ** (snr / 20.0) # 功率 -> 振幅

    for i, img in enumerate(tiff): 

        mu = img.mean()

        scale = (1 + gau_ratio) * (snr **2) / mu 
        photons = img * scale 
        photons_noisy = rng.poisson(photons).astype(np.float64) 
        g_std = np.sqrt(gau_ratio * photons.mean())

        # 高斯噪声
        photons_noisy += rng.normal(0.0, g_std, size=photons.shape)
        photons_noisy = np.maximum(photons_noisy, 0.0) 

        noise = photons_noisy / scale

        ret[i] = np.clip(noise, 0, 65535).astype(np.uint16) 

    return ret

def downsample_y(tiff, fac=4): 
    return tiff[:, ::fac, :]


if __name__ == "__main__": 

  parser = argparse.ArgumentParser(
      description="Evaluate reconstructed TIFF against ground truth (SPARC paper metrics)")
  parser.add_argument("tif", type=str, nargs="+", help="Path to ground truth TIFF stack(s)")
  args = parser.parse_args()

  if len(args.tif) == 2:
    try:
        from metrics import Mixed, SSIM, MAE, PSNR
    except ImportError:
        from tifftool.metrics import Mixed, SSIM, MAE

    a = np.squeeze(load_tiff(args.tif[1]).astype(np.float32))
    b = np.squeeze(load_tiff(args.tif[0]).astype(np.float32))

    print(a.shape, b.shape)

    print(f"PSNR: {PSNR(np.expand_dims(a, 0), np.expand_dims(b, 0)):.6f}")
    print(f"SSIM: {SSIM(np.expand_dims(a, 0), np.expand_dims(b, 0)):.6f}")
    print(f"MAE: {MAE(np.expand_dims(a, 0), np.expand_dims(b, 0)):.6f}")
    print(f"Mixed: {Mixed(np.expand_dims(a, 0), np.expand_dims(b, 0)):.6f}")
    sys.exit(0)

  tiff = load_tiff(args.tif[0])
  summary = compute_tiff_summary(tiff)
  print_tiff_summary(summary)
  hist = print_ascii_histogram(tiff)

  results = {"summary": summary, "histogram": hist}

  try:
      flag, info, neutif = check_compliance(tiff, summary, 1500)
  except RuntimeError as e:
      print(str(e))
      sys.exit(1)

  print(info)

  if flag:
      try:
          ans = input("Do you want to fix them now? (y/N): ").strip()
      except Exception:
          ans = ""

      if ans in ("y", "Y", "yes", "YES"):
          
          outpath = os.path.splitext(args.tif)[0] + "_standard.tif"
          try:
              tifffile.imwrite(outpath, neutif)
              print(f"Saved fixed TIFF to {outpath}")
          except Exception as e:
              print(f"Failed to save fixed TIFF: {e}")
      


