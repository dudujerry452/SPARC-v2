import numpy as np 
from load_tiff import load_correct_tiff, add_noise, downsample_y
import os, argparse, tifffile

if __name__ == "__main__": 
  parser = argparse.ArgumentParser(
        description="generate dataset from clean tiff file")
  parser.add_argument("tif_folder", type=str, help="Path to ground truth TIFF stack")
  parser.add_argument("-o", type=str, help="Path to output folder")
  parser.add_argument('--snr', type=float, default=5.0, help='snr')
  parser.add_argument('--ds', type=int, default=4, help='downsample ratio')
  parser.add_argument('--dryrun', action='store_true', help='turn on dryrun')

  args = parser.parse_args()

  files = [os.path.join(args.tif_folder, f) for f in os.listdir(args.tif_folder) if os.path.isfile(os.path.join(args.tif_folder, f)) and f.endswith(".tif")]
  
  output_folder = args.o 
  if output_folder is None: 
    output_folder = args.tif_folder
  if os.path.isdir(output_folder) is False: 
    raise LookupError(f"output dir {output_folder} is not existed! ")  


  sample_folder = os.path.join(output_folder, "samples") 
  label_folder = os.path.join(output_folder, "labels") 

  os.makedirs(sample_folder, exist_ok=True) 
  os.makedirs(label_folder, exist_ok=True) 

  # suffix = f"_s{args.snr:.1f}d{args.ds}".replace(".", "p")

  if args.dryrun is True: 
    print("(dryrun)")

  for f in files: 

    basename = os.path.splitext(os.path.basename(f))[0] 
    basename_label = basename+ "_label" 
    
    write_path = os.path.join(sample_folder, basename + ".tif")
    write_path_label = os.path.join(label_folder, basename_label + ".tif")

    print("Write to" + write_path)
    print("Write to" + write_path_label)

    if args.dryrun is False: 
      tiff = load_correct_tiff(f) 
      noise = downsample_y(add_noise(tiff, args.snr, 0.2), args.ds)
      frame = tiff[0, :, :] 

      tifffile.imwrite(write_path, noise)
      tifffile.imwrite(write_path_label, frame)

    

    

    



