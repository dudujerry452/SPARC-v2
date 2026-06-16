import torch 
from torch import nn 
import torch.utils.data as torchdata
import numpy as np 
from tifftool.load_tiff import  load_correct_tiff, load_tiff, write_tiff
import os, random

def random_transform(input, label):
    p_trans = random.randrange(4)
    if p_trans == 0:
        pass
    elif p_trans == 1:
        input = np.rot90(input, k=2, axes=(1, 2))
        label = np.rot90(label, k=2, axes=(0, 1))
    elif p_trans == 2:
        input = input[:, ::-1, :]
        label = label[::-1, :]
    elif p_trans == 3:
        input = input[:, :, ::-1]
        label = label[:, ::-1]
    return input, label

class PatchDataset(torchdata.Dataset): 
  def __init__(self,
               raw_data_folder, # t,h,w
               label_data_folder, # h,w
               load_data_num=-1,  # -1: all the data 
               patch_t=32, 
               patch_y=32, 
               patch_x=128, 
               gap=0.5, 
               use_video=False, 
               use_random=True, 
               ):
    super().__init__()

    self.use_video = use_video
    self.use_random = use_random

    files_raw = [f for f in os.listdir(raw_data_folder) if os.path.isfile(os.path.join(raw_data_folder, f)) and f.endswith(".tif")]
    files_label = [f for f in os.listdir(label_data_folder) if os.path.isfile(os.path.join(label_data_folder, f)) and f.endswith("_label.tif")]

    file_pair = []

    for f in files_raw: 
      t = os.path.splitext(f)[0] + "_label"
      for lf in files_label: 
        lt = os.path.splitext(lf)[0]
        if lt == t: 
          file_pair.append((f, lf))
          break 

    tmp = []
    for r, l in file_pair: 
      r = os.path.join(raw_data_folder, r)
      l = os.path.join(label_data_folder, l) 
      tmp.append((r, l))
      
    # file_pair is raw_data - label_data pair 
    file_pair = tmp  

    if load_data_num == -1: 
      load_data_num = len(file_pair) 

    # ---- clip patches -----
    # sliding window, gap: jump distance

    self.sample_patchs = []
    self.label_patchs = []
    
    for ind in range(load_data_num): 
      raw = load_tiff(file_pair[ind][0]).astype(np.float32)
      label = load_tiff(file_pair[ind][1]).astype(np.float32)

      T, H, W = raw.shape 
      H1, W1 = label.shape 
      scale_h = int(H1 / H)
      scale_w = int(W1 / W)

      if label.shape[0] != H or label.shape[1] != W: 
        print(f"warning: label has different shape {label.shape} with raw data {raw.shape} ! ")

      gap_t = int(gap * patch_t)
      gap_y = int(gap * patch_y) 
      gap_x = int(gap * patch_x)

      
      for y in range(0, H-patch_y+1, gap_y): 
        for x in range(0, W-patch_x+1, gap_x): 
          for t in range(0, T-patch_t+1, gap_t):
            self.sample_patchs.append(raw[t:t+patch_t, y:y+patch_y, x:x+patch_x])
            y1 = y * scale_h
            x1 = x * scale_w 
            self.label_patchs.append(label[y1:y1+patch_y*scale_h, x1:x1+patch_x*scale_w])

    self.sample_patchs = np.stack(self.sample_patchs) 
    self.label_patchs = np.stack(self.label_patchs)
    # print(self.sample_patchs.shape) 
    # print(self.label_patchs.shape)

      # write_tiff(self.sample_patchs[0], "tests.tif")
      # write_tiff(self.label_patchs[0], "testl.tif")
      # write_tiff(self.sample_patchs[1], "tests1.tif")
      # write_tiff(self.label_patchs[1], "testl1.tif")

  def __len__(self): 
    return self.sample_patchs.shape[0] 
  def __getitem__(self, idx):
    raw_img = self.sample_patchs[idx]
    label_img = self.label_patchs[idx] 
    
    if self.use_random == True: 
      raw_img, label_img = random_transform(raw_img, label_img)

    if self.use_video == False: 
      raw_img = raw_img[0:1, :, :].squeeze(axis=0) # 暂时丢弃3D数据, 只使用其第一帧

    # print(raw_img.shape, label_img.shape)

    raw_tensor = torch.from_numpy(np.expand_dims(raw_img, 0).copy())
    label_tensor = torch.from_numpy(np.expand_dims(label_img, 0).copy())

    return raw_tensor, label_tensor


      
      


# dataset = PatchDataset(
#    "../zzydata/dataset_st/samples", 
#    "../zzydata/dataset_st/labels", 
#    2)

# dataloader = torchdata.DataLoader(
#    dataset, 
#    batch_size=4, 
#    shuffle=True, 
#    num_workers=0
# )


