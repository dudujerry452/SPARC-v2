import torch 
from torch import nn 
import torch.utils.data as torchdata
import numpy 
from tifftool.load_tiff import  load_correct_tiff 
import os 

class PatchDataset(torchdata.Dataset): 
  def __init__(self,
               raw_data_folder, 
               label_data_folder,
               load_data_num=-1,  # -1: all the data 
               patch_t=32, 
               patch_y=32, 
               patch_x=128.  
               
               ):
    super().__init__()

    files = [f for f in os.listdir(raw_data_folder) if os.path.isfile(os.path.join(raw_data_folder, f))]

    print(files) 

dataset = PatchDataset("../zzydata/standard")
    
