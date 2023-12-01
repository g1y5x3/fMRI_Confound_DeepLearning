import math
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import num2vect

class IXIDataset(Dataset):
  def __init__(self, data_dir, label_file, bin_range=None):
    print(f"Loading file: {label_file}")
    self.directory = data_dir
    self.info = pd.read_csv(data_dir+"/"+label_file)
    if not bin_range:
      self.bin_range = [math.floor(self.info['AGE'].min()), math.ceil(self.info['AGE'].max())]
      print(f"Age min {self.info['AGE'].min()}, Age max {self.info['AGE'].max()}")
      print("Computed Bin Range: ", self.bin_range)
    else:
      self.bin_range  = bin_range
      print(f"Provided Bin Range: {self.bin_range}")

    age = np.array([71.3])
    _, bc = num2vect(age, self.bin_range, 1, 1)
    self.bin_center = torch.tensor(bc, dtype=torch.float32)
    
    print(len(self.info))
    # Pre-load the images and labels (if RAM is allowed)
    nii_image = nib.load(self.directory+"/"+self.info["FILENAME"][0])
    tensor_image = torch.unsqueeze(torch.tensor(nii_image.get_fdata(), dtype=torch.float32),0)
    print(tuple(tensor_image.shape))
    data_image = torch.empty(tuple(tensor_image.shape), dtype=torch.float32) #.repeat(len(self.info), 1)
    print(data_image.shape)
    for i in tqdm(range(len(self.info))):
      nii_image = nib.load(self.directory+"/"+self.info["FILENAME"][i])

  def __len__(self):
    return len(self.info)

  def __getitem__(self, idx):
    # fetch the image
    nii_image = nib.load(self.directory+"/"+self.info["FILENAME"][idx])
    tensor_image = torch.unsqueeze(torch.tensor(nii_image.get_fdata(), dtype=torch.float32),0)

    # convert age labels from a single value to a distribution and 
    # add with a small value (1e-16) to prevent log(0) problem
    age = self.info["AGE"][idx]
    # NAN bug sometimes
    y, _ = num2vect(age, self.bin_range, 1, 1)
    assert not np.isnan(y).any(), f"y contains NaN values with age {age}"
    y += 1e-16
    tensor_label = torch.tensor(y, dtype=torch.float32)

    return tensor_image, tensor_label

if __name__ == "__main__":
  bin_range   = [21,85]
  data_train = IXIDataset(data_dir="data", label_file="IXI_train.csv", bin_range=bin_range)
