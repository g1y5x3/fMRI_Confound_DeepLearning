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

    # Pre-load the images and labels (if RAM is allowing)
    nii = nib.load(self.directory+"/"+self.info["FILENAME"][0])
    image = torch.unsqueeze(torch.tensor(nii.get_fdata(), dtype=torch.float32),0)
    self.image_all = torch.empty((len(self.info),) + tuple(image.shape), dtype=torch.float32)

    age = np.array([71.3])
    y, bc = num2vect(age, self.bin_range, 1, 1)
    label = torch.tensor(y, dtype=torch.float32)
    self.label_all = torch.empty((len(self.info),) + tuple(label.shape)[1:], dtype=torch.float32)

    for i in tqdm(range(len(self.info)), desc="Loading Data"):
      nii = nib.load(self.directory+"/"+self.info["FILENAME"][i])
      self.image_all[i,:] = torch.unsqueeze(torch.tensor(nii.get_fdata(), dtype=torch.float32),0)

      age = self.info["AGE"][i]
      y, _ = num2vect(age, self.bin_range, 1, 1)
      y += 1e-16
      self.label_all[i,:] = torch.tensor(y, dtype=torch.float32)

    self.bin_center = torch.tensor(bc, dtype=torch.float32)
    print(self.image_all[0,:].shape)
    print(self.label_all[0,:].shape)

  def __len__(self):
    return len(self.info)

  def __getitem__(self, idx):
    return self.image_all[idx,:], self.label_all[idx,:]

if __name__ == "__main__":
  bin_range   = [21,85]
  data_train = IXIDataset(data_dir="data", label_file="IXI_train.csv", bin_range=bin_range)
