import os
import math
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from utils import preprocess_split, num2vect

class IXIDataset(Dataset):
  def __init__(self, data_dir, label_file, bin_range=None):
    print("\nData Loader")
    print(f"Load the label file: {label_file}")
    self.info = pd.read_csv(label_file)
    self.directory = data_dir
    if not bin_range:
      self.bin_range = [math.floor(self.info['AGE'].min()), math.ceil(self.info['AGE'].max())]
      print(f"Age min {self.info['AGE'].min()}, Age max {self.info['AGE'].max()}")
      print("Computed Bin Range: ", self.bin_range)
    else:
      self.bin_range  = bin_range
      print(f"Provided Bin Range: {self.bin_range}")

    age = np.array([71.3])
    y, bc = num2vect(age, self.bin_range, 1, 1)
    self.bin_center = bc
    y = torch.tensor(y, dtype=torch.float32)
    print(f"Label Shape: {y.shape}")

  def __len__(self):
    return len(self.info)

  def __getitem__(self, idx):
    filepath = os.path.join(self.directory, self.info["FILENAME"][idx])
    nii_image = nib.load(filepath)
    # tensor_image = torch.from_numpy(nii_image.get_fdata()).type(torch.FloatTensor)
    tensor_image = torch.unsqueeze(torch.tensor(nii_image.get_fdata(), dtype=torch.float32),0)
    age = self.info["AGE"][idx]
    y, _ = num2vect(age, self.bin_range, 1, 1)
    tensor_label = torch.tensor(y, dtype=torch.float32)

    return tensor_image, tensor_label
