import os
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# original label file contains additional entries from subjects not appeared in the fMRI images
def preprocess(data_dir, label_file):
  info = pd.read_csv(data_dir+"/"+label_file)
  df = pd.DataFrame(columns=info.columns)
  dir_list = os.listdir(data_dir)
  for i in dir_list:
    if "nii" in i:
      df_tmp = info[info["IXI_ID"]==int(i[7:10])].copy()  # just not have to deal with the warning
      df_tmp["FILENAME"] = i
      if "HH" in i:
        df_tmp["SITE"] = "HH"
      if "Guys" in i:
        df_tmp["SITE"] = "Guys"
      if "IOP" in i:
        df_tmp["SITE"] = "IOP"
      df = pd.concat([df, df_tmp], ignore_index=True)
  df = df.sort_values(by="IXI_ID", ascending=True)
  df = df.reset_index(drop=True)
  
  # split into training and validation
  df_Guys = df[df["SITE"]=="Guys"]
  df_HH   = df[df["SITE"]=="HH"]
  df_IOP  = df[df["SITE"]=="IOP"]

  df_train = pd.DataFrame(columns=info.columns)
  df_test  = pd.DataFrame(columns=info.columns)
  for df in [df_Guys, df_HH, df_IOP]:
    train_df = df.sample(frac=0.9, random_state=123)
    test_df  = df.drop(train_df.index)
    df_train = pd.concat([df_train, train_df], ignore_index=True)
    df_test  = pd.concat([df_test, test_df], ignore_index=True)

  df_train.to_csv("IXI_train.csv", index=False)
  df_test.to_csv("IXI_test.csv", index=False)
 
class IXIDataset(Dataset):
  def __init__(self, data_dir, label_file):
    self.info = pd.read_csv(label_file)
    self.directory = data_dir
    print(self.info)
    print(f"Age min {self.info['AGE'].min()}, Age max {self.info['AGE'].max()}")
  
  def __len__(self):
    return len(self.info)

  def __getitem__(self, idx):
    filepath = os.path.join(self.directory, self.info["FILENAME"][idx])
    nii_image = nib.load(filepath)
    tensor_image = torch.from_numpy(nii_image.get_fdata()).type(torch.FloatTensor)
    tensor_label = torch.tensor(self.info["AGE"][idx]).type(torch.FloatTensor)

    return tensor_image, tensor_label

if __name__ == "__main__":
  preprocess(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI.csv")

  IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_train.csv")
  IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_test.csv")
