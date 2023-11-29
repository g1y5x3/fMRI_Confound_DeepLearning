import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# original label file contains additional entries from subjects not appeared in the fMRI images
def preprocess_split(data_dir, label_file):
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

def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
      print("bin's range should be divisible by bin_step!")
      return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
      x = np.array(x)
      i = np.floor((x - bin_start) / bin_step)
      i = i.astype(int)
      return i, bin_centers
    elif sigma > 0:
      if np.isscalar(x):
        v = np.zeros((bin_number,))
        for i in range(bin_number):
          x1 = bin_centers[i] - float(bin_step) / 2
          x2 = bin_centers[i] + float(bin_step) / 2
          cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
          v[i] = cdfs[1] - cdfs[0]
        return v, bin_centers
      else:
        v = np.zeros((len(x), bin_number))
        for j in range(len(x)):
          for i in range(bin_number):
            x1 = bin_centers[i] - float(bin_step) / 2
            x2 = bin_centers[i] + float(bin_step) / 2
            cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
        return v, bin_centers

if __name__ == "__main__":
  preprocess_split(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI.csv")

