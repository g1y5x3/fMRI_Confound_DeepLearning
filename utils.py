import os
import numpy as np
import pandas as pd
from scipy.stats import norm

def filter_IXIDataset(info, data_dir):
  df = pd.DataFrame(columns=info.columns)
  dir_list = os.listdir(data_dir)
  for i in dir_list:
    if "nii" in i:
      df_tmp = info[info["IXI_ID"]==int(i[7:10])].copy()
      if not df_tmp.empty and not np.isnan(df_tmp["AGE"]).any():
        # for some reason there's duplicate from the original csv
        df_tmp = df_tmp.drop_duplicates(subset=["IXI_ID"])
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
  df["AGE"] = df["AGE"].round(2)
  return df

# original label file contains additional entries from subjects not appeared in the fMRI images
def preprocess_split(data_dir, label_file):
  info = pd.read_excel("data/"+label_file)
  df = filter_IXIDataset(info, data_dir)
  
  # split into training and validation
  df_Guys     = df[df["SITE"]=="Guys"]
  df_Guys_old = df_Guys[df_Guys["AGE"] >= 47]
  df_HH     = df[df["SITE"]=="HH"]
  df_HH_old = df_HH[df_HH["AGE"] >= 47]
  df_IOP  = df[df["SITE"]=="IOP"]

  df_train = pd.DataFrame(columns=df.columns)
  df_test  = pd.DataFrame(columns=df.columns)
  for df in [df_Guys, df_HH, df_IOP]:
    df_train_tmp = df.sample(frac=0.9, random_state=123)
    df_test_tmp  = df.drop(df_train_tmp.index)
    df_train = pd.concat([df_train, df_train_tmp], ignore_index=True)
    df_test  = pd.concat([df_test, df_test_tmp], ignore_index=True)
  df_train.to_csv(data_dir + "/IXI_all_train.csv", index=False)
  df_test.to_csv(data_dir + "/IXI_all_test.csv", index=False)

  df_train = pd.DataFrame(columns=df.columns)
  df_test  = pd.DataFrame(columns=df.columns)
  for df in [df_Guys_old, df_HH_old]:
    df_train_tmp = df.sample(frac=0.9, random_state=123)
    df_test_tmp  = df.drop(df_train_tmp.index)
    df_train = pd.concat([df_train, df_train_tmp], ignore_index=True)
    df_test  = pd.concat([df_test, df_test_tmp], ignore_index=True)
  df_train.to_csv(data_dir + "/IXI_unbiased_train.csv", index=False)
  df_test.to_csv(data_dir + "/IXI_unbiased_test.csv", index=False)

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
  preprocess_split(data_dir="data/IXI_4x4x4", label_file="IXI.xls")
  preprocess_split(data_dir="data/IXI_10x10x10", label_file="IXI.xls")

