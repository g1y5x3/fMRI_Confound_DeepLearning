import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm
from sfcn import SFCN
from imageloader import IXIDataset

parser = argparse.ArgumentParser(description="Example:")
parser.add_argument("--bs",    type=int, default=8,  help="batch size")
parser.add_argument("--nw",    type=int, default=6,  help="number of workers for the dataloader")
parser.add_argument("--epoch", type=int, default=10, help="total number of epochs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Test with other range that does not produce a x64 output
bin_range   = [21,85]

batch_size  = args.bs
num_workers = args.nw
num_epochs  = args.epoch

print("\nDataloader:")
data_train = IXIDataset(data_dir="data", label_file="IXI_train.csv", bin_range=bin_range)
dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
bin_center = data_train.bin_center.reshape([-1,1])

x, y = next(iter(dataloader_train))
print("\nTraining data summary:")
print(f"Total data: {len(data_train)}")
print(f"Input {x.shape}")
print(f"Label {y.shape}")

for epoch in tqdm(range(num_epochs)):
  for images, labels in dataloader_train:
    x, y = images.to(device), labels.to(device)

