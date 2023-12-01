import os
import argparse
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import trange
from sfcn import SFCN
from imageloader import IXIDataset

WANDB = os.getenv("WANDB", False)

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
lr = 1e-2
wd = 1e-3

if WANDB:
  wandb.login()
  run = wandb.init(
    project="Confounding in fMRI Deep Learning",
    config={
      "num_workers":   num_workers,
      "num_epochs":    num_epochs,
      "learning_rate": lr,
      "weight_decay":  wd,
    },
  )

print("\nDataloader:")
data_train = IXIDataset(data_dir="data", label_file="IXI_train.csv", bin_range=bin_range)
data_test  = IXIDataset(data_dir="data", label_file="IXI_test.csv",  bin_range=bin_range)
dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
dataloader_test  = DataLoader(data_test,  batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
bin_center = data_train.bin_center.reshape([-1,1])

x, y = next(iter(dataloader_train))
print("\nTraining data summary:")
print(f"Total data: {len(data_train)}")
print(f"Input {x.shape}")
print(f"Label {y.shape}")

x, y = next(iter(dataloader_test))
print("\nTesting data summary:")
print(f"Total data: {len(data_test)}")
print(f"Input {x.shape}")
print(f"Label {y.shape}")

model = SFCN(output_dim=y.shape[1])
print(f"\nModel Dtype: {next(model.parameters()).dtype}")
summary(model, x.shape)
# TODO load pretrained weights from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/raw/master/brain_age/run_20190719_00_epoch_best_mae.p
model.to(device)

criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

t = trange(num_epochs, desc="\nTraining", leave=True)
for epoch in t:
  loss_kl_train = 0.0
  MAE_age_train = 0.0
  for images, labels in dataloader_train:
    optimizer.zero_grad()
    x, y = images.to(device), labels.to(device)
    output = model(x)
    loss = criterion(output.log(), y.log())
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      age_target = labels @ bin_center
      age_pred   = output.cpu() @ bin_center
      MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")

    loss_kl_train += loss.item()
    MAE_age_train += MAE_age.item()

  loss_kl_train = loss_kl_train / len(dataloader_train)
  MAE_age_train = MAE_age_train / len(dataloader_train)

  with torch.no_grad():
    loss_kl_test = 0.0
    MAE_age_test = 0.0
    for images, labels in dataloader_test:
      x, y = images.to(device), labels.to(device)
      output = model(images.to(device))
      loss = criterion(output.log(), y.log())

      age_target = labels @ bin_center
      age_pred = output.cpu() @ bin_center
      MAE_age = F.l1_loss(age_pred, age_target, reduction="mean")

      loss_kl_test += loss.item()
      MAE_age_test += MAE_age.item()

  loss_kl_test = loss_kl_test / len(dataloader_test)
  MAE_age_test = MAE_age_test / len(dataloader_test)

  scheduler.step()

  t.set_description(f"Training: train/loss_kl {loss_kl_train:.2f}, train/MAE_age {MAE_age_train:.2f} test/loss_kl {loss_kl_test:.2f}, test/MAE_age {MAE_age_test:.2f}")
  if WANDB:
    wandb.log({"train/loss_kl": loss_kl_train,
               "train/MAE_age": MAE_age_train,
               "test/loss_kl": loss_kl_test,
               "test/MAE_age": MAE_age_test,
               })

torch.save(model.state_dict(), "model/model.pth")

if WANDB:
  artifact = wandb.Artifact("model", type="model")
  artifact.add_file("model/model.pth")
  run.log_artifact(artifact)
  run.finish()
