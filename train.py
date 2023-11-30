import wandb
import torch
import torch.optim as optim
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from sfcn import SFCN
from imageloader import IXIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Test with other range that does not produce a x64 output
bin_range   = [21,85] 
batch_size  = 8
num_workers = 6
num_epochs  = 130
lr = 1e-2
wd = 1e-3

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

data_train = IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_train.csv", bin_range=bin_range)
data_test  = IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_test.csv",  bin_range=bin_range)
dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataloader_test  = DataLoader(data_test,  batch_size=batch_size, num_workers=num_workers, shuffle=False)

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

# print("\nTesting data summary:")

model = SFCN(output_dim=y.shape[1])
print(f"\nModel Dtype: {next(model.parameters()).dtype}")
summary(model, x.shape)
model.to(device)

# this loss expects the argument input in log-space, default would automatically
# convert target into log space
# https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

print("\nTraining:")
for epoch in tqdm(range(num_epochs)):
  epoch_loss = 0.0
  for images, labels in dataloader_train:
    x, y = images.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output.log(), y.log())
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  print(f"epoch={epoch}, loss={epoch_loss}")
  wandb.log({"loss": epoch_loss})
