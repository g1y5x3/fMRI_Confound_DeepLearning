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
num_epochs  = 1

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

optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)

print("\nTraining:")
# Debug the training loop
# epoch_loss = 0.0
# optimizer.zero_grad()
# x, y = x.to(device), y.to(device)
# output = model(x)
# print(output)
# print(y)
# loss = criterion(output.log(), y.log())
# print(loss)

for epoch in tqdm(range(num_epochs)):
  for images, labels in dataloader_train:
    x, y = images.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output.log(), y.log())
    loss.backward()
    optimizer.step()
    print(loss)
    if torch.isnan(loss):
      print(output)
      print(y)
      break

#   print(f"[{epoch}] loss: {epoch_loss}")
