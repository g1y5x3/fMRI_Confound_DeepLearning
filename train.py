import torch
import torch.optim as optim
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from sfcn import SFCN, my_KLDivLoss
from imageloader import IXIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bs = 8
data_train = IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_train.csv",
                        bin_range=[21, 85])
data_test  = IXIDataset(data_dir="/home/iris/yg5d6/Workspace/IXI_dataset/preprocessed", label_file="IXI_test.csv",
                        bin_range=[21, 85])
dataloader_train = DataLoader(data_train, batch_size=bs, shuffle=True,  num_workers=8)
dataloader_test  = DataLoader(data_test,  batch_size=bs, shuffle=False, num_workers=8)
x, y = next(iter(dataloader_train))
print("\nTraining data summary:")
print(f"Total data: {len(data_train)}")
print(f"Input {x.shape}")
print(f"Output {y.shape}")

x, y = next(iter(dataloader_test))
print("\nTesting data summary:")
print(f"Total data: {len(data_test)}")
print(f"Input  {x.shape}")
print(f"Output {y.shape}")

# print("\nTesting data summary:")

model = SFCN(output_dim=y.shape[1])
print(f"\nModel Dtype: {next(model.parameters()).dtype}")
summary(model, x.shape)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
num_epochs = 10

# Debug the training loop
x, y = x.to(device), y.to(device)
output = model(x)
output = output[0].reshape(y.shape[0],-1)
print(output)
print(output.shape)
print(y)
loss = my_KLDivLoss(output, y)
print(loss)

# print("\nTraining:")
# for epoch in tqdm(range(num_epochs)):
#   epoch_loss = 0.0
#   for images, labels in tqdm(dataloader_train):
#     x, y = images.to(device), labels.to(device)
#     outputs = model(x)
#
#     optimizer.zero_grad()
#     loss = my_KLDivLoss(outputs[0].reshape(y.shape[0],-1), y)
#     loss.backward()
#     optimizer.step()
#     epoch_loss += loss.item()
#
#   print(f"[{epoch}] loss: {epoch_loss}")
