#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: data_loader.py 
@time: 2020/10/25 10:42 PM 
@desc: 
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)


class CustomDataset(Dataset):
  def __init__(self, x_tensor, y_tensor):
    self.x = x_tensor
    self.y = y_tensor

  def __getitem__(self, index):
    return self.x[index], self.y[index]

  def __len__(self):
    return len(self.x)

np.random.seed(88)
x = np.random.randn(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

#我们不希望我们的全部训练数据都被加载到GPU张量中，就像我们到目前为止的例子中所做的那样，因为它占用了我们宝贵的显卡RAM中的空间。
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

dataset = CustomDataset(x_tensor, y_tensor)
print(dataset[0])

# dataset = TensorDataset(x_tensor, y_tensor)
# print(dataset[0])

train_dataset, val_dataset = random_split(dataset, [80, 20])
train_loder = DataLoader(dataset=train_dataset, batch_size=16)
val_loder = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True)
print(next(iter(train_loder)))


def make_train_step(model, loss_fn, optimizer):
  def train_step(x, y):
    model.train()
    y_pred = model(x)
    loss = loss_fn(y, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
  return train_step

lr = 1e-1
n_epochs = 1000
# model = LayerLinearRegression()
model = nn.Sequential(nn.Linear(1, 1)).to(device)
loss_fn = nn.MSELoss(reduction='mean')
opt = optim.SGD(model.parameters(), lr=lr)
train_step = make_train_step(model, loss_fn, opt)
losses = []
val_losses = []

for n in range(n_epochs):
  for x_batch, y_batch in train_loder:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    loss = train_step(x_batch, y_batch)
    losses.append(loss)

  with torch.no_grad():
    for x_val, y_val in val_loder:
      x_val = x_val.to(device)
      y_val = y_val.to(device)

      model.eval()

      y_pred = model(x_val)
      val_loss = loss_fn(y_val, y_pred)
      val_losses.append(val_loss)

print(model.state_dict())
