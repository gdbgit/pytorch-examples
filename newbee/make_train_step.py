#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: make_train_step.py 
@time: 2020/10/25 10:06 PM 
@desc: 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(88)
x = np.random.randn(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)
idx = np.arange(100)
np.random.shuffle(idx)
train_idx, val_idx = idx[:80], idx[80:]
x_train, x_val = x[train_idx], x[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)



class LayerLinearRegression(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(1, 1)

  def forward(self, x):
    return self.linear(x)

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

for n in range(n_epochs):
  loss = train_step(x_train_tensor, y_train_tensor)
  losses.append(loss)

print(model.state_dict())

