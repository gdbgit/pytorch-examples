#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: linear_regression.py 
@time: 2020/10/25 9:34 PM 
@desc: https://zhuanlan.zhihu.com/p/111144134
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

idx = np.arange(100)
np.random.shuffle(idx)
train_idx, val_idx = idx[:80], idx[80:]
x = np.random.randn(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)
x_train, x_val = x[train_idx], x[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ManualLinearRegression(nn.Module):
  def __init__(self):
    super().__init__()
    # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
    self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

  def forward(self, x):
    return self.a + self.b * x

torch.manual_seed(88)

model = ManualLinearRegression().to(device)
print(model.state_dict())

lr = 1e-1
n_epochs = 1000

loss_fn = nn.MSELoss(reduction='mean')
opt = optim.SGD(model.parameters(), lr=lr)

for n in range(n_epochs):
  # Sets model to TRAIN mode, although no train process
  model.train()

  y_pred = model(x_train_tensor)
  loss = loss_fn(y_train_tensor, y_pred)
  print(n, loss.item(), model.a, model.b)
  time.sleep(0.1)
  loss.backward()
  opt.step()
  opt.zero_grad()

print(model.state_dict())