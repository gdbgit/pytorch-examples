#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: SGD.py 
@time: 2020/10/25 1:38 PM 
@desc: 
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn



np.random.seed(88)
# a = np.random.randn(1)
# b = np.random.randn(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(88)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

idx = np.arange(100)
np.random.shuffle(idx)
train_idx, val_idx = idx[:80], idx[80:]

x = np.random.randn(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

x_train, x_val = x[train_idx], x[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

lr = 1e-1
n_epochs = 1000
optimizer = optim.SGD([a, b], lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
  y_pred = a + b * x_train_tensor
  # error = y_train_tensor - y_pred
  # loss = (error ** 2).mean()
  loss = loss_fn(y_train_tensor, y_pred)

  loss.backward()

  optimizer.step()

  optimizer.zero_grad()

print(a, b)

