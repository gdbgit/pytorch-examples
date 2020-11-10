#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: visual_1.py 
@time: 2020/10/22 5:08 PM 
@desc: https://zhuanlan.zhihu.com/p/111144134
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot


np.random.seed(88)
x = np.random.rand(100, 1)
y = 1 + 2*x + .1*np.random.rand(100,1)

idx = np.arange(100)
np.random.shuffle(idx)

train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)



torch.manual_seed(88)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

# print(a, b)

yhat = a + b * x_train_tensor
error = y_train_tensor - yhat
loss = (error ** 2).mean()
# make_dot(yhat).view()
# graph_viz = make_dot(yhat)
# graph_viz.view()
# make_dot(error).view()
# make_dot(loss).view()

if loss > 0:
  yhat2 = b * x_train_tensor
  error2 = y_train_tensor - yhat2

loss += error2.mean()
make_dot(loss).view()
'''
lr = 1e-1
n_epochs = 1000
for epoch in n_epochs:
  yhat = a + b * x_train_tensor
  error = y_train_tensor - yhat
  loss = (error ** 2).mean()

  loss.backword()
  print(a.grad)
  print(b.grad)

  with torch.no_grad():
    a -= lr * a.grad
    b -= lr * b.grad

  a.grad.zero_()
  b.grad.zero_()

print(a, b)
'''