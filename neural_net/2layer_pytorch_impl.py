#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: 2layer_pytorch_impl.py 
@time: 2020/10/21 2:45 AM 
@desc: https://pytorch.apachecn.org/docs/1.2/beginner/pytorch_with_examples.html
'''

import torch

dtype = torch.float
device = torch.device('cpu')
# device = torch.device('cuda:0') # Uncomment this to run on GPU

# N是批尺寸大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出数据
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 随机初始化权重
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

lr = 1e-6

for epoch in range(500):
  # 前向传播：计算预测值y_pred
  h = x.mm(w1)
  h_relu = h.clamp(min=0)
  y_pred = h_relu.mm(w2)

  # 计算并输出loss
  loss = (y_pred - y).pow(2).sum().item()
  if epoch % 100 == 99:
    print(epoch, loss)

  # 反向传播，计算w1、w2对loss的梯度
  grad_y_pred = 2.0 * (y_pred -y)
  grad_w2 = h_relu.t().mm(grad_y_pred)
  grad_h_relu = grad_y_pred.mm(w2.t())
  grad_h = grad_h_relu.clone()
  grad_h[h<0] = 0
  grad_w1 = x.t().mm(grad_h)

  # 使用梯度下降更新权重
  w1 -= lr * grad_w1
  w2 -= lr * grad_w2
