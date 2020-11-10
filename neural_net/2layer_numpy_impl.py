#!/usr/bin/env python
# encoding: utf-8
'''
@author: mic
@license: (C) Copyright 2020-2099, Clobotics.com
@contact: michael.li@clobotics.com
@file: 2layer_numpy_impl.py
@time: 2020/10/20 5:03 PM
@desc:https://pytorch.apachecn.org/docs/1.2/beginner/pytorch_with_examples.html
'''

import numpy as np

# N: batch size, D_in: input dim, H: hidden input dim, D_out: output dim
N, D_in, H, D_out = 64, 1000, 100, 10

# generate data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# randomly init parameters
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
n_epochs = 500
for epoch in range(n_epochs):
  # forward propagation, compute prediction y_pred
  h = x.dot(w1)
  h_relu = np.maximum(h, 0)
  y_pred = h_relu.dot(w2)

  # compute loss
  loss = np.square(y_pred - y).sum()
  print(epoch, loss)

  # backward propagation, compute gradient of parameters w1, w2 on loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_w2 = h_relu.T.dot(grad_y_pred)
  grad_h_relu = grad_y_pred.dot(w2.T)
  grad_h = grad_h_relu.copy()
  grad_h[h<0] = 0
  grad_w1 = x.T.dot(grad_h)

  # update parameters
  w1 -= lr * grad_w1
  w2 -= lr * grad_w2

# print(w1, w2)



