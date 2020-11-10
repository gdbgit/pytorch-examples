#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: 2layer_numpy_impl.py
@time: 2020/10/19 3:44 PM 
@desc: 
'''

import numpy as np

# generate data
np.random.seed(88)
x = np.random.rand(100, 1)
y = 1 + 2*x + .1*np.random.rand(100, 1)
# shuffle index
idx = np.arange(100)
np.random.shuffle(idx)
# first 80 for train
train_idx = idx[:80]
# remain 20 for val
val_idx = idx[80:]
# generate train and val dataset
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]





# init parameters a, b randomly
np.random.seed(66)
a = np.random.randn(1)
b = np.random.randn(1)
print(a, b)

# set learning rate
lr = 1e-1
# define epoch
n_epochs = 1000

for epoch in range(n_epochs):
  # compute model predict output
  yhat = a + b*x_train

  # How wrong is our model? That's the error
  error = (y_train - yhat)
  # for regression, compute MSE loss
  loss = (error ** 2).mean()

  # Compute gradient for parameters a and b
  a_grad = -2 * error.mean()
  b_grad = -2 * (x_train * error).mean()

  # Update a, b using gradients and learning rate
  a = a - lr * a_grad
  b = b - lr * b_grad

print(a, b)



#sanity check
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_)

print('mse:', mean_squared_error(y_val, linr.predict(x_val)))
print('variance score:', r2_score(y_val, linr.predict(x_val)))
print('score:', linr.score(x_val, y_val))
# 1 - ( ((y_val - y_pred)**2).sum() / (y_val - y_val.mean())**2).sum())

import matplotlib.pyplot as plt
# plt.title('val plot')
# plt.scatter(x_val, y_val, color='green')
# plt.plot(x_val, linr.predict(x_val), color='red', linewidth=3)
# plt.show()

plt.title('train plot')
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, y_train, color='red', linewidth=1)
plt.show()


