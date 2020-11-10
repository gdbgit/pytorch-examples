#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: MyReLU.py 
@time: 2020/10/21 3:19 AM 
@desc: https://pytorch.apachecn.org/docs/1.2/beginner/pytorch_with_examples.html
'''


import torch

class MyReLU(torch.autograd.Function):
  """
  我们可以通过建立torch.autograd的子类来实现我们自定义的autograd函数，并完成张量的正向和反向传播。
  """

  @staticmethod
  def forward(ctx, input):
    """
    在前向传播中，我们收到包含输入和???返回的张量包含输出的张量???。
    ctx是可以使用的上下文对象存储信息以进行向后计算。
    您可以使用ctx.save_for_backward方法缓存任意对象，以便反向传播使用。
    """
    ctx.save_for_backward(input)
    return input.clamp(min=0)

  @staticmethod
  def backward(ctx, grad_output):
    """
    在反向传播中，我们接收到上下文对象和一个张量，其包含了???????相对于正向传播过程中产生的输出的损失的梯度?????。
    我们可以从上下文对象中检索缓存的数据，并且必须计算并返回与正向传播的输入相关的损失的梯度。
    """
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input<0] = 0
    return grad_input

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N是批尺寸大小； D_in 是输入维度；
# H 是隐藏层维度； D_out 是输出维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 产生输入和输出的随机张量
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 产生随机权重的张量
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # 为了使用我们的方法，我们调用Function.apply方法。 我们将其命名为“ relu”。
  relu = MyReLU.apply

  # 正向传播：使用张量上的操作来计算输出值y;
  # 我们使用自定义的自动求导操作来计算 RELU.
  y_pred = relu(x.mm(w1)).mm(w2)

  # 计算并输出loss
  loss = (y_pred - y).pow(2).sum()
  if t % 100 == 99:
    print(t, loss.item())

  # 使用autograd计算反向传播过程。
  loss.backward()

  # 用梯度下降更新权重
  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    # 在反向传播之后手动清零梯度
    w1.grad.zero_()
    w2.grad.zero_()
