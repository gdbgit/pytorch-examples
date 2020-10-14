#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: cnn1.py 
@time: 2020/10/14 12:43 PM 
@desc: 
'''


# https://zhuanlan.zhihu.com/p/38236978
# 这里我们会搭建一个简单的CNN模型，用以分类来自CIFAR 10数据集的RGB图像。该数据集包含了50000张训练图像和10000张测试图像，所有图像大小为32 X 32。

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
  def __init__(self, num_classes=10):
    super(SimpleNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()

    self.pool = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu3 = nn.ReLU()

    self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu4 = nn.ReLU()

    self.fc = nn.Linear(in_features=16*16*24, out_features=num_classes)

  def forward(self, input):
    output = self.conv1(input)
    output = self.relu1(output)

    output = self.conv2(output)
    output = self.relu2(output)

    output = self.conv3(output)
    output = self.relu3(output)

    output = self.conv4(output)
    output = self.relu4(outpu)
'''
注意：我们在将最后一个卷积 -ReLU 层中的特征图谱输入图像前，必须把整个图谱压平。
最后一层有 24 个输出通道，由于 2X2 的最大池化，在这时我们的图像就变成了16 X 16（32/2 = 16）。
我们压平后的图像的维度会是16 x 16 x 24
'''
    output = self.view(-1, 16*16*24)

    output = self.fc(output)


