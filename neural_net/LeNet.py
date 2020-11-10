#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: LeNet.py 
@time: 2020/11/10 4:44 PM 
@desc: 
'''

# optim中定义了各种各样的优化方法，包括SGD
import torch.optim as optim
import torch.nn as nn

class LeNet(nn.Module):
  # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
  def __init__(self):
    super(LeNet, self).__init__()
    # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    # 最终有10类，所以最后一个全连接层输出数量是10
    self.fc3 = nn.Linear(84, 10)
    self.pool = nn.MaxPool2d(2, 2)

  # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
  def forward(self, x):
    x = nn.ReLU(self.conv1(x))
    x = self.pool(x)
    x = nn.ReLU(self.conv2(x))
    x = self.pool(x)
    # 下面这步把二维特征图变为一维，这样全连接层才能处理
    x = x.view(-1, 16 * 5 * 5)
    x = nn.ReLU(self.fc1(x))
    x = nn.ReLU(self.fc2(x))
    x = self.fc3(x)
    return x

device = torch.device("cuda:0")
net = LeNet().to(device)



# CrossEntropyLoss就是我们需要的损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Start Training...")
for epoch in range(30):
    # 我们用一个变量来记录每100个batch的平均loss
    loss100 = 0.0
    # 我们的dataloader派上了用场
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 注意需要复制到GPU
        # 首先要把梯度清零，不然PyTorch每次计算梯度会累加，不清零的话第二次算的梯度等于第一次加第二次的
        optimizer.zero_grad()
        # 计算前向传播的输出
        outputs = net(inputs)
        # 根据输出计算loss
        loss = criterion(outputs, labels)
        # 算完loss之后进行反向梯度传播，这个过程之后梯度会记录在变量中
        loss.backward()
        # 用计算的梯度去做优化
        optimizer.step()
        loss100 += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss100 / 100))
            loss100 = 0.0

print("Done Training!")

# 构造测试的dataloader
dataiter = iter(testloader)
# 预测正确的数量和总数量
correct = 0
total = 0
# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 预测
        outputs = net(images)
        # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))