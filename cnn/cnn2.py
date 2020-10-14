#!/usr/bin/env python 
# encoding: utf-8 
''' 
@author: mic 
@license: (C) Copyright 2020-2099, Clobotics.com 
@contact: michael.li@clobotics.com 
@file: cnn2.py 
@time: 2020/10/14 1:09 PM 
@desc: 
'''


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam



class SimpleNet(nn.Module):
  def __init__(self, num_classes):
    super(SimpleNet, self).__init__()

    self.unit1 = Unit(in_channels=3, out_channels=32)
    self.unit2 = Unit(in_channels=32, out_channels=32)
    self.unit3 = Unit(in_channels=32, out_channels=32)

    self.pool1 = nn.MaxPool2d(kernel_size=2)

    self.unit4 = Unit(in_channels=32, out_channels=64)
    self.unit5 = Unit(in_channels=64, out_channels=64)
    self.unit6 = Unit(in_channels=64, out_channels=64)
    self.unit7 = Unit(in_channels=64, out_channels=64)

    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.unit8 = Unit(in_channels=64, out_channels=128)
    self.unit9 = Unit(in_channels=128, out_channels=128)
    self.unit10 = Unit(in_channels=128, out_channels=128)
    self.unit11 = Unit(in_channels=128, out_channels=128)

    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.unit12 = Unit(in_channels=128, out_channels=128)
    self.unit13 = Unit(in_channels=128, out_channels=128)
    self.unit14 = Unit(in_channels=128, out_channels=128)

    self.avgpool = nn.AvgPool2d(kernel_size=4)

    self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit14, self.unit5, self.unit6, self.unit7, self.pool2,
                             self.unit8, self.unit9, self.unit10, self.unit11, self.pool3, self.unit12, self.unit13, self.unit14, self.avgpool)

    self.fc = nn.Linear(in_features=128, out_features=num_classes)

  def forward(self, input):
    output = self.net(input)
    '''
    最后一个单元后面的AvgPooling层会计算每个通道中的所有函数的平均值。
    该单元的输出有128个通道，在池化3次后，我们的32 X 32图像变成了4 X 4。
    我们以核大小为4使用AvgPool2D，将我们的特征图谱调整为1X1X128。
    '''
    output = output.view(-1, 128)
    output = self.fc(output)
    return output



class Unit(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Unit, self).__init__()

    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm2d(num_features=out_channels)
    self.relu = nn.ReLU()

  def forward(self, input):
    output = self.conv(input)
    output = self.bn(output)
    output = self.relu(output)

    return output

def adjust_learning_rate(epoch):
  lr = 0.001
  if epoch > 30:
    lr = lr / 10
  elif epoch > 60:
    lr = lr / 100
  elif epoch > 90:
    lr = lr / 1000
  elif epoch > 120:
    lr = lr / 10000
  elif epoch > 150:
    lr = lr / 100000
  elif epoch > 180:
    lr = lr / 1000000

  for param_group in optimizer.param_groups:
    param_group["lr"] = lr

def train(num_epochs):
  best_acc = 0.0
  for epoch in range(num_epochs):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loder):
      if cuda_avail:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
      optimizer.zero_grad()
      outputs = model(images)
      loss = loss_fn(outputs, labels)
      loss.backword()

      optimizer.step()

      train_loss += loss.cpu().data[0] * image.size(0)
      _, prediction = torch.max(outputs.data, 1)

      train_acc += torch.sum(prediction == labels.data)

  adjust_learning_rate(epoch)
  train_acc = train_acc / 50000
  train_loss = train_loss / 50000

  test_acc = test()

  if test_acc > best_acc:
    save_models(epoch)
    best_acc = test_acc

  print(f'Epoch {epoch}, Train Accuracy {train_acc}, Train Loss {train_loss}, Test Accuracy {test_acc}')


def save_models(epoch):
  torch.save(model.state_dict(), f'cifar10model_{epoch}.model')
  print('Checkpoint saved')

def test():
  model.eval()
  test_acc = 0.0
  for i, (images, labels) in enumerate(test_loader):
    if cuda_avail:
      images = Variable(images.cuda())
      labels = Variable(labels.cuda())

    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    test_acc += torch.num(prediction == labels.data)

  test_acc = test_acc / 10000
  return test_acc


# define train transformation
train_transformations = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(32,padding=4),
  transforms.ToTensor(),
  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# define train set and train data loader
train_set = CIFAR10('/datadrive/mic/pytorch-examples', train=True, transform=train_transformations, download=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

# define test transformation
test_transformations = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# define test set and test data loader
test_set = CIFAR10('/datadrive/mic/pytorch-examples', train=False, transform=test_transformations, download=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)


#defin model
model = SimpleNet(num_classes=10)

# if GPU available, move model to GPU
cuda_avail = torch.cuda.is_available()
if cuda_avail:
  model.cuda()

# define optimizer and loss functino
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


if __name__ == '__main__':
  train(200)


