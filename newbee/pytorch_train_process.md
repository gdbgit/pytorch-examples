## Pytorch train process: https://zhuanlan.zhihu.com/p/111144134

1. 创建参数（权重）张量，最好使用方法一：在创建参数张量时指定device，方法二注意：先to(device)，再requires_grad_()

   ```python
   import numpy as np
   import torch
   
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   torch.manual_seed(88)
   #方法一
   a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
   b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
   #方法二
   a = torch.randn(1, dtype=torch.float).to(device)
   b = torch.randn(1, dtype=torch.float).to(device)
   a.requires_grad_()
   b.requires_grad_()
   ```

2. 生成data, include x_train/x_val, y_train/y_val

   ```python
   idx = np.arange(100)
   np.random.shuffle(idx)
   train_idx, val_idx = idx[:80], idx[80:]
   
   x = np.random.randn(100, 1)
   y = a + b * x + .1 * np.random.randn(100, 1)
   
   x_train, x_val = x[train_idx], x[val_idx]
   y_train, y_val = y[train_idx], y[val_idx]
   ```

3. numpy narray 转 pytorch tensor，创建数据张量，数据张量不同于参数张量，不需要requires_grad

   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   x_train_tensor = torch.from_numpy(x_train).float().to(device)
   y_train_tensor = torch.from_numpy(y_train).float().to(device)
   ```

4. 设置learing rate，迭代n_epochs, 

   ```python
   lr = 1e-1
   n_epoches = 1000
   #optimizer = optim.SGD([a, b], lr=lr)
   for epoch in n_epochs:
   	xxx
   ```

   1. 每个迭代都要计算y_pred, error, loss

      ```python
      y_pred = a + b * x_train_tensor
      error = y_train_tensor - y_pred
      loss = (error ** 2).mean()
      ```

   2. 计算参数张量的梯度，并用learing rate更新参数张量

      ```python
      #方法一，使用numpy narray，需要手动计算梯度
      a_grad = -2 * error.mean()
      b_grad = -2 * (x_train_tensor * error).mean()
      a -= lr * a_grad
      b -= lr * b_grad
      #方法二，使用pytorch，只需要调用loss.backward()，即可计算出梯度
      loss.backward()
      with torch.no_grad():
        a -= lr * a_grad
        b -= lr * b_grad
      a.grad.zero()
      b.grad.zero()
      #方法三，定义优化器，更新参数张量，并zero all gradient of parameters
      optimizer = optim.SGD([a, b], lr=lr) #这一步在迭代epoch之前做
      	loss.backward() #计算梯度
        optimizer.step() #使用梯度和lr更新参数a, b
        optimizer.zero_grad() #zero gradient of a, b
      ```

      

