---
title: pytorch基础知识介绍、pytorch搭建基础神经网络合集
date: 2019-03-02 19:55:38
tags:
- Pytorch
categories:
- Pytorch
---

# pytorch介绍
PyTorch可以追溯到2002年诞生的Torch。Torch是一个与Numpy类似的张量(Tensor)操作库，它使用了一种不是很大众的语言Lua作为接口。在2017年，Torch的幕后团队推出了PyTorch。PyTorch不是简单地封装Lua Torch提供Python接口，而是对Tensor之上的所有模块进行了重构，并新增了最先进的自动求导系统，成为当下最流行的动态图框架。
PyTorch设计时遵循tensor→variable(autograd)→nn.Module 三个由低到高的抽象层次，分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块），而且这三个抽象之间联系紧密，可以同时进行修改和操作。
pytorch的主要特点：
* Numpy风格的Tensor操作。pytorch中tensor提供的API参考了Numpy的设计，因此熟悉Numpy的用户基本上可以无缝理解，并创建和操作tensor，同时torch中的数组和Numpy数组对象可以无缝的对接。
* 变量自动求导。在一序列计算过程形成的计算图中，参与的变量可以方便的计算自己对目标函数的梯度。这样就可以方便的实现神经网络的后向传播过程。
* 神经网络层与损失函数优化等高层封装。网络层的封装存在于torch.nn模块，损失函数由torch.nn.functional模块提供，优化函数由torch.optim模块提供。
* 与tensorflow的静态计算图不同，pytorch的计算图是动态的，可以根据计算需要实时改变计算图。

# pytorch张量
张量是PyTorch里面基础的运算单位,与Numpy的ndarray相同，都表示一个多维的矩阵。
与ndarray的最大区别就是，PyTorch的Tensor可以在GPU上运行，而numpy的ndarray只能在CPU上运行，大大加快了运算速度。
在同构的意义下，第零阶张量为标量(Scalar)，第一阶张量(r = 1)为向量(Vector)，第二阶张量(r = 2)则成为矩阵(Matrix)，第三阶以上的统称为多维张量。
张量有五种基本数据类型：
* 32位浮点型：torch.FloatTensor；
* 64位整型：torch.LongTensor；
* 32位整型：torch.IntTensor；
* 16位整型：torch.ShortTensor；
* 64位浮点型：torch.DoubleTensor。

除了上面的数字类型外，还有byte和chart型。

numpy方法可以将Tensor转为ndarray，from_numpy方法可以将numpy转化为Tensor。
**注意：**
Tensor和numpy对象共享内存，所以他们之间的转换很快。但这也意味着，如果其中一个对象内容改变，另外一个对象也会随之改变。
举例：
```python
import torch

# 打印torch版本
print(torch.__version__)

# 生成一个2行3列的的矩阵,元素为随机数,torch.rand从区间(0,1)的均匀分布中抽取随机数
x = torch.rand(2, 3)
print(x)
# 查看张量可以使用与numpy相同的shape属性查看,也可以使用size()函数,返回的结果相同
print(x.shape, x.size())

# 生成一个标量
scalar = torch.tensor(3.1433223)
# 打印标量
print(scalar)
# 标量可以直接提取出数值
print(scalar.item())
# 打印标量的大小
print(scalar.size())

# numpy方法可以将Tensor转为ndarray
a = torch.randn((3, 2))
numpy_a = a.numpy()
print(numpy_a)

# from_numpy方法可以将numpy转化为Tensor
torch_a = torch.from_numpy(numpy_a)
print(torch_a)
```
我们可以使用.cuda方法将tensor对象移动到gpu，这步操作需要cuda设备支持。如果我们有多GPU，可以使用to方法来确定使用哪块GPU。
举例：
```python
import torch

# cpu上创建一个tensor(tensor对象存储在内存中)
cpu_a = torch.rand(4, 3)
print(cpu_a.type())
# 将该tensor对象移动到GPU显存中
gpu_a = cpu_a.cuda()
print(gpu_a.type())
# 将该tensor对象从GPU显存再移回内存中
cpu_b = gpu_a.cpu()
print(cpu_b.type())

# 可使用torch.cuda.is_available()来确定是否有cuda设备
print(torch.cuda.is_available())
# 先检测有无GPU，若有，将该tensor张量移动到GPU显存中
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
gpu_b = cpu_b.to(device)
print(gpu_b.type())
# 只有一块GPU时,将该tensor对象移动到GPU显存中
# gpu_a = cpu_a.cuda()
# 假如有多块GPU时,将该tensor对象转移到第2、3、4...块GPU上
# gpu_a=cpu_a.cuda(1)
```
Pytorch中张量的初始化方法有：.rand()均匀分布初始化，初始化为0或1，对角线初始化等。
```python
import torch

# 使用[0,1]均匀分布随机初始化矩阵
rnd = torch.rand(5, 3)
print(rnd)
# 初始化值均为1
one = torch.ones(2, 2)
print(one)
# 初始化值均为0
zero = torch.zeros(2, 2)
print(zero)
# 初始化一个单位矩阵,对角线为1,其他为0
eye = torch.eye(2, 2)
print(eye)
```
PyTorch中对张量的操作和NumPy非常相似。
```python
import torch

x = torch.randn(3, 3)
print(x)
# 按行取最大值和最大值的下标
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)
# 按行求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
# 两个相同规格矩阵相加
y = torch.randn(3, 3)
z = x + y
print(z)
# 以_结尾调用的方法会改变该对象的值，如下面x.add_会改变x的值
x.add_(y)
print(x)
```
# pytorch Autograd:自动求导机制
pytorch的autograd包为张量上的所有操作提供了自动求导。
torch.Tensor是这个包的核心类。如果设置.requires_grad 为True，那么将会追踪所有对于该张量的计算操作。当完成计算后通过调用.backward()，自动计算所有的梯度，这个张量的所有梯度将会自动积累到.grad属性。
Tensor 和 Function互相连接并生成一个非循环图，它表示和存储了完整的计算历史。每个张量都有一个.grad_fn属性，这个属性引用了一个创建Tensor的Function(除非这个张量是用户手动创建的，那么这个张量的
grad_fn 是 None)。
要阻止张量跟踪历史记录，可以调用.detach()方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。为了防止跟踪计算历史记录(和使用内存)，可以将代码块包装在with torch.no_grad():中。这在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练参数，但是我们不需要梯度计算。
```python
import torch

# 创建一个张量并设置requires_grad=True用来追踪他的计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)
# 张量进行一次计算
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
# .requires_grad可以改变现有张量的requires_grad属性,默认False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
# 反向传播,因为out是一个标量,out.backward()等于out.backward(torch.tensor(1))
out.backward()
# 打印反向传播后的x值
print(x.grad)

# 如果.requires_grad=True但是你不希望进行autograd的计算,那么可以将变量包裹在with torch.no_grad()中
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
   print((x ** 2).requires_grad)
```
# torchvision包介绍
一般情况下我们处理图像、文本、音频和视频样本数据时，可以使用标准的Python包来加载数据到一个numpy数组中。然后把这个数组转换成 torch.Tensor。
* 对于图像，可以使用 Pillow, OpenCV
* 对于音频，可以使用 scipy, librosa
* 对于文本，可以使用Python和Cython来加载，或者使用 NLTK或SpaCy 处理

torchvision包含了处理一些常用的图像数据集。这些数据集包括Imagenet, CIFAR10, MNIST等。除了数据加载以外，torchvision还包含了图像转换器、torchvision.datasets和torch.utils.data.DataLoader。
# pytorch训练、验证、测试实例
mnist训练集有60000个样本，测试集有10000个样本，将训练集分为50000个样本的训练集和10000个样本的验证集，每训练一个epoch(50000个样本)就使用10000个样本的验证集来测试模型准确率，训练10个epoch后使用测试集来测试模型对测试集的准确率。
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10
# 自动下载pytorch提供的mnist数据集,可能需要代理才能下载下来,我们这里以mnist数据集为例说明pytorch中交叉验证的用法
# transforms.ToTensor()将shape为(H,W,C)的nump.ndarray或img转为shape为(C,H,W)的tensor,其将每一个数值归一化到[0,1]直接除以255即可
# H和W分别为28和28,C为1
# transforms.Normalize((0.1307,), (0.3081,))将数据集按均值0.1307、标准差0.3081进行标准化,这个均值和标准差是官方例子给的
# 取得训练集
train_data_1 = datasets.MNIST("data", train=True, download=True, transform=transforms.Compose(
   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
# 从训练集中取一个batch_size的样本用来训练,取时打乱数据
train_batch_data = DataLoader(train_data_1, batch_size=batch_size, shuffle=True)
# 取得测试集
test_data = datasets.MNIST("data", train=False, transform=transforms.Compose(
   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
# 从测试集中取一个batch_size的样本用来测试,取时打乱数据
test_batch_data = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# print("训练集大小:{},测试集大小:{}".format(len(train_data_1), len(test_data)))
#  torch.utils.data.random_split按照给定的长度将数据集划分成没有重叠的新数据集组合,50000+10000应当等于原有训练集的大小
train_data_1, validation_data_2 = torch.utils.data.random_split(train_data_1, [50000, 10000])
# 训练集重新划分成两部分,第一部分大小50000,第二部分大小10000
# print("训练集第一部分:{},训练集第二部分:{}".format(len(train_data_1), len(train_data_2)))
# 重新划分的两部分训练集各取一个batch_size的样本
train_batch_data_1 = torch.utils.data.DataLoader(train_data_1, batch_size=batch_size, shuffle=True)
validation_batch_data_2 = torch.utils.data.DataLoader(validation_data_2, batch_size=batch_size, shuffle=True)


class FullyConnectedNeuralNet(nn.Module):
   def __init__(self):
      super(FullyConnectedNeuralNet, self).__init__()
      # 定义一个全连接神经网络结构,每层使用Leakly ReLU激活函数
      self.model = nn.Sequential(
         nn.Linear(784, 200),
         nn.LeakyReLU(inplace=True),
         nn.Linear(200, 200),
         nn.LeakyReLU(inplace=True),
         nn.Linear(200, 10),
         nn.LeakyReLU(inplace=True),
      )

   # 定义前向网络
   def forward(self, x):
      x = self.model(x)

      return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = FullyConnectedNeuralNet().to(DEVICE)
# 定义优化算法为SGD,定义损失函数为交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(DEVICE)
iteration = int(len(train_data_1) / batch_size)
# 训练epochs轮
for epoch in range(epochs):
   # train_batch_data_1每次从train_data_1训练集中取出batch_size个样本进行训练
   for iter_index, (data, label) in enumerate(train_batch_data_1):
      # 改变张量形状为(len(train_batch_data),28*28)
      data = data.view(-1, 28 * 28)
      # 训练数据和训练标签都放到GPU中训练
      data, label = data.to(DEVICE), label.to(DEVICE)
      # 计算前向网络最后结果
      preds = net(data)
      # 计算损失函数结果
      loss = criteon(preds, label)
      # pytorch中计算时梯度是被积累的,但是每个batch_size训练后梯度需要清零
      optimizer.zero_grad()
      # 每轮iteration反向传播梯度更新参数
      loss.backward()
      # optimizer.step()用在每个batch中,使用optimizer.step(),模型参数才会更新
      optimizer.step()
      # 每训练100轮iteration100轮打印loss值
      if iter_index % 100 == 0:
         print("epoch:{} [{}/{}({:.1f}%)],train_loss:{:.4f}".format(epoch, iter_index * len(data),
                                                                    len(train_batch_data_1.dataset),
                                                                    100.0 * iter_index / len(train_batch_data_1),
                                                                    loss.item()))

   # test_loss为验证集的平均loss,acc为预测准确率
   validation_loss = 0
   validation_acc = 0
   # 每一轮epoch训练完后从validation_data_2中取出batch_size个样本用来验证
   for data, label in validation_batch_data_2:
      data = data.view(-1, 28 * 28)
      data, label = data.to(DEVICE), label.to(DEVICE)
      preds = net(data)
      validation_loss += criteon(preds, label).item()
      # 求每个预测向量中最大值的下标即为预测的标签
      # 第一个1是指在1号维度上找每个Tensor的最大值
      # 下标[0]为每个Tensor的最大值,[1]为每个Tensor最大值对应的index下标
      pred_label = preds.data.max(1)[1]
      # 计算预测标签与真实标签是否相同,相同说明预测正确
      validation_acc += pred_label.eq(label.data).sum()

   validation_loss /= len(validation_batch_data_2.dataset)
   print("Validation average loss:{:.4f},Validation accuracy:{}/{}({:.2f}%)".format(validation_loss, validation_acc,
                                                                                    len(
                                                                                       validation_batch_data_2.dataset),
                                                                                    100.0 * validation_acc / len(
                                                                                       validation_batch_data_2.dataset)))

# 测试集平均loss和准确率
test_loss = 0
test_acc = 0
# 从test_data中取出batch_size个样本用来测试
for data, label in test_batch_data:
   data = data.view(-1, 28 * 28)
   data, label = data.to(DEVICE), label.to(DEVICE)
   preds = net(data)
   test_loss += criteon(preds, label).item()
   pred_label = preds.data.max(1)[1]
   test_acc += pred_label.eq(label.data).sum()

test_loss /= len(test_batch_data.dataset)
print("Test average loss:{:.4f},Test accuracy:{}/{}({:.2f}%)".format(test_loss, test_acc, len(test_batch_data.dataset),
                                                                     100.0 * test_acc / len(test_batch_data.dataset)))
```
# pytorch搭建逻辑回归神经网络
logistic回归就是在线性回归函数y=wx+b的基础上再加一层非线性函数，一般就是加sigmoid函数，因为Sigmod函数的输出的是是对于0到1之间的概率值，当概率大于0.5预测为1，小于0.5预测为0。
我们使用UCI German Credit数据集，这是UCI的德国信用数据集，里面有原数据和数值化后的数据。German Credit数据是根据个人的银行贷款信息和申请客户贷款逾期发生情况来预测贷款违约倾向的数据集，数据集包含24个维度的1000条数据。
数据集下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/ 。
我们这里使用german.data-numeric数据集，这是numpy处理好的数值化数据，直接使用numpy的load方法读取即可。
下面是代码实现：
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 载入数据
data = np.loadtxt("german.data-numeric")

# 数据归一化处理
row, col = data.shape
for j in range(col - 1):
   # 每列求期望和方差
   meanVal = np.mean(data[:, j])
   stdVal = np.std(data[:, j])
   # 数据归一化
   data[:, j] = (data[:, j] - meanVal) / stdVal

# 打乱数据顺序
np.random.shuffle(data)
# 前900条作为训练集,后100条作为测试集
# 标签全部减一是因为原本标签值为1或2,减一后变为0或1,符合sigmoid函数要求
train_data = data[:900, :col - 1]
train_label = data[:900, col - 1] - 1
test_data = data[900:, :col - 1]
test_label = data[900:, col - 1] - 1


# 定义逻辑回归模型
class LR(nn.Module):
   def __init__(self):
      super(LR, self).__init__()
      # 输入24维度，输出2维度
      self.fc = nn.Linear(24, 2)

   # 输出的2维度再经过sigmoid函数计算
   def forward(self, x):
      out = self.fc(x)
      out = torch.sigmoid(out)
      return out


# 测试函数
def test(pred, lab):
   t = pred.max(-1)[1] == lab
   return torch.mean(t.float())


net = LR()
# 损失函数使用CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# 优化算法使用Adam
optm = torch.optim.Adam(net.parameters())
epochs = 10000

# 训练模型
for i in range(epochs):
   # 指定模型为训练模式，计算梯度
   net.train()
   # 输入样本都需要转化成torch的Tensor
   x = torch.from_numpy(train_data).float()
   y = torch.from_numpy(train_label).long()
   y_hat = net(x)
   # 计算损失
   loss = criterion(y_hat, y)
   # 前一轮的梯度归零
   optm.zero_grad()
   # 反向传播
   loss.backward()
   # 更新参数
   optm.step()
   # 每100次epoch验证一下准确率
   if (i + 1) % 100 == 0:
      # 指定模型为计算模式
      net.eval()
      test_in = torch.from_numpy(test_data).float()
      test_l = torch.from_numpy(test_label).long()
      test_out = net(test_in)
      # 使用测试函数计算准确率
      accu = test(test_out, test_l)
      print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))
```
# pytorch搭建CNN神经网络（mnist手写体数字分类）
下面是代码实现：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# batch_size=512大约需要2G显存
batch_size = 512
epochs = 5
# 让torch判断,如果有GPU则使用GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载训练集
trainloader = torch.utils.data.DataLoader(
   datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
      transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)
# 加载测试集
testloader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, transform=transforms.Compose([
   transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)


# 定义模型
class ConvNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
      self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
      self.fc1 = nn.Linear(20 * 10 * 10, 500)
      self.fc2 = nn.Linear(500, 10)

   def forward(self, x):
      in_size = x.size(0)
      out = self.conv1(x)
      out = F.relu(out)
      out = F.max_pool2d(out, kernel_size=2, stride=2)
      out = self.conv2(out)
      out = F.relu(out)
      out = out.view(-1, 20 * 10 * 10)
      out = self.fc1(out)
      out = F.relu(out)
      out = self.fc2(out)
      out = F.log_softmax(out, dim=1)
      return out


# 创建网络
net = ConvNet().to(DEVICE)
# 优化算法选择Adam
op = optim.Adam(net.parameters())


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
   model.train()
   for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      # 损失函数NLLLoss()输入是一个对数概率向量和一个目标标签,它不会为我们计算对数概率,适合最后一层是log_softmax()的网络
      # 即CrossEntropyLoss()=log_softmax() + NLLLoss()
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      # 每训练30个batch计算一下loss值
      if (batch_idx + 1) % 30 == 0:
         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            loss.item()))


# 测试函数
def test(model, device, test_loader):
   model.eval()
   test_loss = 0
   correct = 0
   for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
      pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
      correct += pred.eq(target.view_as(pred)).sum().item()

   test_loss /= len(test_loader.dataset)
   print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


# 先训练再测试
for epoch in range(1, epochs + 1):
   train(net, DEVICE, trainloader, op, epoch)
   test(net, DEVICE, testloader)
```

# pytorch搭建CNN神经网络(CIFAR10分类)
我们使用CIFAR10数据集，它有如下10个类别:‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’,‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。CIFAR-10的图像都是3x32x32大小的，即3颜色通道，32x32像素。
**网络构建过程：**
* 使用torchvision加载和归一化CIFAR10训练集和测试集；
* 定义一个卷积神经网络；
* 定义损失函数；
* 在训练集上训练网络；
* 在测试集上测试网络。

**使用GPU进行计算的方法：**
如果要使用GPU进行训练，你安装的pytorch包必须是可支持CUDA的版本。在编码代码时，请将整个网络的权重、输入样本的数据和标签、预测样本的数据和标签在定义完成后统统使用 .to() 方法移动到GPU显存中进行计算。

下面是代码实现：
```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchvision输出为[0,1]的PILImage图像，我们把它转换为归一化为[-1, 1]的张量
# ToTensor()将shape为(H,W,C)的nump.ndarray或img转为shape为(C,H,W)的tensor,其将每一个数值归一化到[0,1]，即/255
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))将每个元素归一化到[-1, 1]的张量
# input[channel] = (input[channel] - mean[channel]) / std[channel],前三个数是std,后三个数是mean
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# download为True,自动下载数据集;train为True加载训练集，False时加载测试集
# transform表示是否需要对数据进行预处理,这里直接用了我们上面定义的transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# batch_size每批训练输入多少样本;shuffle每次迭代时是否打乱顺序;num_workers默认是0,即使用多少个子进程来导入数据
# 注意使用子进程导入数据,则整个代码块前面必须加上if __name__ == '__main__'
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
# 定义类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 用于显示图像的函数
def imshow(img):
   img = img / 2 + 0.5
   npimg = img.numpy()
   # 张量转换为numpy数组，然后把维度调换一下,因为张量维度为(C,H,W),(H,W,C)才是用来显示图片的维度
   # np.transpose原始维度编号为(0,1,2)
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()


# 随机获取4张图片的张量数据
dataiter = iter(trainloader)
images, labels = dataiter.next()
# 显示图像
imshow(torchvision.utils.make_grid(images))
# 显示图像标签
print(" ".join("%5s" % classes[labels[j]] for j in range(4)))

start_epoch = 0
start_batch = 0
epochs = 5

# 使用GPU来训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 定义一个神经网络模型
class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      # 定义一组基本操作,先设定好这些操作的具体参数
      # 二维卷积输入尺度是(N,C_in,H,W),输出尺度(N,C_out,H_out,W_out)
      self.out_channel_1 = 32
      self.out_channel_2 = 64
      self.kernel_size = 5
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.out_channel_1, kernel_size=self.kernel_size)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv2 = nn.Conv2d(in_channels=32, out_channels=self.out_channel_2, kernel_size=self.kernel_size)
      # 默认bias=True
      self.fc1 = nn.Linear(in_features=self.out_channel_2 * self.kernel_size * self.kernel_size, out_features=768)
      self.fc2 = nn.Linear(in_features=768, out_features=120)
      self.fc3 = nn.Linear(in_features=120, out_features=10)

   # 定义forward()函数,即定义了整个模型的计算步骤
   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, self.out_channel_2 * self.kernel_size * self.kernel_size)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x


# 建立网络,使用GPU训练时网络中参数全部移动到GPU显存中
net = Net().to(device)
# 定义使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义使用Adam优化算法,学习率0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 如果存在保存的模型，那么加载保存的模型
if os.path.isdir("model"):
   try:
      net_record = torch.load("./model/save1.t7")
      # 从字典中读取上次保存的参数
      net.load_state_dict(net_record["state"])
      start_epoch = net_record["epoch"]
      start_batch = net_record["batch"]
   except FileNotFoundError:
      print("Can\'t found ./model/save1.t7")
else:
   # 否则epoch从第0轮开始
   start_epoch = 0
   start_batch = 0

# 训练网络,这里start_epoch就是我们读取的模型上次训练达到的eopch轮数,如果你不想再训练本轮,下面改成start_epoch+1就会直接跳过训练
for epoch in range(start_epoch + 1, epochs):
   # 指定模型为训练模式，计算梯度
   net.train()
   running_loss = 0.0
   for i, (inputs, labels) in enumerate(trainloader, 0):
      # 获取一个batch_size样本的inputs和labels;在GPU上训练时，网络权重和输入样本及标签都要移动到GPU显存中
      inputs, labels = inputs.to(device), labels.to(device)
      # pytorch中的backward()函数的计算时梯度是被积累的而不是被替换掉
      # 但是在每一个batch时并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch开始时设置zero_grad将梯度重置为0
      optimizer.zero_grad()

      # 正向传播
      outputs = net(inputs)
      # 计算loss
      loss = criterion(outputs, labels)
      # 反向传播计算梯度,更新参数
      loss.backward()
      # optimizer.step()用在每个batch中,使用optimizer.step(),模型参数才会更新
      # scheduler的step_size表示scheduler.step()每调用step_size次,对应的学习率就会按照策略调整一次
      # 如果scheduler.step()是放在batch里面，那么step_size指的是经过这么多次迭代,学习率改变一次
      optimizer.step()

      # 打印loss值,这个loss是每2000次迭代的loss平均值
      running_loss += loss.item()
      if i % 2000 == 1999 or i == 0:
         print("epoch={},i={:5d},loss={:.3f}".format(epoch + 1, i + 1, running_loss / 2000))
         running_loss = 0.0
         # 用字典形式保存模型参数、epoch和batch，加载时需要先加载网络,然后将参数写入网络
         state = {"state": net.state_dict(), "epoch": epoch, "batch": i}
         # 模型保存在.py文件目录中model文件夹内,如果没有这个文件夹就创建一个
         if not os.path.isdir("model"):
            os.mkdir("model")
         # pytorch保存数据的格式为.t7文件或者.pth文件,t7文件是沿用torch7中读取模型权重的方式
         # pth文件是python中存储文件的常用格式
         torch.save(state, "./model/save1.t7")

# 加载模型
net = Net().to(device)
net_record = torch.load("./model/save1.t7")
net.load_state_dict(net_record["state"])
# 打印读取的模型训练保存时的epoch和batch
print("epoch={},batch={}".format(net_record["epoch"], net_record["batch"]))

# 测试网络
correct = 0
total = 0
# 测试时必须调用.eval()方法将网络设为评估模式
net.eval()
# 评估模型时,我们不需要这些参数参与自动求导,使用torch.no_grad()禁止跟踪计算记录
for (images, labels) in testloader:
   # 测试时必须调用.eval()方法将网络设为评估模式
   net.eval()
   images, labels = images.to(device), labels.to(device)
   outputs = net(images)
   _, predicted = torch.max(outputs.data, 1)
   total += labels.size(0)
   correct += (predicted == labels).sum().item()

print("Accuracy of the network on the 10000 test images:{}%".format(100 * correct / total))

# 计算一下测试每种类图片时的预测准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for (images, labels) in testloader:
   # 测试时必须调用.eval()方法将网络设为评估模式
   net.eval()
   images, labels = images.to(device), labels.to(device)
   outputs = net(images)
   _, predicted = torch.max(outputs, 1)
   c = (predicted == labels).squeeze()
   for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1

for i in range(10):
   print("Accuracy of {}:{}%".format(classes[i], 100 * class_correct[i] / class_total[i]))
```