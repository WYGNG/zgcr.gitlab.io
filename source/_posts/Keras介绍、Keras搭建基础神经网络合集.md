---
title: Keras介绍、Keras搭建基础神经网络合集
date: 2019-03-02 19:54:16
tags:
- Keras
categories:
- Keras
---

# Keras介绍
Keras是一个构建神经网络进行训练和测试的高级神经网络API，由Python语言编写而成，可使用TensorFlow、Theano及CNTK作为后端。
严格意义上讲，Keras并不能称为一个深度学习框架，它更像一个深度学习接口，它构建于第三方框架之上。Keras的过度封装导致其不够灵活，同时也导致其运行速度较慢。
但Keras学习起来最简单，另外，利用Keras去验证一些简单模型时非常的方便。
目前Keras的最新版本号为2.2.4。
# Keras搭建线性回归神经网络
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import os

# 指定backend框架为tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
# 指定警告等级,这样运行时不出现红字提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定使用的GPU编号,从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建一组数据X和Y
X = np.linspace(-1, 1, 200)
# 打乱X数据顺序
np.random.seed(1337)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))
# 将前160个数据样本作为训练数据，后40个数据样本作为测试数据
x_train, y_train = X[:160], Y[:160]
x_test, y_test = X[160:], Y[160:]

# 建立模型
model = Sequential()
# 输入和输出维度均为一维
model.add(Dense(input_dim=1, units=1))
# 添加下一个神经层的时候，不用再定义输入的维度，因为它默认就把前一层的输出作为下一层的输入
# loss函数用mse均方误差,优化器用sgd随机梯度下降
model.compile(loss='mse', optimizer='sgd')

# 训练模型
iteration = 1000
for step in range(iteration):
   # 用model.train_on_batch每次使用x_train,y_train尺寸的样本训练,默认的返回值是cost
   cost = model.train_on_batch(x_train, y_train)
   if step % 100 == 0:
      print('iteration:{} cost:{}'.format(step, cost))

# 测试模型
cost = model.evaluate(x_test, y_test, batch_size=40)
print('Test cost:{}'.format(cost))
w, b = model.layers[0].get_weights()
print('Weights:{},biases={}.'.format(w, b))

# 可视化训练结果
# 在图上画出我们的训练集数据点,用红色点表示
plt.scatter(x_train, y_train, color="r")
# 在图上画出我们的测试集数据点,用绿色点表示
plt.scatter(x_test, y_test, color="g")
# 画出模型预测的直线
y_test_prediction = model.predict(x_test)
plt.plot(x_test, y_test_prediction, color="b")
plt.show()
```
# Keras搭建分类神经网络(mnist手写体数字分类)
```python
from __future__ import print_function
import keras
import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras import backend as K

# 指定backend框架为tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
# 指定警告等级,这样运行时不出现红字提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定使用的GPU编号,从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#
# if K.image_data_format() == 'channels_first':
#  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#  input_shape = (1, img_rows, img_cols)
# else:
#  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#  input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# # mnist数据集的标签y要用到Keras改造的numpy的一个函数np_utils.to_categorical,把标签y变成one-hot编码形式
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# 如果不能自动下载数据集,那么手动从这里下载数据集:https://pan.baidu.com/s/1jH6uFFC 密码: dw3d
# 把下载好的数据集放到本.py文件所在目录下,然后添加:
path = './mnist.npz'
mnist_data = np.load(path)
x_train, y_train = mnist_data['x_train'], mnist_data['y_train']
x_test, y_test = mnist_data['x_test'], mnist_data['y_test']
mnist_data.close()
# 对x_train和x_test的数据进行0-1标准化
x_train = x_train.reshape(x_train.shape[0], -1) / 255.
x_test = x_test.reshape(x_test.shape[0], -1) / 255.
# mnist数据集的标签y要用到keras.utils.to_categorical把标签y变成one-hot编码形式
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
# print(x_train[0], y_train[0])
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)


# 创建模型
model = Sequential()
# 两层全连接神经网络
model.add(Dense(units=32, input_dim=784, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
# 设置损失函数、优化算法Adam、评价函数accuracy
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# verbose为日志显示级别,=0为不在标准输出流输出日志信息,=1为输出进度条记录,=2为每个epoch输出一行记录
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
# model.evaluate()返回的是损失值和你选定的评价函数值
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:{};Test accuracy:{}.'.format(score[0], score[1]))
```

# Keras搭建CNN神经网络(mnist手写体数字识别)
```python
from __future__ import print_function
import keras
import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras import backend as K

# 指定backend框架为tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
# 指定警告等级,这样运行时不出现红字提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定使用的GPU编号,从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#
# if K.image_data_format() == 'channels_first':
#  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#  input_shape = (1, img_rows, img_cols)
# else:
#  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#  input_shape = (img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# # mnist数据集的标签y要用到Keras改造的numpy的一个函数np_utils.to_categorical,把标签y变成one-hot编码形式
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# 如果不能自动下载数据集,那么手动从这里下载数据集:https://pan.baidu.com/s/1jH6uFFC 密码: dw3d
# 把下载好的数据集放到本.py文件所在目录下,然后添加:
path = './mnist.npz'
mnist_data = np.load(path)
x_train, y_train = mnist_data['x_train'], mnist_data['y_train']
x_test, y_test = mnist_data['x_test'], mnist_data['y_test']
mnist_data.close()
# 对x_train和x_test的数据进行0-1标准化
x_train = x_train.reshape(-1, 28, 28, 1) / 255.
x_test = x_test.reshape(-1, 28, 28, 1) / 255.
# mnist数据集的标签y要用到keras.utils.to_categorical把标签y变成one-hot编码形式
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
# print(x_train[0], y_train[0])
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

model = Sequential()
# 两层卷积+一层池化+一层dropout
model.add(
   Conv2D(filters=32, strides=1, kernel_size=(3, 3), padding='same', activation='relu', data_format='channels_last',
          input_shape=input_shape))
model.add(Conv2D(filters=64, strides=1, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.25))
model.add(Flatten())
# 全连接层
model.add(Dense(128))
# 输出层
model.add(Dense(num_classes, activation='softmax'))
# 设置损失函数、优化算法Adam、评价函数accuracy
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr=1e-4), metrics=['accuracy'])
# 输入训练样本进行训练,batch_size=128
# verbose为日志显示级别,=0为不在标准输出流输出日志信息,=1为输出进度条记录,=2为每个epoch输出一行记录
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
# model.evaluate()返回的是损失值和你选定的评价函数值
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:{};Test accuracy:{}.'.format(score[0], score[1]))
```
