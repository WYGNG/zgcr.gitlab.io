---
title: Tensorflow基础知识介绍、Tensorflow搭建基础神经网络合集
date: 2019-03-03 18:37:38
tags:
- Tensorflow
categories:
- Tensorflow
---

# Tensorflow介绍
Tensorflow由Google开发，是GitHub上最受欢迎的机器学习/深度学习库之一。TensorFlow的核心概念是节点、张量和计算图。
节点一般表示施加的某个数学操作，也可以表示数据输入的起点/输出的终点，或者是读取/写入持久变量的终点。线表示之间的输入/输出关系。张量可以理解为是不同维度的矩阵，张量沿着线的方向在计算图中流动，这就是取名为Tensorflow的原因。
需要注意的是，在TensorFlow中，我们必须要先完整地构建一个计算图，然后按照计算图启动一个会话，然后才能在会话中完成变量的初始化和计算，最终得到结果。

# Tensorflow张量、placeholder占位符、Session会话
在TensorFlow中，可以将张量理解为数组。如果是0阶张量，那么这个张量是一个标量，也就是一个数字，如果是一阶张量可以理解为向量或者是一维数组，n阶张量可以理解为n维的数组。张量中包含了三个重要的属性，名字、维度、类型。
TensorFlow张量的实现并没有直接采用数组的形式，张量它只是对运算结果的引用，如果我们直接print某个张量，我们只能打印出一个指向该张量内容的指针，而不能打印出张量的内容。
如果我们想创建两个张量，但这两个张量的值需要从外部输入，这时我们可以用tf.placeholder创建placeholder占位符张量，占位符张量的值可以在Session会话时输入。
Session会话会话用来执行定义好的运算图，我们在会话中初始化所有变量，输入样本数据后，就可以开始计算了。当计算完成之后，需要通过关闭会话来帮助系统回收资源，否则可能导致资源泄露的问题。
举例：
```python
import tensorflow as tf

# 张量,采用高斯分布初始化和常量初始化
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=10))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=10))

# 定义占位符
x = tf.placeholder(tf.float32, shape=(1, 2), name="input")

# 前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 创建会话开始计算,with as这种形式创建的会话执行完运算后会自动关闭会话
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
```
我们还可以定义多个计算图，在创建会话时指定某个计算图进行计算。
我们还可以通过tf.ConfigProto()函数，配置会话并行的线程数、GPU的分配策略、运算超时等参数。
举例：
```python
import tensorflow as tf

# 创建g1计算图
g1 = tf.Graph()
with g1.as_default():
   v = tf.get_variable('v', [1], initializer=tf.zeros_initializer())

# 创建g2计算图
g2 = tf.Graph()
with g2.as_default():
   v = tf.get_variable("v", [2], initializer=tf.ones_initializer())

# 创建一个会话计算g1计算图
with tf.Session(graph=g1) as sess:
   tf.global_variables_initializer().run()
   # 创建一个命名空间,将参数reuse设置为True.这样tf.get_variable函数将直接获取已经声明的变量
   with tf.variable_scope("", reuse=True):
      print(sess.run(tf.get_variable("v")))

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
c = a + b
print(c)
# tf.InteractiveSession()是一种交互式的session方式，它让自己成为默认的session
# 用户在不需要指明用哪个session运行的情况下,就可以运行起来,run()和eval()函数都可以不指明是哪个session
sess = tf.InteractiveSession()
print(c.eval())
sess.close()
# 创建一个默认会话,会运行g1计算图之后到这个会话之前中间定义的计算图
with tf.Session() as sess:
   print("a+b:", sess.run(c))

# tf.ConfigProto()函数可配置会话并行的线程数、GPU的分配策略、运算超时等参数
# allow_soft_placement=True:如果你指定的设备不存在，允许TF自动分配设备;log_device_placement=True为是否打印设备分配日志
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
print("configproto:")
print(sess1.run(c))
print(sess2.run(c))
```
# 使用tf.data.Dataset加载数据，加快训练过程
使用feed_dict加载数据这个过程是CPU完成的，此时GPU空闲未工作，如果我们加载数据需要的时间比较多，就会大大拉长我们的模型训练时间。
我们可以使用tf.data API来加载数据。在tf.data工作流中，能够以异步方式预读取下个批次的数据，这样就可以缩短GPU等待加载数据的时间(如果训练一个batch的时间大于读取下个批次数据的时间，那么GPU就可以连续地进行计算)。
下面是分别使用numpy数组、tensorflow张量、tensorflow占位符、tensorflow生成器创建dataset对象的实例：
```python
import tensorflow as tf
import numpy as np
import os

# 下面的设置可以使这个警告不出现:Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 指定使用的GPU的编号，从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 创建numpy数组形式的数据集
x = np.random.sample((100, 2))
# 创建一个dataset对象,传入数据集x
dataset = tf.data.Dataset.from_tensor_slices(x)
# 用dataset对象创建一个One Shot迭代器,One Shot迭代器在会话中不需要初始化
iter_1 = dataset.make_one_shot_iterator()
# 取迭代器下一个数据的操作
el_1 = iter_1.get_next()
# 创建会话,运行
with tf.Session() as sess:
   print(sess.run(el_1))

# 创建有特征和标签的数据集
features, labels = (np.random.sample((100, 2)), np.random.sample((100, 1)))
# 创建dataset对象,再创建One Shot迭代器和取迭代器下一个数据的操作
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
iter_2 = dataset.make_one_shot_iterator()
el_2 = iter_2.get_next()

with tf.Session() as sess:
   # 连续取5次迭代器下一个数据
   for i in range(3):
      print(sess.run(el_2))

# 直接使用tensorflow张量创建dataset对象
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))
# 这里要创建可初始化迭代器,定义取迭代器下一个元素对象操作,可初始化迭代器在会话中必须先初始化
iter_3 = dataset.make_initializable_iterator()
el_3 = iter_3.get_next()

with tf.Session() as sess:
   sess.run(iter_3.initializer)
   for i in range(3):
      print(sess.run(el_3))

# 使用tensorflow占位符变量创建dataset对象
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))
features, labels = (np.random.sample((100, 2)), np.random.sample((100, 1)))
# 这里要创建可初始化迭代器,定义取迭代器下一个元素对象操作,可初始化迭代器在会话中必须先初始化和定义feed_dict
iter_4 = dataset.make_initializable_iterator()
el_4 = iter_4.get_next()

with tf.Session() as sess:
   sess.run(iter_4.initializer, feed_dict={x: features, y: labels})
   for i in range(3):
      print(sess.run(el_4))

# 从生成器创建dataset对象
sequence = np.array([[[1]], [[2], [3]], [[4], [5], [6]]])


# 创建一个生成器
def generator():
   for el in sequence:
      yield el


# 使用生成器创建dataset对象
dataset = tf.data.Dataset().batch(1).from_generator(generator, output_types=tf.int64,
                                                    output_shapes=(tf.TensorShape([None, 1])))
# 创建可初始化迭代器
iter_5 = dataset.make_initializable_iterator()
el_5 = iter_5.get_next()

with tf.Session() as sess:
   sess.run(iter_5.initializer)
   for i in range(3):
      print(sess.run(el_5))
```
如果我们想在神经网络训练和测试时使用dataset对象取训练样本和测试样本，可看下面这个用例：
```python
import tensorflow as tf
import numpy as np
import os

# 下面的设置可以使这个警告不出现:Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 指定使用的GPU的编号，从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 4
train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
test_data = (np.random.sample((10, 2)), np.random.sample((10, 1)))
x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE)
# shuffle()来打乱数据集样本顺序,参数buffer_size即下一个元素将从该buffer_size大小的缓存中选取
dataset = dataset.shuffle(buffer_size=100)
iter_1 = dataset.make_initializable_iterator()
features, labels = iter_1.get_next()

with tf.Session() as sess:
   sess.run(iter_1.initializer, feed_dict={x: train_data[0], y: train_data[1]})
   for i in range(3):
      print(sess.run([features, labels]))
   sess.run(iter_1.initializer, feed_dict={x: test_data[0], y: test_data[1]})
   print(sess.run([features, labels]))
```
# Tensorflow保存和恢复模型
tensorflow中保存和恢复模型需要创建一个tf.train.Saver对象，使用.save()方法即可保存，使用.restore()方法可恢复模型。注意保存时的目录必须先手动建好，否则无法保存。
用例如下：
```python
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "./tmp/model.ckpt"
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# 定义x和y占位符
X, Y = tf.placeholder(tf.float32, [None, n_input]), tf.placeholder(tf.float32, [None, n_classes])


# 定义模型
def multilayer_perceptron(x, w, b):
   layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w["h1"]), b["b1"]))
   layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w["h2"]), b["b2"]))
   out_layer = tf.matmul(layer_2, w["out"]) + b["out"]
   return out_layer


# 定义weights和biases
weights = {
   "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
   "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
   "out": tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
   "b1": tf.Variable(tf.random_normal([n_hidden_1])),
   "b2": tf.Variable(tf.random_normal([n_hidden_2])),
   "out": tf.Variable(tf.random_normal([n_classes]))
}
# 预测值
pred = multilayer_perceptron(X, weights, biases)
# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

if not os.path.exists("./tmp/"):
   os.mkdir("./tmp/")
# 定义saver对象，用来保存/恢复模型
saver = tf.train.Saver(max_to_keep=10)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # 训练模型
   for epoch in range(5):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples / batch_size)
      for i in range(total_batch):
         batch_x, batch_y = mnist.train.next_batch(batch_size)
         _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
         avg_cost += c / total_batch
      # 每一轮epoch后打印一次cost值
      if epoch % display_step == 0:
         print("Epoch={:04d},cost={:.9f}".format((epoch + 1), avg_cost))
         # 保存模型
         save_path = saver.save(sess, model_path, global_step=epoch)
         print("模型保存到文件夹:{}".format(save_path))

   # 测试模型准确率
   correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   print("Test accuracy={}".format(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})))

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # 恢复模型
   if os.path.exists("./tmp/checkpoint"):
      # 判断最新的保存模型检查点是否存在，如果存在则从最近的检查点恢复模型
      saver.restore(sess, tf.train.latest_checkpoint("./tmp/"))
   saver.restore(sess, model_path)
   # 继续训练模型
   for epoch in range(5):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples / batch_size)
      for i in range(total_batch):
         batch_x, batch_y = mnist.train.next_batch(batch_size)
         _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
         avg_cost += c / total_batch
      # 每过一轮epoch打印cost值
      if epoch % display_step == 0:
         print("Epoch={:04d},cost={:.9f}".format((epoch + 1), avg_cost))
         # 保存模型
         save_path = saver.save(sess, model_path, global_step=epoch)
         print("模型保存到文件夹:{}".format(save_path))

   # 测试模型准确率
   correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   print("Test accuracy={}".format(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})))
```
# Tensorflow GPU使用设置/log日志显示级别设置
设置使用哪块GPU：
```python
import os

# 只使用第0块GPU，编号是从0开始的，或者同时使用多块GPU，如"0"改为"0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

```
设置log日志级别：
```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="1" # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # 只显示 warning 和 Error 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3" # 只显示 Error  
```
# Tensorboard工具的使用
Tensorboard可以记录与展示以下数据形式:
* 标量Scalars；
* 图片Images；
* 音频Audio；
* 计算图Graph； 
* 数据分布Distribution；
* 嵌入向量Embeddings。

使用tf.summary.scalar记录标量；使用tf.summary.histogram记录数据的直方图；使用tf.summary.distribution记录数据的分布图；使用tf.summary.image记录图像(比如每层卷积出来的特征图显示成图像)。
```python
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = "./tmp/"

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

X = tf.placeholder(tf.float32, [None, 784], name="InputData")
Y = tf.placeholder(tf.float32, [None, 10], name="LabelData")


# 定义模型
def multilayer_perceptron(x, w, b):
   layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w["w1"]), b["b1"]))
   tf.summary.histogram("layer_1", layer_1)
   layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w["w2"]), b["b2"]))
   tf.summary.histogram("layer_2", layer_2)
   out_layer = tf.add(tf.matmul(layer_2, w["w3"]), b["b3"])
   return out_layer


weights = {
   "w1": tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="W1"),
   "w2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="W2"),
   "w3": tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="W3")
}

biases = {
   "b1": tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
   "b2": tf.Variable(tf.random_normal([n_hidden_2]), name="b2"),
   "b3": tf.Variable(tf.random_normal([n_classes]), name="b3")
}
# 每一个需要在tensorboard上显示出来的单独模块都放在一个tf.name_scope()命名空间内
with tf.name_scope("Model"):
   pred = multilayer_perceptron(X, weights, biases)

with tf.name_scope("Loss"):
   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

with tf.name_scope("SGD"):
   # 正常自动进行反向传播,调用minimize方法时,底层进行的工作：
   # 计算trainable_variables集合中所有参数的梯度
   # 将梯度应用到变量上进行梯度下降
   # 在调用sess.run(train_op)时,会对variables进行更新
   optimizer = tf.train.GradientDescentOptimizer(learning_rate)
   # 如果我们希望对梯度进行截断,那么就要自己计算出梯度,然后进行clip,最后对变量进行更新
   # tf.gradients计算出loss对tf.trainable_variables()中所有变量的梯度
   grads = tf.gradients(loss, tf.trainable_variables())
   # 如果要截断梯度(防止梯度爆炸),要使用下面这句
   # 该函数先求所有梯度的L2范数,然后与15比较,如果大于则求缩放因子=15/L2范数
   # 最后将所有梯度乘以这个缩放因子，得到clipped_gradients
   # clipped_gradients, norm = tf.clip_by_global_norm(grads, 15)
   # 返回值clipped_gradients即裁剪后的梯度,norm为全局规约数
   # grads = list(zip(clipped_gradients, tf.trainable_variables()))
   # 将梯度和对应变量打包成元组,再转化成列表
   grads = list(zip(grads, tf.trainable_variables()))
   # 用梯度更新变量
   apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope("Accuracy"):
   acc = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
   acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# tf.summary.scalar用来显示标量信息图
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", acc)

for var in tf.trainable_variables():
   tf.summary.histogram(var.name, var)

for grad, var in grads:
   tf.summary.histogram(var.name + "/gradient", grad)

# 用来汇总所有summary信息,注意只有sess.run()之后才真正汇总
merged_summary_op = tf.summary.merge_all()
# 日志书写器实例化，在实例化的同时将当前计算图写入日志
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # 训练模型
   for epoch in range(training_epochs):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples / batch_size)
      for i in range(total_batch):
         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
         # merged_summary_op只有sess.run()之后才真正汇总所有summary信息
         _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={X: batch_xs, Y: batch_ys})
         # add_summary仅仅是向FileWriter对象的缓存中存放每轮的event data,向disk上写数据是由FileWrite对象控制的
         summary_writer.add_summary(summary, epoch * total_batch + i)
         avg_cost += c / total_batch
      if (epoch + 1) % display_step == 0:
         print("Epoch={:04d},cost={:.9f}".format((epoch + 1), avg_cost))
   # 训练完成后,调用日志书写器实例对象summary_writer的close()方法将对象缓存(即所有记录的summary数据)写入文件
   # 否则它每隔120s自动写入一次
   summary_writer.close()

   # 测试模型
   print("Accuracy={}".format(acc.eval({X: mnist.test.images, Y: mnist.test.labels})))

# 要在tensorboard中查看数据,在保存日志文件的上一级目录中打开cmd.exe,输入下列命令:
# tensorboard --logdir=./tmp/
# 然后在浏览器中输入类似http://zgcr-win10-PC:6006这样的地址即可查看
```
# Tensorflow搭建线性回归神经网络
使用maplotlib画图时，如果有中文字符，一定要设置中文字体，因为matplotlib默认字体不支持中文，无法正常显示。
最好设置输入X和Y的占位符变量，这样我们可以使用feed_dict来灵活地定制要输入的样本数据。如果我们的样本数据是list形式，在输入到feed_dict后能够自动转换成相应的numpy数组，可直接进行tensorflow中的计算。
创建会话后一定要先初始化所有变量后才可以开始迭代训练模型！
如果我们想获得训练过程中的某项数据，找到计算图中定义的该数据变量名，sess.run()它即可，即可获得计算得到的数据(是numpy数组形式)，注意run时看看上面是否载入了feed_dict的样本数据，没载入的话还得重新载入一次。
使用matplotlib画图时必须使用上面sess.run()后得到的计算结果(是numpy数组形式)，tensorflow的张量是不能用来绘图的(张量相当于是个引用，直接print张量只能得到张量的属性信息)!
代码实现如下：
```python
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
# 用于画图时显示中文字符
from pylab import mpl

# 定义matplotlib画图时的全局使用仿宋中文字体，如果不定义成中文字体，有中文时不能正常显示
mpl.rcParams["font.sans-serif"] = ["FangSong"]
# 下面的设置可以使这个警告不出现:Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 指定使用的GPU的编号，从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 超参数
learning_rate = 0.0005
iteration = 5000
display_step = 50

# 训练集数据
sample_number = 1000
# x值从以0为均值,以0.55为标准差的高斯分布中取随机数
train_x = [np.random.normal(0.0, 0.55) for i in range(sample_number)]
# y值在y1=x1*0.1+0.3上小范围浮动
train_y = [train_x[i] * 0.1 + 0.3 + np.random.normal(0.0, 0.03) for i in range(sample_number)]
plt.scatter(train_x, train_y, s=5, color="blue", alpha=0.5)
plt.show()

train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape, train_y.shape)

# 定义X和Y占位符
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 定义模型权重参数,随机分布初始化和零值初始化
# W权重为[1,10]个元素的矩阵,矩阵元素Weight全部从标准正态分布中随机去除的数
# 生成1维的W矩阵,取值是均匀分布[-1,1]之间的随机数;# 生成1维的b矩阵，初始值是0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
# 定义线性回归模型
pred = tf.add(W * X, b)
# 定义损失函数为均方误差损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - Y), name="loss"))
# 定义优化算法为梯度下降
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name="train")

# 创建会话
with tf.Session() as sess:
   # 初始化计算图所有变量
   sess.run(tf.global_variables_initializer())
   # 打印初始化的变量
   print("W={},b={}".format(sess.run(W), sess.run(b)))
   for i in range(iteration):
      sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
      # 每迭代50轮,计算loss值
      if (i + 1) % display_step == 0:
         cost = sess.run(loss, feed_dict={X: train_x, Y: train_y})
         print("Epoch={:04d},cost={:.9f},w={},b={}".format((i + 1), cost, sess.run(W), sess.run(b)))
   # 注意这里已经退出了上面的循环，因此再次sess.run时要重新设置feed_dict
   pred_, cost = sess.run([pred, loss], feed_dict={X: train_x, Y: train_y})
   W_, b_ = sess.run(W), sess.run(b)
   print("Training complete,cost={},W={},b={}".format(cost, W_, b_))

   # 显示数据点和预测曲线
   plt.scatter(train_x, train_y, s=5, color="blue", alpha=0.5)
   plt.plot(train_x, pred_, label="模型预测曲线")
   # 显示图例
   plt.legend()
   plt.show()
```
# Tensorflow搭建第一个逻辑回归神经网络(mnist)手写体数字分类
神经网络搭建过程：
下载mnist数据集；
设置超参数；
设置x和y占位符用来接收输入的样本数据和样本标签；
定义网络权重和网络计算过程；
定义loss函数和优化算法；
创建会话，开始训练；
训练完毕后，测试模型，计算准确率。
**代码实现如下：**
```python
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 下面的设置可以使这个警告不出现:Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 指定使用的GPU的编号，从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mnist = input_data.read_data_sets("/data/", one_hot=True)

learning_rate = 0.05
training_epochs = 25
batch_size = 100
display_step = 1

# x和y占位符,mnist数据集图片shape为28*28=784,标签为one_hot编码
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 创建网络权重
# tf.random_normal从正态分布中取随机数,tf.truncated_normal从截断正态分布中取随机数,tf.constant取指定的常数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 定义网络计算过程
pred = tf.nn.softmax(tf.matmul(X, W) + b)
# 定义损失函数
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), axis=0))
# 优化算法为梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
   # 开始计算前初始化所有变量
   sess.run(tf.global_variables_initializer())
   # 训练模型
   for epoch in range(training_epochs):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples / batch_size)
      for i in range(total_batch):
         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
         _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
         avg_cost += c / total_batch
      # 每轮epoch打印cost值
      if (epoch + 1) % display_step == 0:
         print("Epoch={:04d},cost={:.9f}".format((epoch + 1), avg_cost))

   # 计算模型准确率
   # tf.argmax函数对在指定维度上取最大值的下标,这里即在第1个维度上取最大值的下标
   # 也就是模型预测的是哪个数字的概率最大,其下标就是对应的这个数字
   # tf.equal(A,B)是对比两个矩阵同位置上元素,相等返回True,否则False,返回的也是和A一样维度的矩阵
   correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
   # tf.cast将bool型的True和False转换为1和0,这样就可以计算准确率了
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   # .eval()相当于Sess.run()一个Tensor对象,得到一个numpy数组结果
   print("Accuracy={}".format(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})))
```
# Tensorflow搭建第一个CNN神经网络(mnist)手写体数字分类
这里我们使用上面介绍的tf.data方式加载数据，可在每轮训练时就预加载下一轮的训练数据，训练起来速度更快。
如果我们想把训练集和测试集分开，那么需要对训练集和测试集分别创建dataset对象。

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# mnist样本为一维的28*28=784向量,标签为one_hot编码
n_input = 784
class_number = 10
drop = 0.75

# 使用tf.data方式加载训练集和测试集
train_dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels)).repeat().batch(
   batch_size)
train_dataset = train_dataset.prefetch(buffer_size=batch_size)
train_iterator = train_dataset.make_initializable_iterator()
train_batch_x, train_batch_y = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_tensor_slices((mnist.test.images, mnist.test.labels)).repeat().batch(
   batch_size)
test_iterator = test_dataset.make_initializable_iterator()
test_batch_x, test_batch_y = test_iterator.get_next()


# 定义模型
def conv_net(x, n_classes, dropout, reuse, is_training):
   # is_training参数表示是在训练/测试,reuse参数为True获取变量,False为创建变量
   with tf.variable_scope("ConvNet", reuse=reuse):
      # mnist图片数据为1维784向量,需要reshape成[-1, 28, 28, 1]
      # 即tensorflow需要的维度格式:[Batch Size, Height, Width, Channel]
      x = tf.reshape(x, shape=[-1, 28, 28, 1])
      # 32层卷积,卷积核5x5,卷积核参数自动初始化
      conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
      # 最大池化,2x2,步长2
      pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
      # 64层卷积,卷积核3x3
      conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
      # 最大池化,2x2,步长2
      pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
      # 将得到的特征图拉平为1维数据
      fc1 = tf.contrib.layers.flatten(pool2)
      # 全连接层,只需要定义全连接层权重数,自动匹配上一级拉平的1维特征图尺寸做矩阵乘法
      fc1 = tf.layers.dense(fc1, 1024)
      # 训练时激活dropout层
      fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
      # 输出层
      out = tf.layers.dense(fc1, n_classes)
      # 只在测试时计算softmax函数输出值,因为损失函数附带计算过了
      out = tf.nn.softmax(out) if not is_training else out

   return out


# 建立训练和测试网络
# reuse=False时,作用域就是为创建新变量所设置的;reuse=True时,作用域是为重用变量所设置
logits_train = conv_net(train_batch_x, class_number, drop, reuse=False, is_training=True)
logits_test = conv_net(test_batch_x, class_number, drop, reuse=True, is_training=False)
# 定义loss函数为交叉熵函数
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=train_batch_y))
# 定义优化算法为Adam算法
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
# 计算准确率
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(test_batch_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   sess.run(train_iterator.initializer)
   sess.run(test_iterator.initializer)
   # 训练模型
   for step in range(num_steps):
      sess.run(train_op)
      # 每训练100个batch测试一下模型
      if step % display_step == 0 or step == 0:
         loss, acc = sess.run([loss_op, accuracy])
         print("step={},loss={},test accuracy={}".format(step, loss, acc))
```
# Tensorflow使用高级API tf.Estimators搭建CNN神经网络(mnist手写体数字分类)
下面的代码使用了tensorflow高级API tf.Estimators，训练时速度更快。具体代码实现：
```python
import os
import numpy as np
import tensorflow as tf

# 传统方式下载mnist数据集
# from tensorflow.examples.tutorials.mnist import input_data

# 下面的设置可以使这个警告不出现:Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 指定使用的GPU的编号，从0开始
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 传统方式加载mnist数据集
# mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

# 加载mnist数据集
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

learning_rate = 0.001
num_steps = 2000
num_epochs = 10
batch_size = 128

num_input = 784
num_classes = 10
drop = 0.25


# 定义网络模型,这里采用高级API Estimators的形式定义模型
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
   # is_training参数表示是在训练/测试,reuse参数为True获取变量,False为创建变量
   with tf.variable_scope("ConvNet", reuse=reuse):
      x = x_dict["images"]
      # mnist图片数据为1维784向量,需要reshape成[-1, 28, 28, 1]
      # 即tensorflow需要的维度格式:[Batch Size, Height, Width, Channel]
      x = tf.reshape(x, shape=[-1, 28, 28, 1])
      # 32层卷积,卷积核5x5,卷积核参数自动初始化
      conv_1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
      # 最大池化,2x2,步长2
      pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
      # 64层卷积,卷积核3x3
      conv_2 = tf.layers.conv2d(pool_1, 64, 3, activation=tf.nn.relu)
      # 最大池化,2x2,步长2
      pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
      # 将得到的特征图拉平为1维数据
      fc1 = tf.contrib.layers.flatten(pool_2)
      # 全连接层,只需要定义全连接层权重数,自动匹配上一级拉平的1维特征图尺寸做矩阵乘法
      fc1 = tf.layers.dense(fc1, 1024)
      # 训练时激活dropout层
      fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
      # 输出层
      out = tf.layers.dense(fc1, n_classes)

   return out


# 自定义的model_fn
def model_fn(features, labels, mode):
   # 根据训练和测试的不同需要建立两个网络,训练时要新建变量,同时dropout层要激活
   logits_train = conv_net(features, num_classes, drop, reuse=False, is_training=True)
   logits_test = conv_net(features, num_classes, drop, reuse=True, is_training=False)
   # 定义损失函数为交叉熵,优化算法为Adam,原始labels转为0、1二值
   loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
   # tf.train.get_global_step()获取global_step,这个变量是Tensorflow随着训练过程自动更新的
   train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op,
                                                                           global_step=tf.train.get_global_step())
   # 预测结果
   pred_classes = tf.argmax(logits_test, axis=1)
   # 计算准确率
   acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
   if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
   estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op,
                                            eval_metric_ops={'accuracy': acc_op})

   return estim_specs


# 建立estimator
model = tf.estimator.Estimator(model_fn)
# 定义训练用样本的数据
input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": mnist.train.images}, y=mnist.train.labels,
                                              batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
# 训练模型
model.train(input_fn, steps=num_steps)
# 定义测试用样本的数据
input_fn = tf.estimator.inputs.numpy_input_fn(x={"images": mnist.test.images}, y=mnist.test.labels,
                                              batch_size=batch_size, shuffle=False)
# 评估模型
el = model.evaluate(input_fn)
print("Testing Accuracy={}".format(el["accuracy"]))
```
# Tensorflow搭建第一个LSTM神经网络(mnist手写体数字分类)
代码实现如下：
```python
import tensorflow as tf
import os
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate = 0.001
training_steps = 10000
batch_size = 128
test_size = 128
display_step = 200

# 每个时刻输入一行上28个像素的数据
num_input = 28
# 分为28个时刻
time_steps = 28
# 神经元数量
num_hidden = 128
num_classes = 10

X, Y = tf.placeholder("float", [None, time_steps, num_input]), tf.placeholder("float", [None, num_classes])

weights = {"out": tf.Variable(tf.random_normal([num_hidden, num_classes]))}
biases = {"out": tf.Variable(tf.random_normal([num_classes]))}


# 定义模型
def rnn_net(x, w, b):
   # tf.stack()是一个矩阵拼接函数,tf.unstack()是一个矩阵分解函数
   # 输入数据的shape:[batch_size, time_steps, n_input]
   # tf.unstack()将给定的R维张量拆分成R-1维张量,即将x按axis=1分解成time_steps个张量,最后返回time_steps个张量组成的列表
   x = tf.unstack(x, time_steps, axis=1)
   # 定义LSTM层,该层有128个神经元,forget_bias=1.0即初始第一个时刻遗忘门的偏置,这是为了减少在开始训练时遗忘的规模
   # state_is_tuple默认为True,就是表示返回的每个时刻输出ht和每个时刻的记忆Ct用一个元组表示
   lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
   # rnn.static_rnn()的输出对应于每一个时刻,如果只关心最后一个时刻的输出,取outputs[-1]即可
   # outputs即每个时刻的输出ht,states即每个时刻的记忆Ct
   outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
   # 去最后一个时刻的ht输出,进行一次线性变换
   return tf.matmul(outputs[-1], w["out"]) + b["out"]


# 创建rnn网络
logits = rnn_net(X, weights, biases)
# 最终预测结果
prediction = tf.nn.softmax(logits)
# 损失函数为交叉熵,优化算法使用Adam算法
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)
# 计算预测准确率
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # 训练模型
   for step in range(training_steps):
      train_batch_x, train_batch_y = mnist.train.next_batch(batch_size)
      # 将输入数据reshape成[batch_size, time_steps, n_input]
      train_batch_x = train_batch_x.reshape((batch_size, time_steps, num_input))
      sess.run(train_op, feed_dict={X: train_batch_x, Y: train_batch_y})
      # 每迭代10轮打印loss值和预测准确率
      if step % display_step == 0 or step == 0:
         test_batch_x, test_batch_y = mnist.test.next_batch(batch_size)
         test_batch_x = test_batch_x.reshape((batch_size, time_steps, num_input))
         loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_batch_x, Y: test_batch_y})
         print("step={},loss={:.4f},training accuracy={:.3f}".format(step, loss, acc))

   # 测试模型
   test_batch_x, test_batch_y = mnist.test.next_batch(batch_size)
   test_batch_x = test_batch_x.reshape((batch_size, time_steps, num_input))
   print("Testing Accuracy={}".format(sess.run(accuracy, feed_dict={X: test_batch_x, Y: test_batch_y})))
```