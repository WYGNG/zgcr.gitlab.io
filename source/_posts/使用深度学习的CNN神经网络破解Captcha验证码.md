---
title: 使用深度学习的CNN神经网络破解Captcha验证码
date: 2019-03-04 23:35:43
tags:
- 深度学习算法实践
categories:
- 深度学习算法实践
---

# Captcha样本数据的生成与处理
我们熟知的mnist数据集中的每张图片是一个(784,)的一维向量。同时，向量中的每个值都/255进行了0-1二值化。其标签是one_hot编码，即一个(10,1)的一维向量，假如该图片是7，那么这个向量中只有下标为7的元素为1，其他都为0。
下面我们要进行Captcha样本数据的生成与处理，将样本数据处理成类似mnist数据集的格式。
我们先利用Captcha库的自带函数生成验证码图片，生成的图片是彩色的。但对于数字识别而言，彩色与灰度图片并无区别，因此我们可以将其先转为灰度图片，减少计算量。生成验证码时，从0-9十个数字中随机挑选数字。然后我们仿照mnist的图片和标签的数据形式来处理该图片的数据，生成可以直接输入CNN网络模型训练的图片和标签数据。
具体代码实现如下：
我们需要先安装Captcha库，命令如下：
```python
python -m pip install captcha
```
然后创建一个CaptchaGenerator.py文件。
```python
# CaptchaGenerator.py
from captcha.image import ImageCaptcha
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# 本例中未使用大小写字母生成随机验证码,但我们可以使用数字+大小写字母生成随机验证码,只需少许修改即可
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"]
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "Z"]

# 以0-9中的数字生成随机验证码,如果要用数字+大小写字母生成随机验证码,只需char_set = number + alphabet + ALPHABET
char_set = number


# 生成验证码字符列表的函数
def random_captcha_text(ch_len, ch_set):
   # 将验证码包含的字符存在ch_text
   ch_text = []
   # 进行captcha_size个循环，每次循环将1个随机抽到的字符放入ch_text
   [ch_text.append(random.choice(ch_set)) for _ in range(ch_len)]

   return ch_text


# 获取和生成的验证码对应的字符图片
def gen_captcha_text_and_image(wid, hei, ch_len, ch_set):
   # 生成指定大小的图片
   img = ImageCaptcha(width=wid, height=hei)
   # 生成一个随机的验证码序列
   ch_text = random_captcha_text(ch_len, ch_set)
   # 将字符串序列中每个字符连接起来
   ch_text = "".join(ch_text)
   # 根据验证码序列生成对应的字符图片
   ch_text_img = img.generate(ch_text)
   ch_img = np.array(Image.open(ch_text_img))
   # 将图片转换成一个数组，这个数组有3个维度
   # 因为图片是用RGB模式表示的，将其转换成数组即图片的分辨率160X60的矩阵，矩阵每个元素是一个像素点上的RGB三个通道的值
   # 返回字符串形式的验证码和对应的图片矩阵(RGB形式)
   return ch_text, ch_img


# 把彩色图像转为灰度图像
def convert_image_to_gray(img):
   # 这是彩色图像转换为灰度图像的公式
   r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
   # 求得灰度值后/255进行0-1二值化
   gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
   # 也可以求r、g、b三个值的平均值作为灰度值
   img[:, :, 0], img[:, :, 1], img[:, :, 2] = gray, gray, gray
   gray_img = img
   # 创建一个新矩阵，这个矩阵中的元素个数与img矩阵一样。
   # 但是每个元素只是原来RGB三个通道中的值的一个（转换成灰度图片后，RGB三个通道的数值相等）
   # 把矩阵变成一维数组，数组元素按矩阵行顺序排列
   gray_img_array = np.array(img[:, :, 0]).flatten()
   # 获取灰度图像矩阵(RGB形式,但三通道值相同)和拉平的灰度图片一维数组(同mnist数据集的图片数据形式)
   return gray_img, gray_img_array


# 生成图像对应的标签,one_hot编码形式,但不同字符之间全部连在一起
def generate_captcha_text_label(ch_text, ch_set):
   ch_text_label = np.zeros(len(ch_text) * len(ch_set))
   for i in range(len(ch_text)):
      char = ch_text[i]
      if char.isdigit():
         char = ord(char) - 48
      elif char.islower():
         char = ord(char) - 97 + 10
      elif char.isupper():
         char = ord(char) - 65 + 10 + 26
      # 请注意选择char_set时按number、alphabet、ALPHABET的顺序
      ch_text_label[i * len(ch_set) + char] = 1

   return ch_text_label


# 把生成的标签转换回字符序列
def text_label_turn_to_char_list(ch_text_label, ch_len, ch_set):
   ch_list = []
   for i in range(ch_len):
      for j in range(len(ch_set)):
         if ch_text_label[i * len(ch_set) + j] == 1.0:
            ch_list.append(ch_set[j])
   ch_list = "".join(ch_list)

   return ch_list


# 把预测得到的标签(经过tf.argmax函数处理后的数组)转换回字符序列
def pred_label_turn_to_char_list(pred_label, ch_len, ch_set):
   ch_list = []
   [ch_list.append(ch_set[pred_label[i]]) for i in range(ch_len)]
   ch_list = "".join(ch_list)

   return ch_list


# 测试
if __name__ == "__main__":
   width = 160
   height = 60
   char_size = 4
   text, image = gen_captcha_text_and_image(width, height, char_size, char_set)
   print("生成的验证码字符序列为：", text)
   print("生成的验证码图片数据的维度：", image.shape)
   plt.figure()
   plt.title(text)
   plt.imshow(image)
   plt.show()
   gray_image, gray_image_array = convert_image_to_gray(image)
   print("生成的灰度图片对应的一维数组的：", gray_image_array)
   print("生成的灰度图片对应的一维数组的维度大小：", gray_image_array.shape)
   label = generate_captcha_text_label(text, char_set)
   print("图片的标签数组为：\n", label)
   print("标签数组的维度大小为：", label.shape)
   char_list = text_label_turn_to_char_list(label, char_size, char_set)
   print("由标签数组生成对应的字符序列：", char_list)
```
# 使用CNN神经网络训练、测试验证码图片
代码实现如下：
```python
# CaptchaTrain.py
from CaptchaGenerator import char_set
from CaptchaGenerator import gen_captcha_text_and_image, convert_image_to_gray, generate_captcha_text_label, \
   text_label_turn_to_char_list, pred_label_turn_to_char_list
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 基本参数设置：验证码图像的大小、验证码字符串长度、训练批样本大小
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60
TEXT_LEN = 4
CHAR_SET = char_set
BATCH_SIZE = 64
VALIDATION_SIZE = 100
TEST_SIZE = 100
ITERATION = 5

if not os.path.exists("./tmp/"):
   os.mkdir("./tmp")
if not os.path.exists("./train_image/"):
   os.mkdir("./train_image/")
if not os.path.exists("./validation_image/"):
   os.mkdir("./validation_image/")
if not os.path.exists("./test_image/"):
   os.mkdir("./test_image/")


# 生成一个训练batch
def get_next_batch(batch_size, mode="train"):
   im_save_path = ""
   batch_x = np.zeros(shape=[batch_size, IMAGE_WIDTH * IMAGE_HEIGHT])
   batch_y = np.zeros(shape=[batch_size, TEXT_LEN * len(CHAR_SET)])
   for index in range(batch_size):
      tex, img = gen_captcha_text_and_image(IMAGE_WIDTH, IMAGE_HEIGHT, TEXT_LEN, CHAR_SET)
      im = Image.fromarray(img)
      if mode == "train":
         im_save_path = "./train_image/" + str(index) + "_train.jpg"
      elif mode == "validation":
         im_save_path = "./validation_image/" + str(index) + "_validation.jpg"
      elif mode == "test":
         im_save_path = "./test_image/" + str(index) + "_test.jpg"
      im.save(im_save_path)
      gray_image, gray_image_matrix = convert_image_to_gray(img)
      batch_x[index, :] = gray_image_matrix
      batch_y[index, :] = generate_captcha_text_label(tex, CHAR_SET)

   return batch_x, batch_y


# 测试
# batch_x, batch_y = get_next_batch(BATCH_SIZE)
# print(batch_x[0], batch_y[0])

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, TEXT_LEN * len(CHAR_SET)])
keep_prob = tf.placeholder(tf.float32)


# 定义生成w变量的函数
def weight_variable(shape):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial)


# 定义生成b变量的函数
def bias_variable(shape):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial)


# 定义卷积函数,x是输入的图像，W是此卷积层的权重矩阵
def conv2d(x, w):
   return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# 定义池化函数，x是输入的矩阵
def max_pool2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 定义神经网络函数
def cnn(x_input, hei, wid, text_len, len_char_set):
   x = tf.reshape(x_input, shape=[-1, hei, wid, 1])
   # x：一个shape为(BATCH_SIZE,160,60,1)的张量
   # conv_1p输出：一个shape为(BATCH_SIZE,80,30,32)的张量
   # conv_2p输出：一个shape为(BATCH_SIZE,40,15,64)的张量
   # conv_3p输出：一个shape为(BATCH_SIZE,20,8,64)的张量
   # conv_3r输出：一个shape为(BATCH_SIZE,20X8X64)的张量
   # fc_1输出：一个shape为(BATCH_SIZE,1024)的张量
   # out输出：一个shape为(BATCH_SIZE,40)的张量
   conv_width_input = wid
   conv_height_input = hei
   w_c1 = weight_variable([3, 3, 1, 32])
   b_c1 = bias_variable([32])
   w_c2 = weight_variable([3, 3, 32, 64])
   b_c2 = bias_variable([64])
   w_c3 = weight_variable([3, 3, 64, 64])
   b_c3 = bias_variable([64])
   full_connect_width = math.ceil(conv_width_input / 8)
   full_connect_height = math.ceil(conv_height_input / 8)
   w_d1 = weight_variable([full_connect_width * full_connect_height * 64, 1024])
   b_d1 = bias_variable([1024])
   w_out = weight_variable([1024, text_len * len_char_set])
   b_out = bias_variable([text_len * len_char_set])

   conv_1c = tf.nn.relu(conv2d(x, w_c1) + b_c1)
   conv_1p = max_pool2x2(conv_1c)
   conv_1d = tf.nn.dropout(conv_1p, keep_prob)
   conv_2c = tf.nn.relu(conv2d(conv_1d, w_c2) + b_c2)
   conv_2p = max_pool2x2(conv_2c)
   conv_2d = tf.nn.dropout(conv_2p, keep_prob)
   conv_3c = tf.nn.relu(conv2d(conv_2d, w_c3) + b_c3)
   conv_3p = max_pool2x2(conv_3c)
   conv_3d = tf.nn.dropout(conv_3p, keep_prob)
   conv_3r = tf.reshape(conv_3d, [-1, full_connect_width * full_connect_height * 64])
   fc_1 = tf.nn.relu(tf.matmul(conv_3r, w_d1) + b_d1)
   fc_1d = tf.nn.dropout(fc_1, keep_prob)
   out = tf.matmul(fc_1d, w_out) + b_out

   return out


# 将预测值规整成BATCH_SIZE个TEXT_LEN行len(CHAR_SET)列的矩阵，这样每行就对应字符序列中的一个字符的标签
output = cnn(X, IMAGE_WIDTH, IMAGE_HEIGHT, TEXT_LEN, len(CHAR_SET))
# 定义loss函数和优化算法
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
# 预测值
predict = tf.argmax(tf.reshape(output, [-1, TEXT_LEN, len(CHAR_SET)]), 2)
# 将真实标签规整成BATCH_SIZE个TEXT_LEN行len(CHAR_SET)列的矩阵，这样每行就对应字符序列中的一个字符的标签
real_value = tf.argmax(tf.reshape(Y, [-1, TEXT_LEN, len(CHAR_SET)]), 2)
# 计算预测准确率
correct_pre = tf.equal(predict, real_value)
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# 保存模型的对象saver
saver = tf.train.Saver()

with tf.Session() as sess_train:
   sess_train.run(tf.global_variables_initializer())
   if os.path.exists("./tmp/checkpoint"):
      # 判断模型是否存在，如果存在则从模型中恢复变量
      saver.restore(sess_train, tf.train.latest_checkpoint("./tmp/"))
   step = 0
   while True:
      batch_x_train, batch_y_train = get_next_batch(BATCH_SIZE)
      _, batch_loss = sess_train.run([train_op, loss],
                                     feed_dict={X: batch_x_train, Y: batch_y_train, keep_prob: 0.75})
      if step % 5 == 0:
         # 每训练5次打印一次loss值
         print("iteration：%d , batch_loss：%s" % (step, batch_loss))
      if step % 100 == 0:
         # 每训练100次保存一次模型
         saver.save(sess_train, "./tmp/train_model", global_step=step)
         # 每训练100次计算并打印一次准确率
         batch_x_validation, batch_y_validation = get_next_batch(VALIDATION_SIZE, mode="validation")
         acc = sess_train.run(accuracy, feed_dict={X: batch_x_validation, Y: batch_y_validation, keep_prob: 1})
         print("iteration：%d , acc：%s" % (step, acc))
         # 如果准确率大于设定值,保存模型,完成训练
         if acc > 0.9:
            saver.save(sess_train, "./tmp/train_model", global_step=step)
            break
      step = step + 1
# 建立用于测试模型的Session会话

with tf.Session() as sess_test:
   sess_test.run(tf.global_variables_initializer())
   saver.restore(sess_test, tf.train.latest_checkpoint("./tmp/"))
   for i in range(ITERATION):
      batch_x_test, batch_y_test = get_next_batch(TEST_SIZE, mode="test")
      pred, real, acc = sess_test.run([predict, real_value, accuracy],
                                      feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1})
      # 注意pred和real都是tensorflow的张量tensor，虽然它们和多维数组形式非常类似，但不是数组，不是可迭代对象
      # tensorflow的张量tensor不能直接用在for循环里迭代，另外数组中的元素如果是float型不能直接用作数组的下标
      pred_int = tf.cast(pred, tf.int32)
      real_int = tf.cast(real, tf.int32)
      pred_array = pred_int.eval(session=sess_test)
      real_array = real_int.eval(session=sess_test)
      # pred, real的张量形式是2维矩阵，TEXT_SIZE行X4列矩阵，之所以是4列是因为前面用tf.argmax逐行提取了每行最大值的下标，一共四行，所以是4个下标
      # 上面两步先将张量的值全部转为int型，然后将张量转换成多维数组
      # 把张量转换成多维数组后，我们就可以进行迭代，然后使用pred_label_turn_to_char_list函数将标签转换成对应的字符序列
      for j in range(TEST_SIZE):
         # 这里预测值的标签经过tf.argmax只提取出了四个字符的下标，真实值的标签也这么处理了
         # 因此由标签得到字符序列都使用函数pred_label_turn_to_char_list
         pred_char_list = pred_label_turn_to_char_list(pred_array[j], TEXT_LEN, CHAR_SET)
         real_char_list = pred_label_turn_to_char_list(real_array[j], TEXT_LEN, CHAR_SET)
         print("第{}轮ITERATION中第{}个验证码预测：预测验证码为：{} 真实验证码为：{}".format(i + 1, j + 1, pred_char_list, real_char_list))
      print("第{}轮ITERATION识别测试正确率：{}".format(i + 1, acc))
```
该模型训练时准确率超过0.9后会自动停止训练，随后开始运行sess_test会话，开始测试模型。我设置的TEST_SIZE为100，iteration为5，即进行5轮测试，每轮测试100个样本。
该模型在我上面代码中的超参数设置的情况下，初始时的acc一直保持在0.1上下，在训练时进行大约500次-700次循环时准确率时准确率开始向上提升，在大约2800次-3000次循环时达到0.9，之后就在0.9到-0.95之间震荡。