---
title: 使用LSTM神经网络+CTC loss识别freetype库生成的不定长验证码
date: 2019-03-05 13:03:07
tags:
- 深度学习算法实践	
categories:
- 深度学习算法实践
---

# freetype库使用介绍
freetype为字体字库提供了一套解决方案，支持文字字体渲染等操作，主要还是其为C语言编写，跨平台，为很多不支持矢量字体格式的嵌入式系统提供使用嵌入式字体的可能，且效率不低。我们使用的是从C语言移植过来的python可调用的freetype-py库。
**字形:**
字符映像叫做字形，单个字符能够有多个不同的映像，即多个字形。多个字符也可以有一个字形(可以理解为一个字形就是一种书写风格)。
**字符图:**
字体文件包含一个或多个表，叫做字符图，用来将某种字符码转换成字形索引。一种字符编码方式(如ASCII、Unicode、Big5)对应一张表。
**像素、点和设备解析度:**
通常计算机的输出设备是屏幕或打印机，在水平和垂直方向都有多种分辨率，当我们使用FreeType渲染文本时要注意这些情况。设备的分辨率通常使用dpi(每英寸点(dot)数)表示的两个数。如，一个打印机的分辨率为300x600dpi表示在水平方向，每英寸有300 个像素，在垂直方向有600个像素。
文本的大小通常用点(point)表示。点是一种简单的物理单位，在数字印刷中，一点等于1/72英寸。我们可以用点数大小来计算像素数，公式如下：
像素数 = 点数X分辨率/72
单个点数通常定义不同象素文本宽度和高度。
**轮廓线:**
字形轮廓的源格式是一组封闭的路径，称为轮廓线。每个轮廓线划定字形的外部或内部区域，它们可以是线段或者Bezier曲线。
**EM正方形:**
字体在创建字形轮廓时，字体创建者所使用的假象的正方形。他可以将此想象成一个画字符的平面。它是用来将轮廓线缩放到指定文本尺寸的参考，如在300x300dpi中的12pt大小对应12X300/72=50象素。正方形的大小，即它边长的网格单元是很重要的。如从网格单元缩放到象素可以使用下面的公式：
象素数 ＝ 点数 × 分辨率/72
象素坐标＝ 网格坐标X象素数/EM大小
EM尺寸越大，可以达到更大的分辨率，例如一个极端的例子，一个4单元的EM，只有25个点位置，显然不够，通常TrueType字体之用2048单元的EM；Type1 PostScript字体有一个固定1000网格单元的EM，但是点坐标可以用浮点值表示。
字形可以自由超出EM正方形。网格单元通常交错字体单元或EM单元。上边的象素数并不是指实际字符的大小，而是EM正方形显示的大小，所以不同字体，虽然同样大小，但是它们的高度可能不同。
**位图渲染:**
指从字形轮廓转换成一个位图的过程。
**基线:**
基线是一个假象的线，用来在渲染文本时知道字形，它可以是水平或垂直的。
**笔位置(原点):**
为了渲染文本，在基线上有一个虚拟的点，叫做笔位置或原点，它用来定位字形。
**布局:**
布局有水平布局和垂直布局，每种布局使用不同的规约来放置字形。仅有少数字体格式支持垂直布局。
Glyph metrics(字形指标)指创建一个文本布局组织每个字形时描述字形如何定位的确切距离。
对水平布局，字形简单地搁在基线上，通过增加笔位置来渲染文本，既可以向右也可以向左增加。两个相邻笔位置之间的距离叫做步进宽度(advanceX)。注意这个值总是正数 。笔位置总是放置在基线上。
对垂直布局，字形在基线上居中放置。
**重要字体布局参数:**
* 边界框(bounding box,bbox)：这是一个假想的框子，即一个紧密地包围字符的轮廓。通过四个值来表示，叫做xMin 、yMin 、xMax 、yMax ，对任何轮廓都可以计算；
* 上行高度(ascent)：从基线到放置轮廓点最高(上)的距离；
* 下行高度(descent)：从基线到放置轮廓点最低(下)的距离；
* 左跨距(bearingX)：从当前笔位置（原点）到包围字符的轮廓的水平距离，用于水平布局；
* 上跨距(bearingY)：从当前笔位置（原点）到包围字符的轮廓的垂直距离，用于垂直布局；
* 步进宽度(advanceX)：相邻两个原点的水平距离(字间距)， 用于水平布局，即用来确定下一个字符的笔位置(原点)。
* 步进宽度(advanceY)：相邻两个原点的垂直距离(字间距)， 用于垂直布局，即用来确定下一个字符的笔位置(原点)。
* 字形宽度(width)：包围字符的轮廓的水平长度，与是水平布局还是垂直布局无关；
* 字形高度(height)：包围字符的轮廓的垂直长度，与是水平布局还是垂直布局无关。

**注意:**
每个字形最初的轮廓点放置在一个不可分割单元的网格中，点通常在字体文件中以16 位整型网格坐标存储，网格的原点在(0,0) ，它的范围是-16384 到-16383。
AdvanceX值通常四舍五入为整数像素坐标(例如是64的倍数)，字体驱动器用它装载字形图像。
**使用FreeType进行字符转换的基本流程:**
创建一个face对象，加载字体字库文件.ttf；将字符串转换成一系列字形索引；设定笔位置(原点)到字符串第一个字形原点位置；将字形渲染到目标设备；根据字形的步进象素增加笔位置；对剩余的字形重复第三步向下的步骤。
# 使用freetype库生成不定长验证码
首先安装第三方库freetype和opencv。使用下面的命令：
```python
python -m pip install freetype-py
python -m pip install opencv-python==3.4.1.15
```
**使用freetype库生成不定长验证码的步骤：**
* 创建一个face对象，加载特定的字体字库文件.ttf(相当于规定了字体)；
* 输入一个字符串、字符串在图片起始的位置(以像素点为单位)、字符的字体大小(以像素点为单位)、字符的颜色；
* 对字符串的每一个字符创建对应的字形，规定该字形在图片上的位置(这时候以1/64像素点为单位)；
* 将每个字形转化为位图画在背景图片上；
* 对画好的彩色图片加噪声，可以选择加高斯噪声或椒盐噪声；
* 加完噪声后，可以选择用方框滤波对图片进行降噪；
* 对彩色图片灰度化，并生成图片的一维数组形式(一个像素点用一个灰度值代表)，这个一维数组就是可以送入深度学习模型训练的数据形式。

**PutChineseText类:**
* init创建一个face对象，装载一种字体文件.ttf；
* draw_text方法输入一个空白图片(三维数组形式)、要画在图片上的文本内容、文本在图片上的起始位置(像素点为单位)、字体大小(像素点为单位)、字体颜色(RGB颜色)，调用draw_string方法返回一个画好文本的图片(三维数组形式)；
* draw_string方法用来将输入的文本一个字符一个字符地转换成字形，再调用draw_ft_bitmap方法将字形一个一个地画在图片上在空白图片(三维数组形式)上，该方法输入文本在图片上的起始位置(1/64像素点为单位)、文本内容和颜色；
* draw_ft_bitmap方法输入空白图片(三维数组形式)、要画成位图的字形、字形的原点位置(1/64像素点为单位)、字体颜色，画出一个带有字形转换成的位图的图片。

**GenerateCharListImage()类:**
* init初始化候选字符集为0-9十个数字，候选字符集的长度，生成的不定长验证码的最大长度，一个PutChineseText类对象(定义使用OCR-B.ttf字体)；
* random_text方法生成一个随机长度的字符序列和其对应的标签向量；
* generate_color_image方法生成一个带不定长的验证码的彩色图片；
* image_add_salt_noise方法和image_add_gaussian_noise方法可以给一张图片(三维数组形式)添加椒盐噪声和高斯噪声；
* image_reduce_noise方法使用方框滤波给一张图(三维数组形式)降噪；
* color_image_to_gray_image方法将一张三维数组形式的图片转变成灰度图片，并且每个像素点只保留一个值(灰度图片的RGB三个通道的值相同)，并将图片形式变成一维数组；
* one_char_to_one_element_vector方法将单个字符生成对应的标签向量；
* text_vector_to_text方法将由一个标签向量生成对应字符串。

**具体代码实现如下:**
```python
import numpy as np
import freetype
import copy
import random
import cv2
import os


# 使用FreeType库生成验证码图片时，我们输入的文本的属性pos位置和text_size大小是以像素点为单位的，但是将文本转化成字形时
# 需要把这些数据转换成1/64像素点单位计数的值，然后在画位图时，还要把相关的数据重新转化成像素点单位计数的值
# 这就是本class类中几个方法主要做的工作

class PutChineseText(object):
   def __init__(self, ttf):
      # 创建一个face对象，装载一种字体文件.ttf
      self._face = freetype.Face(ttf)

   # 在一个图片（用三维数组表示）上绘制文本字符
   def draw_text(self, image, pos, text, text_size, text_color):
      """
      draw chinese(or not) text with ttf
      :param image:     一个图片平面，用三维数组表示
      :param pos:       在图片上开始绘制文本字符的位置，以像素点为单位
      :param text:      文本的内容
      :param text_size: 文本字符的字体大小，以像素点为单位
      :param text_color:文本字符的字体颜色
      :return:          返回一个绘制了文本的图片
      """
      # self._face.set_char_size以物理点的单位长度指定了字符尺寸，这里只设置了宽度大小，则高度大小默认和宽度大小相等
      # 我们将text_size乘以64倍得到字体的以point单位计数的大小，也就是说，我们认为输入的text_size是以像素点为单位来计量字体大小
      self._face.set_char_size(text_size * 64)
      # metrics用来存储字形布局的一些参数，如ascender，descender等
      metrics = self._face.size
      # 从基线到放置轮廓点最高(上)的距离，除以64是重新化成像素点单位的计数
      # metrics中的度量26.6象素格式表示，即数值是64倍的像素数
      # 这里我们取的ascender重新化成像素点单位的计数
      ascender = metrics.ascender / 64.0
      # 令ypos为从基线到放置轮廓点最高(上)的距离
      ypos = int(ascender)
      # 如果文本不是unicode格式，则用utf-8编码来解码，返回解码后的字符串
      if isinstance(text, str) is False:
         text = text.decode("utf-8")
      # 调用draw_string方法来在图片上绘制文本，也就是说draw_text方法其实主要是在定位字形位置和设定字形的大小，然后调用draw_string方法来在图片上绘制文本
      img = self.draw_string(image, pos[0], pos[1] + ypos, text, text_color)

      return img

   # 绘制字符串方法
   def draw_string(self, img, x_pos, y_pos, text, color):
      """
      draw string
      :param x_pos: 文本在图片上开始的x轴位置，以1/64像素点为单位
      :param y_pos: 文本在图片上开始的y轴位置，以1/64像素点为单位
      :param text:  unicode形式编码的文本内容
      :param color: 文本的颜色
      :return:      返回一个绘制了文本字形的图片（三维数组形式）
      """
      prev_char = 0
      # pen是笔位置或叫原点，用来定位字形
      pen = freetype.Vector()
      # 设定pen的x轴位置和y轴位置，注意pen.x和pen.y都是以1/64像素点单位计数的，而x_pos和y_pos都是以像素点为单位计数的
      # 因此x_pos和y_pos都左移6位即乘以64倍化成1/64像素点单位计数
      pen.x = x_pos << 6
      pen.y = y_pos << 6
      hscale = 1.0
      # 设置一个仿射矩阵
      matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000), int(0.0 * 0x10000), int(1.1 * 0x10000))
      cur_pen = freetype.Vector()
      pen_translate = freetype.Vector()
      # 将输入的img图片三维数组copy过来
      image = copy.deepcopy(img)
      # 一个字符一个字符地将其字形画成位图
      for cur_char in text:
         # 当字形图像被装载时，对该字形图像进行仿射变换,这只适用于可伸缩（矢量）字体格式。set_transform()函数就是做这个工作
         self._face.set_transform(matrix, pen_translate)
         # 装载文本中的每一个字符
         self._face.load_char(cur_char)
         # 获取两个字形的字距调整信息，注意获得的值是1/64像素点单位计数的。因此可以用来直接更新pen.x的值
         kerning = self._face.get_kerning(prev_char, cur_char)
         # 更新pen.x的位置
         pen.x += kerning.x
         # 创建一个字形槽，用来容纳一个字形
         slot = self._face.glyph
         # 字形图像转换成位图
         bitmap = slot.bitmap
         # cur_pen记录当前光标的笔位置
         cur_pen.x = pen.x
         # pen.x的位置上面已经更新过
         # bitmap_top是字形原点(0,0)到字形位图最高像素之间的垂直距离，由于是像素点计数的，我们用其来更新cur_pen.y时要转换成1/64像素点单位计数
         cur_pen.y = pen.y - slot.bitmap_top * 64
         # 调用draw_ft_bitmap方法来画出字形对应的位图，注意这里是循环，也就是一个字符一个字符地画
         self.draw_ft_bitmap(image, bitmap, cur_pen, color)
         # 每画完一个字符，将pen.x更新成下一个字符的笔位置（原点位置）,advanceX即相邻两个原点的水平距离(字间距)
         pen.x += slot.advance.x
         # prev_char更新成当前新画好的字符的字形的位置
         prev_char = cur_char
      # 返回包含所有字形的位图的图片（三维数组）

      return image

   # 将字形转化成位图
   def draw_ft_bitmap(self, img, bitmap, pen, color):
      """
      draw each char
      :param bitmap: 要转换成位图的字形
      :param pen:    开始画字形的位置，以1/64像素点为单位
      :param color:  RGB三个通道值表示，每个值0-255范围
      :return:       返回一个三维数组形式的图片
      """

      # 获得笔位置的x轴坐标和y轴坐标，这里右移6位是重新化为像素点单位计数的值
      x_pos = pen.x >> 6
      y_pos = pen.y >> 6
      # rows即位图中的水平线数
      # width即位图的水平象素数
      cols = bitmap.width
      rows = bitmap.rows
      # buffer数一个指向位图象素缓冲的指针，里面存储了我们的字形在某个位置上的信息，即字形轮廓中的所有的点上哪些应该画成黑色，或者是白色
      glyph_pixels = bitmap.buffer
      # 循环画位图
      for row in range(rows):
         for col in range(cols):
            # 如果当前位置属于字形的一部分而不是空白
            if glyph_pixels[row * cols + col] != 0:
               # 写入每个像素点的三通道的值
               img[y_pos + row][x_pos + col][0] = color[0]
               img[y_pos + row][x_pos + col][1] = color[1]
               img[y_pos + row][x_pos + col][2] = color[2]


# 快速设置带字符串的图片的属性
class GenerateCharListImage(object):
   # 初始化图片属性
   def __init__(self):
      # 候选字符集为数字0-9
      self.number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
      # 令char_set为候选字符集
      self.char_set = self.number
      # 计算候选字符集的字符个数
      self.len = len(self.char_set)
      # 生成的不定长验证码最大长度
      self.max_size = 4
      self.ft = PutChineseText("fonts/OCR-B.ttf")

   # 生成随机长度0-max_size之间的字符串，并返回字符串及对应的标签向量
   def random_text(self):
      # 空字符串
      text = ""
      # 空标签向量
      text_vector = np.zeros((self.max_size * self.len))
      # 设置字符串的长度是随机的
      size = random.randint(1, self.max_size)
      # size = self.max_size
      # 逐个生成字符串和对应的标签向量
      for index in range(size):
         c = random.choice(self.char_set)
         one_element_vector = self.one_char_to_one_element_vector(c)
         # 更新字符串和标签向量
         text = text + c
         text_vector[index * self.len:(index + 1) * self.len] = np.copy(one_element_vector)
      # 返回字符串及对应的标签向量

      return text, text_vector

   # 根据生成的字符串，生成验证码图片，返回图片数据和其标签,默认给图片添加高斯噪声
   def generate_color_image(self, img_shape, noise):
      text, text_vector = self.random_text()
      # 创建一个图片背景，图片背景为黑色
      img_background = np.zeros([img_shape[0], img_shape[1], 3])
      # 设置图片背景为白色
      img_background[:, :, 0], img_background[:, :, 1], img_background[:, :, 2] = 255, 255, 255
      # (0, 0, 0)黑色，(255, 255, 255)白色，(255, 0, 0)深蓝色，(0, 255, 0)绿色，(0, 0, 255)红色
      # 设置字体颜色为黑色
      text_color = (0, 0, 0)
      # 设置文本在图片上起始位置和文本大小，单位都是像素点
      pos = (20, 10)
      text_size = 20
      # 画出验证码图片，返回的image是一个三维数组
      image = self.ft.draw_text(img_background, pos, text, text_size, text_color)
      # 如果想添加噪声
      if noise == "gaussian":
         # 添加20%的高斯噪声
         image = self.image_add_gaussian_noise(image, 0.2)
      elif noise == "salt":
         # 添加20%的椒盐噪声
         image = self.image_add_salt_noise(image, 0.1)
      elif noise == "None":
         pass
      # 返回三维数组形式的彩色图片

      return image, text, text_vector

   # 给一张生成的图片加入随机椒盐噪声
   def image_add_salt_noise(self, image, percent):
      rows, cols, dims = image.shape
      # 要添加椒盐噪声的像素点的数量，用全图像素点个数乘以一个百分比计算出来
      salt_noise_num = int(percent * image.shape[0] * image.shape[1])
      for i in range(salt_noise_num):
         # 获得随机的一个x值和y值，代表一个像素点
         x = np.random.randint(0, rows)
         y = np.random.randint(0, cols)
         # 所谓的椒盐噪声就是随机地将图像中的一定数量(这个数量就是椒盐的数量num)的像素值取极大或者极小
         # 即让维度0第x个，维度1第y个确定的一个像素点的数组(这个数组有三个元素)的三个值都为0,即噪点是黑色，因为我们的图片背景是白色
         image[x, y, :] = 0

      return image

   # 给一张生成的图片加入高斯噪声
   def image_add_gaussian_noise(self, image, percent):
      rows, cols, dims = image.shape
      # 要添加的高斯噪点的像素点的数量，用全图像素点个数乘以一个百分比计算出来
      gaussian_noise_num = int(percent * image.shape[0] * image.shape[1])
      # 逐个给像素点添加噪声
      for index in range(gaussian_noise_num):
         # 随机挑一个像素点
         x_temp, y_temp = np.random.randint(0, rows), np.random.randint(0, cols)
         # 随机3个值，加到这个像素点的3个通道值上，为了不超过255，后面再用clamp函数限定其范围不超过255
         value_temp = np.random.normal(0, 255, 3)
         for subscript in range(3):
            image[x_temp, y_temp, subscript] = image[x_temp, y_temp, subscript] - value_temp[subscript]
            if image[x_temp, y_temp, subscript] > 255:
               image[x_temp, y_temp, subscript] = 255
            elif image[x_temp, y_temp, subscript] < 0:
               image[x_temp, y_temp, subscript] = 0

      return image

   # 图片降噪函数
   def image_reduce_noise(self, image):
      # 使用方框滤波，normalize如果等于true就相当于均值滤波了，-1表示输出图像深度和输入图像一样，(2,2)是方框大小
      image = cv2.boxFilter(image, -1, (2, 2), normalize=False)
      return image

   # 将彩色图像转换成灰度图片的一维数组形式的数据形式
   def color_image_to_gray_image(self, image):
      # 将图片转成灰度数据，并进行标准化(0-1之间)
      r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
      gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
      # 生成灰度图片对应的一维数组数据，即输入模型的x数据形式
      gray_image_array = np.array(gray).flatten()
      # 返回度图片的一维数组的数据形式

      return gray_image_array

   # 单个字符转为向量
   def one_char_to_one_element_vector(self, c):
      one_element_vector = np.zeros([self.len, ])
      # 找每个字符是字符集中的第几个字符，是第几个就把标签向量中第几个元素值置1
      for index in range(self.len):
         if self.char_set[index] == c:
            one_element_vector[index] = 1
      return one_element_vector

   # 整个标签向量转为字符串
   def text_vector_to_text(self, text_vector):
      text = ""
      text_vector_len = len(text_vector)
      # 找标签向量中为1的元素值，找到后index即其下标，我们就知道那是候选字符集中的哪个字符
      for index in range(text_vector_len):
         if text_vector[index] == 1:
            text = text + self.char_set[index % self.len]
      # 返回字符串
      return text


if __name__ == "__main__":
   # 创建文件保存路径
   if not os.path.exists("./free_type_image/"):
      os.mkdir("./free_type_image/")
   # 图片尺寸
   image_shape = (40, 120)
   test_object = GenerateCharListImage()
   # 生成一个不加噪声的图片
   test_color_image_no_noise, test_text_no_noise, test_text_vector_no_noise = test_object.generate_color_image(
      image_shape, noise="None")
   test_gray_image_array_no_noise = test_object.color_image_to_gray_image(test_color_image_no_noise)
   print(test_gray_image_array_no_noise)
   print(test_text_no_noise, test_text_vector_no_noise)
   cv2.imwrite("./free_type_image/test_color_image_no_noise.jpg", test_color_image_no_noise)
   # 显示这张不加噪声的图片
   cv2.imshow("test_color_image_no_noise", test_color_image_no_noise)
   # 2000毫秒后刷新图像
   cv2.waitKey(2000)
   # 生成一个加了高斯噪声的图片
   test_color_image_gaussian_noise, test_text_gaussian_noise, test_text_vector_gaussian_noise = test_object.generate_color_image(
      image_shape, noise="gaussian")
   cv2.imwrite("./free_type_image/test_color_image_gaussian_noise.jpg", test_color_image_gaussian_noise)
   cv2.imshow("test_color_image_gaussian_noise", test_color_image_gaussian_noise)
   # 2000毫秒后刷新图像
   cv2.waitKey(2000)
   # 高斯噪声图片降噪后的图片
   test_color_image_reduce_gaussian_noise = test_object.image_reduce_noise(test_color_image_gaussian_noise)
   cv2.imwrite("./free_type_image/test_color_image_reduce_gaussian_noise.jpg", test_color_image_reduce_gaussian_noise)
   cv2.imshow("test_color_image_reduce_gaussian_noise", test_color_image_reduce_gaussian_noise)
   # 2000毫秒后刷新图像
   cv2.waitKey(2000)
   # 生成一个加了椒盐噪声的图片
   test_color_image_salt_noise, test_text_salt_noise, test_text_vector_salt_noise = test_object.generate_color_image(
      image_shape, noise="salt")
   cv2.imwrite("./free_type_image/test_color_image_salt_noise.jpg", test_color_image_salt_noise)
   cv2.imshow("test_color_image_salt_noise", test_color_image_salt_noise)
   # 2000毫秒后刷新图像
   cv2.waitKey(2000)
   # 椒盐噪声图片降噪后的图片
   test_color_image_reduce_salt_noise = test_object.image_reduce_noise(test_color_image_salt_noise)
   cv2.imwrite("./free_type_image/test_color_image_reduce_salt_noise.jpg", test_color_image_reduce_salt_noise)
   cv2.imshow("test_color_image_reduce_salt_noise", test_color_image_reduce_salt_noise)
   # 2000毫秒后刷新图像
   cv2.waitKey(2000)
   cv2.destroyAllWindows()
```
# 使用LSTM神经网络+CTC loss识别freetype库生成的不定长验证码
使用free_type_get_next_batch()函数用来取得一批训练样本，返回值为输入的x数据(图片数据)、压缩过的图片标签(注意这个标签不是独热编码，而是这批样本中每个样本代表的字符串)、seq_length(记录这批样本中每个样本中有多少个时间序列，即time_steps的数量)。
图片标签原本是一批样本的所有字符串(一个样本一个字符串)的集合，放在一个列表中，经过sparse_tuple_from函数转化为稀疏矩阵的记录形式，即有3个结构indices, values, shape。indices记录每一个字符在字符所在的样本中的位置(下标)，values记录了所有的字符串中的字符，shape记录了样本个数和样本字符串最大长度。
训练模型定义了一个cell层，该层有64个神经元，输入图片的尺寸为40X120，每张图片的一列的像素点数据看成一个time_steps中输入的数据，所以time_steps=120。输入数据的shape=[batch_size,num_steps,input_dim]。dynamic_rnn函数的time_major=False，故cell层输出数据的shape=[batch_size,max_time_step,cell.output_size]，cell.output_size即cell层神经元的数目。cell层后接一个全连接层，再经过一系列操作，得到的logits的shape=[max_time_step,batch_size,num_classes]。
定义ctc_loss为损失函数，输入真实标签(是一个经过sparse_tuple_from函数转化为稀疏矩阵的记录形式)，预测标签logits，以及seq_len(每个样本的time_steps的数量)。
通过 tf.nn.ctc_beam_search_decoder函数将预测标签解码成稀疏矩阵的记录形式(即压缩过的标签向量形式)。
每轮使用64个样本训练模型，每训练100次将预测标签和真实标签用decode_sparse_tensor()函数将其解码成字符串形式，然后比对，最后得到预测准确率。
**具体代码实现如下:**
```python
# LSTM+CTC_loss训练识别不定长数字字符图片
from freeTypeGenerateTextImage import GenerateCharListImage
import tensorflow as tf
import numpy as np
import time
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 超参数
# 要生成的图片的像素点大小
char_list_image_shape = (40, 120)
# 隐藏层神经元数量
num_hidden = 64
# 初始学习率和学习率衰减因子
lr_start = 1e-3
lr_decay_factor = 0.9
# 一批训练样本和测试样本的样本数量,训练迭代次数,每经过test_report_step_interval测试一次模型预测的准确率
train_batch_size = 64
test_batch_size = 64
iteration = 5000
test_display_step = 100

# 用来恢复标签用的候选字符集
char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
acc_reach_to_stop = 0.95
obj_number = GenerateCharListImage()
# 类别为10位数字+blank+ctc blank
num_classes = obj_number.len + 1 + 1


# 生成batch_size个样本,样本的shape变为[batch_size,image_shape[1],image_shape[0]]
# 输入的图片是把每一行的数据看成一个时间间隔t内输入的数据,然后有多少行就是有多少个时间间隔
# 使用freetype库生成一批样本
def free_type_get_next_batch(bt_size, img_shape):
   obj_batch = GenerateCharListImage()
   bt_x_inputs = np.zeros([bt_size, char_list_image_shape[1], char_list_image_shape[0]])
   bt_y_inputs = []
   for i in range(bt_size):
      # 生成不定长度的字符串及其对应的彩色图片
      color_image, text, text_vector = obj_batch.generate_color_image(img_shape, noise="gaussian")
      # 图片降噪,然后由彩色图片生成灰度图片的一维数组形式
      color_image = obj_batch.image_reduce_noise(color_image)
      gray_image_array = obj_batch.color_image_to_gray_image(color_image)
      # np.transpose函数将得到的图片矩阵转置成(image_shape[1],image_shape[0])形状的矩阵,且由行有序变成列有序
      # 然后将这个图片的数据写入bt_x_inputs中第0个维度上的第i个元素(每个元素就是一张图片的所有数据)
      bt_x_inputs[i, :] = np.transpose(gray_image_array.reshape((char_list_image_shape[0], char_list_image_shape[1])))
      # 把每个图片的标签添加到bt_y_inputs列表,注意这里直接添加了图片对应的字符串
      bt_y_inputs.append(list(text))
   # 将bt_y_inputs中的每个元素都转化成np数组
   targets = [np.asarray(i) for i in bt_y_inputs]
   # 将targets列表转化为稀疏矩阵
   sparse_matrix_targets = sparse_tuple_from(targets)
   # bt_size个1乘以char_list_image_shape[1],也就是batch_size个样本中每个样本（每个样本即图片）的长度上的像素点个数（或者说列数）
   # seq_length就是每个样本中有多少个时间序列
   seq_length = np.ones(bt_x_inputs.shape[0]) * char_list_image_shape[1]
   # 得到的bt_x_inputs的shape=[bt_size, char_list_image_shape[1], char_list_image_shape[0]]

   return bt_x_inputs, sparse_matrix_targets, seq_length


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
   """
   :param sequences: 一个元素是列表的列表
   :param dtype: 列表元素的数据类型
   :return: 返回一个元组(indices, values, shape)
   """
   indices = []
   values = []
   for index, seq in enumerate(sequences):
      # sequences存储了你的样本对应的字符串(由数字组成)的所有数字
      # 每次取list中的一个元素,即一个数字,代表的是一个样本(即一个字符串)中的一个数字值,注意这个单独的数字是也是一个列表
      # extend()函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
      # zip()函数将对象中对应的元素打包成一个个元组,然后返回由这些元组组成的一个对象。
      # zip(a,b)函数分别从a和b中取一个元素组成元组,再次将组成的元组组合成一个新的迭代器。a与b的维数相同时,正常组合对应位置的元素。
      # 每个seq是一个字符串,index即这是第几个字符串(第几个样本)
      indices.extend(zip([index] * len(seq), range(len(seq))))
      # [index]的值为[0]、[1]、[2]。。。,len(seq)为每个字符串的长度
      # 如[1]*4的结果是[1, 1, 1, 1]
      # * 操作符在实现上是复制了值的引用,而不是创建了新的对象。所以上述的list里面,是4个指向同一个对象的引用,所以4个值都是1
      values.extend(seq)

   indices = np.asarray(indices, dtype=np.int64)
   values = np.asarray(values, dtype=dtype)
   shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
   # indices:二维int64的矩阵,代表元素在batch样本矩阵中的位置
   # values:二维tensor,代表indice位置的数据值
   # dense_shape:一维,代表稀疏矩阵的大小
   # 假设sequences有2个,值分别为[1 3 4 9 2]、[ 8 5 7 2]。(即batch_size=2）
   # 则其indices=[[0 0][0 1][0 2][0 3][0 4][1 0][1 1][1 2][1 3]]
   # values=[1 3 4 9 2 8 5 7 2]
   # shape=[2 5]

   return indices, values, shape


# 解压缩压缩过的所有样本的字符串的列表的集合,return为不压缩的所有样本的字符串的列表的集合
def decode_sparse_tensor(sparse_tensor):
   decoded_indexes = list()
   current_i = 0
   current_seq = []
   # sparse_tensor[0]即sparse_tuple_from函数的返回值中的indices
   # 这里是一批样本的字符串的列表集合经过sparse_tuple_from函数处理后的返回值中的indices
   # offset即indices中元素的下标,即indices中的第几个元素(每个元素是一个单字符,代表这个单字符在这批样本中的位置)
   # i_and_index即sparse_tensor[0]也就是indices中的每个元素,i_and_index[0]即sparse_tensor[0]中每个元素属于第几号样本
   for offset, i_and_index in enumerate(sparse_tensor[0]):
      # i记录现在遍历到的sparse_tensor[0]元素属于第几号样本
      i = i_and_index[0]
      # 如果新遍历到的sparse_tensor[0]元素和前一个元素不属于同一个样本
      if i != current_i:
         # 每次属于同一个样本的sparse_tensor[0]元素遍历完以后,decoded_indexes添加这个样本的完整current_seq
         decoded_indexes.append(current_seq)
         # 更新i
         current_i = i
         # 对这样新编号的样本建立一个新的current_seq
         current_seq = list()
      # current_seq记录我们现在遍历到的sparse_tensor[0]元素在这批样本中的位置(下标)
      current_seq.append(offset)
   # for循环遍历完以后,添加最后一个样本的current_seq到decoded_indexes,这样decoded_indexes就记录了这批样本中所有样本的current_seq
   decoded_indexes.append(current_seq)
   result = []
   # 遍历decoded_indexes,依次解码每个样本的字符串内容
   # 实际上decoded_indexes就是记录了一批样本中每个样本中的所有字符在这批样本中的位置(下标)
   for index in decoded_indexes:
      result.append(decode_a_seq(index, sparse_tensor))
   # result记录了这批样本中每个样本的字符串内容,result的每个元素就是一个样本的字符串的内容
   # 这个元素是一个列表,列表每个元素是一个单字符

   return result


# 将压缩过的所有样本的字符串的列表的集合spars_tensor中取出第indexes号样本中的所有字符在这个样本中的位置(下标),解压缩成字符串
def decode_a_seq(indexes, spars_tensor):
   decoded = []
   # indexes是decoded_indexes中第indexes号样本中的所有字符在这批样本中的位置(下标)
   # for循环取出的m就是这个样本中每个字符在这批样本中的位置(下标)
   for m in indexes:
      # spars_tensor[1][m]即spars_tensor中的values列表的第m个值
      # ch即取出了m对应的spars_tensor中的values列表的第m个值,是一个字符
      ch = char_set[spars_tensor[1][m]]
      # 把这个字符加到decoded列表中
      decoded.append(ch)
   # decoded列表即存储一个样本中的所有字符

   return decoded


# 定义训练模型
def get_train_model():
   x_inputs = tf.placeholder(tf.float32, [None, None, char_list_image_shape[0]])
   # inputs的维度是[batch_size,num_steps,input_dim]
   # 定义ctc_loss需要的标签向量(稀疏矩阵形式)
   targets = tf.sparse_placeholder(tf.int32)
   # 每个样本中有多少个时间序列
   seq_length = tf.placeholder(tf.int32, [None])
   # 定义LSTM网络的cell层,这里定义有num_hidden个单元
   cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
   # state_is_tuple:如果为True,接受和返回的states是n-tuples,其中n=len(cells)。
   # 如果cell选择了state_is_tuple=True,那final_state是个tuple,分别代表Ct和ht,其中ht与outputs中的对应的最后一个时刻的输出ht相等；
   # 如果time_major == False(default),输出张量形如[batch_size, max_time, cell.output_size]。
   # 如果time_major == True, 输出张量形如：[max_time, batch_size, cell.output_size]。
   # cell.output_size其实就是我们的num_hidden,即cell层的神经元的个数。
   outputs, _ = tf.nn.dynamic_rnn(cell, x_inputs, seq_length, time_major=False, dtype=tf.float32)
   # ->[batch_size,max_time_step,num_features]->lstm
   # ->[batch_size,max_time_step,cell.output_size]->reshape
   # ->[batch_size*max_time_step,num_hidden]->affine projection AW+b
   # ->[batch_size*max_time_step,num_classes]->reshape
   # ->[batch_size,max_time_step,num_classes]->transpose
   # ->[max_time_step,batch_size,num_classes]
   # 上面最后的shape就是标签向量的shape,此时标签向量还未压缩
   shape = tf.shape(x_inputs)
   # x_inputs的shape=[batch_size,image_shape[1],image_shape[0]]
   # 所以输入的数据是按列来排的,一列的像素为一个时间序列里输入的数据,一共120个时间序列
   batch_s, max_time_steps = shape[0], shape[1]
   # 输出的outputs为num_hidden个隐藏层单元的所有时刻的输出
   # reshape后的shape=[batch_size*max_time_step,num_hidden]
   outputs = tf.reshape(outputs, [-1, num_hidden])
   # 相当于一个全连接层,做一次线性变换
   w = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="w")
   b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
   logits = tf.matmul(outputs, w) + b
   # 变换成和标签向量一致的shape
   logits = tf.reshape(logits, [batch_s, -1, num_classes])
   # logits的维度交换,第1个维度和第0个维度互相交换
   logits = tf.transpose(logits, (1, 0, 2))
   # 注意返回的logits预测标签此时还未压缩,而targets真实标签是被压缩过的

   return logits, x_inputs, targets, seq_length, w, b


# test_targets即用sparse_tuple_from压缩过的所有样本的字符串的一个列表的集合,decoded_list也是一样
def report_accuracy(decoded_list, test_targets):
   # 将压缩的真实标签和预测标签解压缩,解压缩后都是一个列表,列表中存储了这批样本中的所有字符串。
   # 列表中的每个元素都是一个列表,这个列表中包含一个样本中的所有字符。
   original_list = decode_sparse_tensor(test_targets)
   detected_list = decode_sparse_tensor(decoded_list)
   # 本批样本中预测正确的次数
   correct_prediction = 0
   # 注意这里的标签不是指独热编码,而是这批样本的每个样本代表的字符串的集合
   # 如果解压缩后的真实标签和预测标签的样本个数不一样
   if len(original_list) != len(detected_list):
      print("真实标签样本个数:{},预测标签样本个数:{},真实标签与预测标签样本个数不匹配".format(len(original_list), len(detected_list)))
      return -1
   print("真实标签(长度) <-------> 预测标签(长度)")
   # 注意这里的标签不是指独热编码,而是这批样本的每个样本代表的字符串的集合
   # 如果真实标签和预测标签的样本个数吻合,则分别比对每一个样本的预测结果
   # for循环从original_list中取出一个一个的字符串(注意字符串存在一个列表中,列表中每个元素是单个字符)
   for idx, true_number in enumerate(original_list):
      # detected_list[idx]即detected_list中第idx号字符串(注意字符串存在一个列表中,列表中每个元素是单个字符)
      detect_number = detected_list[idx]
      # signal即真实标签是否与预测标签相等的结果,相等则为true
      signal = (true_number == detect_number)
      # 打印true_number和detect_number直观对比
      print(signal, true_number, "(", len(true_number), ") <-------> ", detect_number, "(", len(detect_number), ")")
      # 如果相等,统计正确的预测次数加1
      if signal is True:
         correct_prediction += 1
   # 计算本批样本预测的准确率
   acc = correct_prediction * 1.0 / len(original_list)
   print("本批样本预测准确率:{}".format(acc))

   return acc


# 定义训练过程
def train():
   global_step = tf.Variable(0, trainable=False)
   # tf.train.exponential_decay函数实现指数衰减学习率
   learning_rate = tf.train.exponential_decay(lr_start, global_step, iteration, lr_decay_factor, staircase=True)
   logits, inputs, targets, seq_len, w, b = get_train_model()
   # 注意得到的logits此时是未压缩的标签向量
   # 设置loss函数是ctc_loss函数
   # CTC ：Connectionist Temporal Classifier 一般译为联结主义时间分类器 ,适合于输入特征和输出标签之间对齐关系不确定的时间序列问题
   # TC可以自动端到端地同时优化模型参数和对齐切分的边界。
   # 本例40X120大小的图片,切片成120列,输出标签最大设定为4(即不定长验证码最大长度为4),这样就可以用CTC模型进行优化。
   # 假设40x120的图片,数字串标签是"123",把图片按列切分（CTC会优化切分模型）,然后分出来的每块再去识别数字
   # 找出这块是每个数字或者特殊字符的概率（无法识别的则标记为特殊字符"-"）
   # 这样就得到了基于输入特征序列（图片）的每一个相互独立建模单元个体（划分出来的块）（包括“-”节点在内）的类属概率分布。
   # 基于概率分布,算出标签序列是"123"的概率P（123）,当然这里设定"123"的概率为所有子序列之和,这里子序列包括"-"和"1"、"2"、"3"的连续重复
   # tf.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
   # labels: label实际上是一个稀疏矩阵SparseTensor,即真实标签(被压缩过的)
   # inputs:是RNN的输出logits,shape=[max_time_step,batch_size,num_classes]
   # sequence_length: bt_size个1乘以char_list_image_shape[1],即bt_size个样本每个样本有多少个time_steps
   # preprocess_collapse_repeated: 设置为True的话, tensorflow会对输入的labels进行预处理, 连续重复的会被合成一个。
   # ctc_merge_repeated: 连续重复的是否被合成一个。
   cost = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len))
   # 这里用Adam算法来优化
   optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
   decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
   # tf.nn.ctc_beam_search_decoder对输入中给出的logits执行波束搜索解码。
   # ctc_greedy_decoder是ctc_beam_search_decoder中参数top_paths=1和beam_width=1（但解码器在这种特殊情况下更快）的特殊情况。
   # 如果merge_repeated是True,则合并输出序列中的重复类。这意味着如果梁中的连续条目相同,则仅发出第一个条目。
   # 如,当顶部路径为时A B B B B,返回值为：A B如果merge_repeated = True；A B B B B如果merge_repeated = False。
   # inputs：3-D float Tensor,尺寸 [max_time x batch_size x num_classes]。输入是预测的标签向量。
   # sequence_length：bt_size个1乘以char_list_image_shape[1],即bt_size个样本每个样本有多少个time_steps
   # beam_width：int标量> = 0（波束搜索波束宽度）。
   # top_paths：int标量> = 0,<= beam_width（控制输出大小）。
   # merge_repeated：布尔值。默认值：True。如果merge_repeated是True,则合并输出序列中的重复类。
   # 返回值：
   # 元组(decoded, log_probabilities)
   # decoded：decoded是一组SparseTensor。由于我们每一次训练只输入一组训练数据,所以decoded里只有一个SparseTensor。
   # 即decoded[0]就是我们这组训练样本预测得到的SparseTensor,decoded[0].indices就是其位置矩阵。
   # log_probability：包含序列对数概率的float矩阵(batch_size)。
   accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

   # tf.edit_distance(hypothesis, truth, normalize=True, name="edit_distance"),计算序列之间的(Levenshtein)莱文斯坦距离
   # 莱文斯坦距离(LD)用于衡量两个字符串之间的相似度。莱文斯坦距离被定义为将字符串a变换为字符串b所需的删除、插入、替换操作的次数。
   # hypothesis: SparseTensor,包含序列的假设.truth: SparseTensor, 包含真实序列.
   # normalize: 布尔值,如果值True的话,求出来的Levenshtein距离除以真实序列的长度. 默认为True
   # name: operation 的名字,可选。
   def do_report():
      # 生成一批样本数据,进行测试
      # 为true时使用freetype生成验证码
      test_inputs, test_targets, test_seq_len = free_type_get_next_batch(test_batch_size, char_list_image_shape)
      test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
      dd = sess.run(decoded[0], feed_dict=test_feed)
      report_acc = report_accuracy(dd, test_targets)
      # 返回准确率
      return report_acc

   def do_batch():
      # 生成一批样本数据,进行训练
      train_inputs, train_targets, train_seq_len = free_type_get_next_batch(train_batch_size, char_list_image_shape)
      train_feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
      b_cost, b_lr, b_acc, steps, _ = sess.run([cost, learning_rate, accuracy, global_step, optimizer],
                                               feed_dict=train_feed)

      return b_cost, steps, b_acc, b_lr

   # 创建模型文件保存路径
   if not os.path.exists("./free_type_image_lstm_model/"):
      os.mkdir("./free_type_image_lstm_model/")
   saver = tf.train.Saver()

   with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      if os.path.exists("./free_type_image_lstm_model/checkpoint"):
         # 判断模型是否存在,如果存在则从模型中恢复变量
         saver.restore(sess, tf.train.latest_checkpoint("./free_type_image_lstm_model/"))

      # 训练循环
      while True:
         start = time.time()
         # 每轮将一个batch的样本喂进去训练
         batch_cost, train_steps, acc, batch_lr = do_batch()
         batch_seconds = time.time() - start
         log = "iteration:{},batch_cost:{:.6f},batch_learning_rate:{:.12f},batch seconds:{:.6f}"
         print(log.format(train_steps, batch_cost, batch_lr, batch_seconds))
         if train_steps % test_display_step == 0:
            # 生成的模型存在free_type_image_lstm_model文件夹
            saver.save(sess, "./free_type_image_lstm_model/train_model", global_step=train_steps)
            acc = do_report()
            if acc > acc_reach_to_stop:
               print("准确率已达到临界值{},目前准确率{},停止训练".format(acc_reach_to_stop, acc))
               break


if __name__ == "__main__":
   train()
```
使用freetype库生成的不定长验证码来训练模型，steps达到2000-3000次左右(从头开始训练后模型收敛需要的迭代次数有一定的随机性，因为参数初始化时是随机的)后预测准确率可达0.98。