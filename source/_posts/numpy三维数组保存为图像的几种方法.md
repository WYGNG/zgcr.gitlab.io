---
title: numpy三维数组保存为图像的几种方法
date: 2019-02-28 10:11:53
tags:
- Python
categories:
- Python
---

# 使用scipy.misc
```python
from PIL import Image
import numpy as np
from scipy import misc

image = Image.open("0_train.jpg")  # 首先在该py文件所在目录下随便放一张图片，使用PIL.Image库的open方法打开
image_array = np.array(image)  # 使用numpy将该图片的二进制数据转换成多维数组形式
print(image_array)
misc.imsave('out.jpg', image_array)  # 使用misc.imsave方法将数组保存为图片
```
# 使用PIL库
```python
from PIL import Image
import numpy as np

im = Image.open("0_train.jpg")  # 打开图片
im_array = np.array(im)  # 将图片转化为numpy数组
print(im_array)
img = Image.fromarray(im_array).convert('RGB')  # 将数组转化回图片
img.save("out.bmp")  # 将数组保存为图片
```
# 使用matplotlib库
```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 使用PIL库和numpy是只是为了快速得到一个可以用于保存为图片的数组，即从现有的图片直接转换成数组
im = Image.open("0_train.jpg")  # 打开图片
im_array = np.array(im)  # 将图片转化为numpy数组
print(im_array)
plt.imshow(im_array)  # 绘制图片
plt.savefig("out_plt2.png")  # 保存图片
```
注意这种方式生成的图片默认是带坐标轴的。你可以使用matplotlib.pyplot中相关方法隐藏坐标轴。另外这种绘制出的图片四周有空白。
# 使用opencv库
```python
from PIL import Image
import numpy as np
import cv2

im = Image.open("0_train.jpg")  # 打开图片
im_array = np.array(im)  # 将图片转化为numpy数组
print(im_array)
cv2.imwrite("out_cv2.jpg", im_array)  # 保存图片
```