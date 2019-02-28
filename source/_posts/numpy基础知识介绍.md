---
title: numpy基础知识介绍
date: 2019-02-28 11:12:04
tags:
- Python
categories:
- Python
---

#  reshape(-1,1)与reshape(1,-1)详解
reshape(行，列)可以根据指定的数值将数据转换为特定的行数和列数，即转换成矩阵。
reshape(-1,1)则比较特殊，根据numpy库官网介绍，这里的-1为未指定值。我们在规定了第1号维度上的元素个数是1后，第0号维度的值由numpy自动计算，并保证所有元素的个数与原来的数组元素的个数相等。同理，reshape(1,-1)即规定了第0号维度上的元素个数是1后，第1号维度的值由numpy自动计算。
举例：
```python
import numpy as np

z = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

print(z.shape)
z1 = z.reshape(-1)
z2 = z.reshape(-1, 1)
z3 = z.reshape(-1, 2)
z4 = z.reshape(1, -1)
z5 = z.reshape(2, -1)
print(z1, z1.shape)
print(z2, z2.shape)
print(z3, z3.shape)
print(z4, z4.shape)
print(z5, z5.shape)
```
运行结果如下：
```python
(4, 4)
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16] (16,)
[[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]
 [12]
 [13]
 [14]
 [15]
 [16]] (16, 1)
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]
 [13 14]
 [15 16]] (8, 2)
[[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]] (1, 16)
[[ 1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16]] (2, 8)

Process finished with exit code 0
```