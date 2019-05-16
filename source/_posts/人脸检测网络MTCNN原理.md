---
title: 人脸检测网络MTCNN原理
date: 2019-05-08 18:58:29
tags:
- 人脸检测与识别
categories:
- 人脸检测与识别
mathjax: true
---

# MTCNN介绍
论文:Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks。
论文地址: https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf 。
论文github项目地址: https://github.com/kpzhang93/MTCNN_face_detection_alignment 。
MTCNN（Multi-task convolutional neural network，多任务卷积神经网络）是一个多任务同时进行的人脸检测网络，该网络使用3个CNN级联算法结构，可以实现人脸检测和5个特征点的标定。
# MTCNN网络的结构
该网络可分为图像金字塔、PNet、RNet、和ONet四级网络结构。
## 图像金字塔
为了检测到不同尺寸大小的人脸，在进入P-Net之前，我们要首先对图像进行金字塔操作。根据设定的min_face_size尺寸，将图像按照一定的尺寸缩小，每次将图像缩小到前级图像面积的一半，形成scales列表，直至边长小于min_face_size，此时可以得到一组不同尺寸的输入图像。
## P-Net（Proposal Network）
前面经过图像金字塔得到的这组不同尺寸的图像作为输入，使用全卷积网络（FCN）来提取与标定边框。anchor大小为12x12，经过P-Net全卷积层后变成1。（假设输出为w*h，则输出的每个点都对应原img中一个12*12的区域）
在训练时该网络最终有3条支路，分别用来做人脸分类（是人脸的概率）、人脸框回归和人脸5个关键点定位（5个点坐标，（x,y）形式），使用0.6的分类阈值来初筛人脸框，并进行边框回归调整与非极大值抑制（NMS，去除一部分重叠窗口）。最后得到的所有边框会放置在一个numberx9的数组里，number表示box的数量，9代表box的坐标信息、score、坐标回归信息[x1、y1、x2、y2、score、reg_x1、reg_y1、reg_x2、reg_y2]，利用线性回归对边框进行坐标修正。修正之后，将修正后不规则的框调整为正方形，将超出原img范围的部分填充为0，大小比例不变。
在测试时这一步的输出只有回归框的4个坐标和分类score。 
**注意:**
没有将网络输出做全连接层展开，而是直接将卷积层结果作为输出。原因是在应用端，每一个输入图像都是尺度不一的。如果增加全连接层，当尺度不一的图像输入pnet时，全连接层的特征向量也会不等长。而利用卷积层能将不同size的输入图像最终输出为同一尺寸的特征图（1x1x？）。
**网络结构:**
```
                                                                                                        --->1x1x2 face classification
                                                                                                       |
input:12x12x3--->conv:3x3 max pool:2x2 size:5x5x10--->conv:3x3  size:3x3x16--->conv:3x3  size:1x1x32--->--->1x1x4 bounding box regression
                                                                                                       |
                                                                                                        --->1x1x10 facial landmark localization
```
## R-Net（refine network）
R-Net主要用来去除大量的非人脸框。该网络输入是前面P-Net生成的回归框，每个框的大小会resize成24x24。在训练时该网络最终有3条支路，分别用来做人脸分类（是人脸的概率）、人脸框回归和人脸5个关键点定位（5个点坐标，（x,y）形式），设置score阈值，大于阈值的框才留下，留下的框进行坐标的回归修正和非极大值抑制（去除非人脸框）。最后将修正后不规则的框调整为正方形，将超出原img范围的部分填充为0，大小比例不变。
在测试时这一步的输出只有回归框的4个坐标和score。
**网络结构:**
```
                                                                                                                                            --->2 face classification
                                                                                                                                           |
input:24x24x3--->conv:3x3 max pool:3x3 size:11x11x28--->conv:3x3 max pool:3x3  size:4x4x48--->conv:2x2  size:3x3x64--->fully connect:128--->--->4 bounding box regression
                                                                                                                                           |
                                                                                                                                            --->10 facial landmark localization
```
## O-Net（output network）
O-Net使用更复杂的CNN网络进一步修正边框坐标，并输出5个人脸特征点位置坐标。该网络输入是前面R-Net最后输出的所有回归框，每个框的大小会resize成48x48。该网络最终也有3条支路，输出包含P个回归框的4个坐标、score和5个关键点坐标。 训练时根据设定的阈值，大于阈值的框才留下，留下的框进行回归框和关键点的坐标的回归修正和非极大值抑制（去除非人脸框）。最后将修正后不规则的框调整为正方形，将超出原img范围的部分填充为0，大小比例不变。
在测试时输出只有回归框的4个坐标、关键点的5个坐标和score。并将坐标映射回原图得到真实的坐标，得到一副包含人脸框与人脸关键点的检测图像。
**网络结构:**
```
                                                                                                                                                                                   --->2 face classification
                                                                                                                                                                                  |
input:48x48x3--->conv:3x3 max pool:3x3 size:23x23x32-->conv:3x3 max pool:3x3  size:10x10x64--->conv:3x3 max pool:2x2  size:4x4x64--->conv:2x2 size:3x3x128--->fully connect:256--->--->4 bounding box regression
                                                                                                                                                                                  |
                                                                                                                                                                                   --->10 facial landmark localization
```
# MTCNN的损失函数
## face classification支路的损失函数（人脸分类）
$$
L_{i}^{d e t}=-\left(y_{i}^{d e t} \log \left(p_{i}\right)+\left(1-y_{i}^{d e t}\right)\left(1-\log \left(p_{i}\right)\right)\right)
$$
采用交叉熵函数。这里是二分类。pi为是否为face的概率， yi_det是真实label。
## bounding box regression支路的损失函数（边界框回归）
$$
L_{i}^{b o x}=||\hat y_{i}^{b o x}-y_{i}^{b o x}||_{2}^{2}
$$
采用平方损失函数。
## facial landmark localization支路的损失函数（关键点回归）
$$
L_{i}^{l a n d m a r k}=||\hat y_{i}^{l a n d m a r k}-y_{i}^{l a n d m a r k}||_{2}^{2}
$$
也采用平方损失函数。
## 总损失函数
$$
\min \sum_{i=1}^{N} \sum_{j \epsilon\{d e t, b o x, l a n d m a r k\}} \alpha_{j} \beta_{i}^{j} L_{i}^{j}
$$
其中N是样本数，α是任务权重。这就是总损失函数的统一形式。
**P-Net训练时权重:**
$$
\quad \alpha_{\text {det}}=1, \alpha_{b o x}=0.5, \alpha_{\text {landmark}}=0.5
$$
**R-Net训练时权重:**
$$
\alpha_{d e t}=1, \alpha_{b o x}=0.5, \alpha_{l a n d m a r k}=0.5
$$
**O-Net训练时权重:**
$$
\alpha_{d e t}=1, \alpha_{b o x}=0.5, \alpha_{l a n d m a r k}=1
$$
$$
\beta_{j} \in [0,1] \quad 代表样本标签
$$
总损失函数可写成上面的统一形式，但是在每个级联网络进行训练时α的权重不同，所以具体的损失函数是有变化的。在P-Net和R-Net中，关键点的损失权重要小于O-Net部分，这是因为前面2个级联网络的重点在于过滤掉非人脸的回归框。β存在的意义是如果是非人脸输入，就只需要计算分类损失，而不需要计算回归和关键点的损失。 
# MTCNN网络的训练
## 三个训练任务
人脸分类任务：利用正样本和负样本进行训练；
人脸边框回归任务：利用正样本和部分样本进行训练；
关键点检测任务：利用关键点样本进行训练。
## 训练样本的生成
论文中作者主要使用了Wider_face和CelebA数据库，其中Wider_face主要用于检测任务的训练，CelebA主要用于关键点的训练。训练集分为四种:负样本，正样本，部分样本，关键点样本。样本比例为3:1:1:2。
**正样本、负样本、部分样本的提取:**
从Wider_face中随机裁切生成样本，然后和标注数据计算IOU，如果大于0.65，则为正样本，大于0.4小于0.65为部分样本，小于0.4为负样本；
计算边框偏移。对于边框，（x1,y1）为左上角坐标，（x2,y2）为右下角坐标，新剪裁的边框坐标为（xn1,yn1），（xn2,yn2），width，height。则offset_x1 =（x1 - xn1）/width。类似地，计算另三个点的坐标偏移。
正样本，部分样本均有边框信息，负样本不需要边框信息。

**关键点样本提取:**
从celeba中提取，可以根据标注的边框，在满足正样本的要求下，随机裁剪出图片，然后调整关键点的坐标。

由于训练过程中需要同时计算３个级联网络的loss函数，但是对于不同的任务，每个任务需要的loss函数不同，因此需要对每张图片进行15个标注：
第1列：正负样本标志（１为正样本，0为负样本，2为部分样本，3为关键点信息）；
第2-5列：边框偏移，float类型，对于无边框信息的数据，全部置为-1；
第6-15列：为关键点偏移，为floagt类型，对于无边框信息的数据，全部置为-1。
## 不同的级联网络训练时loss函数的变化
**总损失函数为:**
$$
\min \sum_{i=1}^{N} \sum_{j \epsilon\{d e t, b o x, l a n d m a r k\}} \alpha_{j} \beta_{i}^{j} L_{i}^{j}
$$
其中N是样本数，α是任务权重。这就是总损失函数的统一形式。
**P-Net训练时权重:**
$$
\quad \alpha_{\text {det}}=1, \alpha_{b o x}=0.5, \alpha_{\text {landmark}}=0.5
$$
**R-Net训练时权重:**
$$
\alpha_{d e t}=1, \alpha_{b o x}=0.5, \alpha_{l a n d m a r k}=0.5
$$
**O-Net训练时权重:**
$$
\alpha_{d e t}=1, \alpha_{b o x}=0.5, \alpha_{l a n d m a r k}=1
$$
$$
\beta_{j} \in [0,1] \quad 代表样本标签
$$
## 困难样本的选择
作者采用了在线困难样本选择的方法，即在训练过程中只对Loss比较大的sample进行反向传播。具体做法是，对每个小批量里所有样本计算loss，对loss进行降序，前70%samples做为hard samples进行反向传播。这个策略在分类和人脸识别中经常用。
# MTCNN网络的测试
原始图像首先经过图像金字塔，生成多个尺度的图像，然后输入PNet,，PNet由于尺寸很小，所以可以很快的选出候选区域，但是准确率不高，然后采用非极大值抑制（NMS）算法，剔除重叠候选框。根据候选框提取图像，作为RNet的输入，RNet可以精确的选取边框，一般最后只剩几个边框，最后输入ONet。ONet虽然速度较慢，但是由于经过前两个网络，已经得到了高概率的边框，所以输入ONet的图像较少，最后由ONet输出精确的边框和关键点信息。