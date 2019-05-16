---
title: GoogLeNet和Inception v1、v2、v3、v4网络介绍
date: 2019-05-16 22:11:04
tags:
- 深度学习原理推导
categories:
- 深度学习原理推导
---

# CNN神经网络的演化过程
```
Hubel&Wiesel
     |
Neocognitron
     |
 LeCun1989
     |
   LeNet
     |增加dropout、relu
  AlexNet
     |多种演化方向
网络加深     增强卷积模块                     目标检测/实例分割             增加新功能单元
   |            |                         /              \                  |     
VGG16          NN                      RCNN           Yolo/SSD     Inception V2（增加BN）
   |            |                        |                |                 |
VGG19       GoogLeNet                fast-RCNN         Yolov2          FCN/FCN+CRF
   |            |                        |                |                 |
MSRANet   Inception V3/V4           faster-RCNN        Yolov3             STNet
   \            /                        |                                  |
    前两条路线合并                      mask-RCNN                        CNN+RNN/LSTM
         |
       ResNet
         |
  Inception ResNet
```
在GoogLeNet出现之前，对神经网络的修改往往是单纯增加层数和宽度，但是这样导致网络的参数变得非常多，容易出现梯度消失问题，也更加难以训练。
# GoogLeNet原始版本
GoogLeNet相比于之前的卷积神经网络的最大改进是设计了一个稀疏参数的网络结构，但是能够产生稠密的数据，既能增加神经网络表现，又能保证计算资源的使用效率。
**具体来说，就是将CNN中常用的卷积（1x1，3x3，5x5）、池化操作（3x3）堆叠在一起（卷积、池化后的尺寸相同，将通道相加），一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。**
**GoogLeNet结构如下:**
```
                                           前一层
                       /             /                   \                  \
                   1x1 conv      3x3 conv             5x5 conv        3x3 max pooling
                       \             \                   /                  /
                                          全连接层
```
**稀疏参数的网络结构特点:**
使用不同大小的卷积核意味着不同大小的感受野，最后拼接意味着不同尺度特征的融合； 
之所以卷积核大小采用1、3和5，主要是为了方便对齐。设定卷积步长stride=1之后，只要分别设定pad=0、1、2，那么卷积之后便可以得到相同维度的特征图，然后这些特征图就可以直接拼接在一起；
3x3的max pooling对提取特征效果也不错，所以也增加pooling结构；
越是深处的网络层，特征越抽象，而且每个特征所涉及的感受野也更大了，因此随着层数的增加，3x3和5x5卷积的比例也要增加。

由于5x5的卷积核计算量仍然很大，造成特征图的通道数很多，因此又进行了一些改进。
# GoogLeNet Inception V1
论文:Going deeper with convolutions
论文地址:https://arxiv.org/pdf/1409.4842.pdf 。
Inception V1在GoogLeNet基础之上，为了减少5x5卷积的计算量，在3x3conv前、5x5conv前、3x3max pooling后分别加上1x1的卷积核，起到减少总的网络参数数量的作用。
**Inception V1结构如下:**
```
                                           前一层
                       /             /                   \                  \
                   1x1 conv      1x1 conv             1x1 conv        3x3 max pooling
                      |             |                     |                  |
                      |          3x3 conv             5x5 conv            1x1 conv 
                       \             \                   /                  /
                                         特征图拼接
```
假如前一层的输出为100x100x128，经过具有256个输出的5x5卷积层之后(stride=1，pad=2)，输出数据为100x100x256。其中，卷积层的参数为128x5x5x256。假如上一层输出先经过具有32个输出的1x1卷积层（1x1卷积降低了通道数，且特征图尺寸不变），再经过具有256个输出的5x5卷积层，最终的输出数据仍为为100x100x256，但卷积参数量已经减少为128x1x1x32 + 32x5x5x256，参数数量减少为原来的约4分之一。
**1x1卷积核的作用:**
1x1卷积核的最大作用是降低输入特征图的通道数。假设输入为6x6x128的特征图，1x1卷积核为1x1x32（32个通道），则输出为6x6x32。即当1x1卷积核的个数小于输入特征图的通道数时，起到降维的作用。
**Inception V1相比GoogLeNet原始版本进行了如下改进:**
* 为了减少5x5卷积的计算量，在3x3conv前、5x5conv前、3x3max pooling后分别加上1x1的卷积核，减少了总的网络参数数量；
* 网络最后层采用平均池化（average pooling）代替全连接层，该想法来自NIN（Network in Network），事实证明这样可以将准确率提高0.6%。但是，实际在最后还是加了一个全连接层，主要是为了方便对输出进行灵活调整；
* 网络中仍然使用Dropout ; 
* 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度（辅助分类器）。辅助分类器是将中间某一层的输出用作分类，并按一个较小的权重（0.3）加到最终分类结果中，这样相当于做了模型融合，同时给网络增加了反向传播的梯度信号，也提供了额外的正则化，对于整个网络的训练很有裨益。在实际测试时，这两个额外的softmax会被去掉。

**使用Inception V1结构改进的GoogLeNet网络结构:**
* 输入:原始输入图像为224x224x3，且都进行了零均值化的预处理操作（图像每个像素减去均值）。
* 第一层（卷积层）:使用7x7的卷积核（滑动步长2，padding为3），64通道，输出为112x112x64，卷积后进行ReLU操作，经过3x3的max pooling（步长为2），输出为((112 - 3+1)/2)+1=56，即56x56x64，再进行ReLU操作。
* 第二层（卷积层）:使用3x3的卷积核（滑动步长为1，padding为1），192通道，输出为56x56x192，卷积后进行ReLU操作，经过3x3的max pooling（步长为2），输出为((56 - 3+1)/2)+1=28，即28x28x192，再进行ReLU操作
* 第三层（Inception 3a层）:分为四个分支，采用不同尺度的卷积核来进行处理。
  64个1x1的卷积核，然后RuLU，输出28x28x64；
  96个1x1的卷积核，作为3x3卷积核之前的降维，变成28x28x96，然后进行ReLU计算，再进行128个3x3的卷积（padding为1），输出28x28x128；
  16个1x1的卷积核，作为5x5卷积核之前的降维，变成28x28x16，进行ReLU计算后，再进行32个5x5的卷积（padding为2），输出28x28x32；
  pool层，使用3x3的核（padding为1），输出28x28x192，然后进行32个1x1的卷积，输出28x28x32；
  将四个结果进行连接，对这四部分输出结果的第三维并联，即64+128+32+32=256，最终输出28x28x256。
* 第三层（Inception 3b层）
  128个1x1的卷积核，然后RuLU，输出28x28x128；
  128个1x1的卷积核，作为3x3卷积核之前的降维，变成28x28x128，进行ReLU，再进行192个3x3的卷积（padding为1），输出28x28x192；
  32个1x1的卷积核，作为5x5卷积核之前的降维，变成28x28x32，进行ReLU计算后，再进行96个5x5的卷积（padding为2），输出28x28x96；
  pool层，使用3x3的核（padding为1），输出28x28x256，然后进行64个1x1的卷积，输出28x28x64；
  将四个结果进行连接，对这四部分输出结果的第三维并联，即128+192+96+64=480，最终输出输出为28x28x480。
* 第四层（4a,4b,4c,4d,4e）、第五层（5a,5b）……，与3a、3b类似，在此就不再重复叙述。
# GoogLeNet Inception V2
论文:Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
论文地址:https://arxiv.org/pdf/1502.03167.pdf 。
**Inception V2结构如下:**
```
                                           前一层
                       /             /                   \                  \
                   1x1 conv      1x1 conv               pool              1x1 conv 
                      |             |                     |                  |
                   3x3 conv      3x3 conv             1x1 conv               |
                      |             |                     |                  | 
                   3x3 conv         |                     |                  |
                       \             \                   /                  /
                                         特征图拼接
```
**Inception V2相比Inception V1进行了如下改进:**
使用Batch Normalization，加快模型训练速度；
使用两个3x3的卷积代替5x5的大卷积，降低了参数数量并减轻了过拟合；
增大学习速率并加快学习衰减速度以适用BN规范化后的数据；
去除Dropout并减轻L2正则化（因BN已起到正则化的作用）；
更彻底地对训练样本进行打乱；
减少数据增强过程中对数据的光学畸变（因为BN训练更快，每个样本被训练的次数更少，因此更真实的样本对训练更有帮助）。
# GoogLeNet Inception V3
论文:Rethinking the Inception Architecture for Computer Vision
论文地址:https://arxiv.org/pdf/1512.00567.pdf 。
Inception V3一个最重要的改进是卷积分解（Factorization），将7x7卷积分解成两个一维的卷积串联（1x7和7x1），3x3卷积分解为两个一维的卷积串联（1x3和3x1），这样既可以加速计算，又可使网络深度进一步增加，增加了网络的非线性（每增加一层都要进行ReLU）。
另外，网络输入从224x224变为了299x299。
# GoogLeNet Inception V4
论文:Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
论文地址:https://arxiv.org/pdf/1602.07261.pdf 。
inception v4把原来的inception结构中加入了ResNet中的Residual Blocks结构，把一些层的输出加上前几层的输出，这样中间这几层学习的实际上是残差。
另外就是V4把一个先1x1卷积再3x3卷积换成了先3x3卷积再1x1卷积。 
论文说引入ResNet中的Residual Blocks结构不是用来提高准确度，只是用来提高模型训练收敛速度。