---
title: 八度卷积Octave Convolution原理
date: 2019-06-03 18:04:11
tags:
- 深度学习原理推导
categories:
- 深度学习原理推导
mathjax: true
---

# Octave Convolution介绍
论文:Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
论文地址:https://export.arxiv.org/pdf/1904.05049 。
代码地址:https://github.com/facebookresearch/OctConv 。

在本文的Octave Convolution方法之前，传统的CNN卷积神经网络是直接在图像上（也称为空域）进行卷积的。
对于数字图像这种离散的空域信号，频率大小表示信号变化的剧烈程度。频率越大，变化越剧烈，频率越小，信号越平缓，对应到图像中，高频信号往往是图像中的边缘信号和噪声信号，而低频信号包含图像中的物体轮廓及背景等信号。
因此，在传统的图像处理中，我们可以借助傅里叶变换将图像（空间域）转为频域图（频率域）。一张图像转换为频域图后可分为高频部分和低频部分。其中低频部分表示图像对应的灰度图中变化平缓的部分（图片中物体的整体结构），而高频部分表示图像对应的灰度图中变化剧烈的部分（图片中各个物体的边缘细节和噪声），通过只保留频域图的低频部分（理想低通滤波、Butterworth低通滤波）或只保留频域图的高频部分（理想高通滤波、Butterworth高通滤波），我们将保留后的频域图通过逆傅里叶变换还原成图像（空间域），可以实现对图像的平滑（理想低通滤波、Butterworth低通滤波）或边缘提取（理想高通滤波、Butterworth高通滤波）。比如，一张切的图片，其频域图的低频部分代表图像上企鹅这个物体内部的白色大肚皮、黑色的北部以及图片的背景；其频域图的高频部分代表图像上企鹅这个物体的边缘线条部分。
如果对传统图像处理中的频域变换这块还是不太清楚，建议看这本书中相应章节:数字图像处理（第三版）（美）冈萨雷斯，（美）伍兹。
**在本文中，作者受传统图像处理中的频域变换的启发，认为特征图也有对应的频域图，其频域图也分为低频部分和高频部分。显然，低频分量是存在冗余的，在编码过程中可以节省。为了降低空间冗余，作者将CNN中的空域上的特征图映射为频域图，并设计了一种全新的卷积运算：Octave Convolution (OctConv)，使用低维度的张量来储存和处理特征图对应的频域图中的低频部分，通过降低低频特征的分辨率，降低了内存和计算成本。Octave一词表示"八音阶"或"八度"，在音乐里降8个音阶表示频率减半。**

自然图像可以分解为描述平稳变化结构的低空间频率分量和描述快速变化精细细节的高空间频率分量。同样，我们认为卷积层的输出特征映射也可以分解为不同空间频率的特征，并提出了一种新的多频特征表示方法，将高频和低频特征映射存储到不同的组中。显然，低频分量中存在大量冗余，在编码过程中可以节省。
本文通过相邻位置间的信息共享，可以安全降低低频组的空间分辨率，减少空间冗余。为了适应新的特征表示，本文提出Octave Convolution（OctConv），它接收包含两个频率的特征映射，并直接从低频映射中提取信息，而无需解码回到高频。作为传统卷积的替代，OctConv消耗的内存和计算资源都大大减少。此外，OctConv利用相应的（低频）卷积处理低频信息，有效地扩大了原始像素空间的感受野，从而提高识别性能。
本文以一种通用的方式设计OctConv，使它成为卷积的替代，而且即插即用。由于OctConv主要侧重于处理多空间频率的特征映射并减少其空间冗余，它与现有的方法是相交且互补的，现有的方法侧重于构建更好的CNN拓扑结构，减少卷积特征映射中的信道冗余和密集模型参数中的冗余。
此外，与利用多尺度信息的方法不同，OctConv 可以很容易地部署为即插即用单元，以替代卷积，而不需要改变网络结构或需要超参数调优。
我们的实验证明，通过简单地用 OctConv 代替传统卷积，可以持续提高流行的2D CNN模型的 ImageNet 图像识别性能，包括 ResNet ResNeXt, DenseNet, MobileNet，以及 SE-Net。采用OctConv的Oct-ResNet-152超过了手工设计的state-of-the-art网络，并且所需的内存和计算成本更低。
**本文的贡献可以总结如下:**
* 将卷积特征映射分解成不同空间频率的两个组，并分别以相应的频率处理不同的卷积，相隔一个八度 (octave)。由于可以降低低频图的分辨率，因此能够节省存储和计算。这也有助于每一层获得更大的感受野，以捕获更多的上下文信息。
* 设计了一种即插即用的运算，名为OctConv，用来代替传统的卷积运算。OctConv直接对新的特征表示进行运算，减少了空间冗余。更重要的是，OctConv在实践中速度很快，接近了理论的极限。
* 我们广泛研究了所提出的OctConv在用于图像和视频任务的各种骨干CNN上的特性，并获得了显著的性能提高，甚至可以与最好的AutoML网络相媲美。

# Octave Feature Representation（八度特征表示）
对于普通卷积而言，所有输入和输出特征图的通道都有着一样的分辨率。而本文提出的八度特征表示中，低频特征图的分辨率仅有高频特征图的一半。
使用:
$$
X \in R^{c\times h \times w}
$$
表示输入的特征图。作者将输入的特征图X分解为高频特征图XH和低频特征图XL（其实就是原始特征图的c个通道，拿出(1-αin)c个通道作为高频特征图输入XH，剩余αin x c个通道作为低频特征图输入XL）:
$$
X=(X^{H}, X^{L})
$$
高频部分:
$$
X^{H} \in R^{(1-\alpha) \times c\times h \times w}
$$
表示物体边缘细节。
低频部分:
$$
X^{L} \in R^{\alpha \times c\times h \times w}
$$
其中，α∈[0,1]表示channel被分配到低频部分的比率。

由于输入特征的空间分辨率不同，传统卷积不能直接对这种表示进行操作。避免这个问题的一种简单方法是将低频部分XL上采样到原始的空间分辨率，将它与XH连接起来，然后进行卷积，这将导致额外的计算和内存开销。为了充分利用紧凑的多频特征表示，我们提出Octave Convolution，它可以直接在分解后的X张量上运行，而不需要任何额外的计算或内存开销。

# Octave Convolution（八度卷积）
**传统卷积（Vanilla Convolution）:**
令:
$$
W \in R^{c \times k \times k}
$$
表示一个kxk，c个通道的卷积核。
$$
X, Y \in R^{c \times h \times w}
$$
表示输入张量和输出张量。
$$
Y_{p, q} \in R^{c}
$$
中的每个feature map可以下面的公式计算:
$$
Y_{p, q}=\sum_{i, j \in N_{k}} W_{i+\frac{k-1}{2}, j+\frac{k-1}{2}}^{\top} X_{p+i, q+j}
$$
其中（p, q）为位置坐标。
$$
N_{k}=\left[(i, j) : i=\left(-\frac{k-1}{2}, \ldots, \frac{k-1}{2}\right), j=\left(-\frac{k-1}{2}, \ldots, \frac{k-1}{2}\right)\right]
$$
即对应卷积核大小的特征图上（p, q）的局部邻域。为简单起见，在所有的方程中我们省略填充，我们假设k是一个奇数，并且输入和输出数据具有相同的维数，即:
$$
c_{i n}=c_{o u t}=c
$$
**八度卷积（Octave Convolution）:**
设八度卷积的输入为Y，那么Y的八度特征表示为:
$$
Y=(Y^{H}, Y^{L})
$$
用:
$$
Y^{A \rightarrow B}
$$
表示从特征图A到B的卷积更新过程。则有:
$$
Y^{H}=Y^{H \rightarrow H}+Y^{L \rightarrow H},Y^{L}=Y^{H \rightarrow L}+Y^{L \rightarrow L}
$$
其中
$$
Y^{H \rightarrow H}, Y^{L \rightarrow L}
$$
表示高频率特征和低频率特征的频率内信息更新。
$$
Y^{L \rightarrow H}, Y^{H \rightarrow L}
$$
表示低频率与高频率间信息交流。 
为了完成卷积运算，本文将卷积核也分为两部分WH、WL，分别用于卷积XH、XL。每个部分又可以进一步分为频率内和频率间两个部分:
$$
\left[W^{H}, W^{L}\right],W^{H}=\left[W^{H \rightarrow H}, W^{L \rightarrow L}\right],
W^{L}=\left[W^{L \rightarrow L}, W^{H \rightarrow L}\right]
$$
WH→H是传统卷积，因为输入、输出图像尺寸一样大；对于WL→H部分，我们先对输入图像进行升采样（upsample），再执行传统卷积。WL→L也是传统卷积，WH→L先执行的是降采样，然后再执行传统卷积。
为了控制输入和输出特征图的低频信息部分的比例，作者令第一层和最后一层八度卷积层的超参数:
$$
\alpha_{i n}=0, \alpha_{o u t}=\alpha
$$
而中间的八度卷积层超参数则设为:
$$
\alpha_{i n}=\alpha_{o u t}=\alpha
$$
如此一来，即可完成即插即用的替换。 

现在设输入的八度特征为:
$$
Y=(Y^{H}, Y^{L})
$$
与之相对应的输出也由两部分组成，即高频特征输出与低频特征输出，如下:
$$
Y_{p, q}^{H}=Y_{p, q}^{H \rightarrow H}+Y_{p, q}^{L \rightarrow H},Y_{p, q}^{L}=Y_{p, q}^{L \rightarrow L}+Y_{p, q}^{H \rightarrow L}
$$
其中第一项为高频输出特征，第二项为低频输出特征。
对于高频特征图，它的频率内信息更新过程就是普通卷积过程，而频率间的信息交流过程，则可以对使用上采样操作然后再进行普通卷积。类似地，对于低频特征图，它的频率内信息更新过程就是普通卷积过程，而频率间的信息交流过程则通过对进行平均池化操作（下采样）然后再进行普通卷积。
具体计算公式如下:
$$
Y_{p, q}^{H}=Y_{p, q}^{H \rightarrow H}+Y_{p, q}^{L \rightarrow H}
$$
$$
=\sum_{i, j \in N_{k}} W_{i+\frac{k-1}{2}, j+\frac{k-1}{2}}^{\top} X_{p+i, q+j}^{H}+\sum_{i, j \in N_{k}} W_{i+\frac{k-1}{2}, j+\frac{k-1}{2}}^{\top} X_{\left(\left\lfloor\frac{p}{2}\right]+i\right),\left(\left\lfloor\frac{q}{2}\right]+j\right)}^{L}
$$
$$
Y_{p, q}^{L}=Y_{p, q}^{L \rightarrow L}+Y_{p, q}^{H \rightarrow L}
$$
$$
=\sum_{i, j \in N_{k}} W_{i+\frac{k-1}{2}, j+\frac{k-1}{2}}^{\top} X_{p+i, q+j}^{L}+\sum_{i, j \in N_{k}} W_{i+\frac{k-1}{2}, j+\frac{k-1}{2}}^{\top} X_{(2 \ast p+0.5+i),(2 \ast q+0.5+j)}^{H}
$$
在八度卷积中，高频特征图卷积需要经过下采样，随后才能卷积到低频特征图。在这里作者分析了两种降采样方式：stride=2的convolution与average pooling。作者发现，使用步长为2的卷积之后（高频到低频），再经过上采样（低频到高频）会导致出现中心偏移的错位情况（misalignment），如果此时继续进行特征图融合中会造成特征不对齐，进而影响性能。所以作者最终选择了average pooling来进行下采样。
（2p + 0.5 +i）,（2q + 0.5 +j）中的0.5是因为让从H到L下采样之后的特征图与input一致以消除不对齐。
最终输出的高频输出特征与低频输出特征如下:
$$
Y^{H}=f\left(X^{H} ; W^{H \rightarrow H}\right)+\text { upsample }\left(f\left(X^{L} ; W^{L \rightarrow H}\right), 2\right)
$$
$$
Y^{L}=f\left(X^{L} ; W^{L \rightarrow L}\right)+f\left(\text{average pool}\left(X^{H}, 2\right) ; W^{H \rightarrow L}\right) )
$$
**Octave卷积核:**
Octave卷积核的尺寸如下:
```
WH→H                            WH→L
k x (1-αin)cin x (1-αout)cout   k x (1-αin)cin x αoutcout
WL→H                            WL→L
k x αincin x (1-αout)cout       k x αincin x αoutcout
WH→H和WL→H一起产生高频特征图
WL→L和WH→L一起产生低频特征图
```
对于八度卷积而言，由于低频特征图的分辨率变小，实际上八度卷积的感受野反而变大了，所以在使用卷积核去卷积低频特征图情况下，八度卷积有着几乎等价于普通卷积2倍普通卷积感受野的能力，这可以进一步帮助八度卷积层捕捉远距离的上下文信息从而提升性能。 

对于低频特征所使用的低频所占比例α的值不同，则网络需要的计算力和内存消耗也不同。当α=0时（即没有低频成分），OctConv就会退化为普通卷积。α=1.0时，计算力和内存消耗只有α=0时的25%。需要注意的是，无论比例α选择是多少，OctConv卷积核的参数数量都与同尺寸的普通卷积核相同（kxk的Octave卷积核与普通的kxk卷积核具有完全相同的参数量）。

# 实验和结论
在实验和评估部分，我们验证了 Octave Convolution 在 2D 和 3D 网络中的有效性和效率。我们分别进行了 ImageNet 上图像分类的研究，然后将其与目前最先进的方法进行了比较。然后，我们用Kinetics-400和dynamics 600数据集，证明所提出的OctConv也适用于 3D CNN。采用OctConv的模型比基线模型更有效、更准确。
本文针对传统CNN模型中普遍存在的空间冗余问题，提出了一种新颖的Octave Convolution运算，分别存储和处理低频和高频特征，提高了模型的效率。Octave卷积具有足够的通用性，可以代替常规的卷积运算，可以在大多数二维和三维CNNs中使用，无需调整模型结构。除了节省大量的计算和内存外，Octave Convolution还可以通过在低频段和高频段之间进行有效的通信，增大接收域的大小，从而获得更多的全局信息，从而提高识别性能。我们在图像分类和视频动作记录方面进行了广泛的实验验证了我们的方法在识别性能和模型效率之间取得更好权衡的优越性，不仅在失败的情况下，而且在实践中也是如此。