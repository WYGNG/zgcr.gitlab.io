---
title: Object Detection one-stage算法的演进：YOLO、SSD、YOLOv2、YOLOv3
date: 2019-03-11 10:38:03
tags:
- 计算机视觉
categories:
- 计算机视觉
mathjax: true
---

# Object Detection算法的分类
two-stage算法，这类算法将检测问题划分为两个阶段，首先产生候选框，然后对候选框分类（一般还需要对位置精修），这一类的典型代表是RCNN, Fast RCNN, Faster RCNN算法家族。他们的识别错误率低，漏识别率也较低，但速度较慢，不能满足实时检测场景。
one-stage算法直接产生物体的类别得分和位置坐标值，经过单次检测即可直接得到最终的检测结果，因此有着更快的检测速度，比较典型的算法如YOLO，SSD，YOLOv2，YOLOv3等。
# YOLOv1（you only look once）
论文:You Only Look Once: Unified, Real-Time Object Detection
论文地址: https://arxiv.org/pdf/1506.02640.pdf 。
YOLO算法中把目标检测问题处理成回归问题，用单个神经网络结构就可以从输入图像直接预测回归框坐标和框中物体类别概率。
## YOLO网络计算过程
YOLO网络首先将一个输入图像分成SxS个网格（grid cell），如果某个object的中心落在这个网格中，则这个网格就负责预测这个object。
每个网格要预测B个bounding box（包裹住目标的框），每个bounding box除了要回归自身的位置之外，还要附带预测一个confidence值。这个confidence代表了所预测的box中是否含有object的置信度和这个bounding box坐标值预测的有多准两重信息：
$$
Pr(\text { Object }) \times \text { IOU }
$$
其中如果有object落在一个网络（grid cell）里，第一项取1，否则取0。第二项是预测的bounding box和实际的边框坐标之间的IOU值。
除此之外，每个网格还要预测一个类别信息，记为C类。即SxS个网格，每个网格除了要预测B个bounding box外，还要预测C个类别的概率。输出就是SxSx（5xB+C）的一个向量。需要注意的是，C个类别对应的是网格而不是bounding box，而confidence信息是针对每个bounding box的。
在论文中，作者使用PASCAL VOC数据集，输入448x448图像，取S=7，共49个网格。B=2，一共有20个类别，C=20。则输出就是7x7x30的一个向量。
在测试时，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score:
$$
Pr(Class_{i} | Object) \times Pr(Object) \times \text { IOU }=Pr(Class_{i}) \times \text { IOU }
$$
其中等式左边第一项就是每个网格预测的类别信息，第二、三项就是每个bounding box预测的confidence和bounding box与真实边框的IOU。这个乘积既包含了预测的box属于某一类的概率，也有该box准确度的信息。
得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，然后对保留的bounding boxes进行非极大值抑制，去掉一些bounding boxes，就得到最终的检测结果。
其中中心点坐标x、y用bounding box相对grid的偏置归一化到0-1之间，w、h除以图像的width和height也归一化到0-1之间。

YOLO网络的实现采用GoogLeNet网络为主体修改而成，卷积层主要用来提取特征，全连接层主要用来预测类别概率和坐标。最后的输出是7x7x30的向量。
## YOLO网络的损失函数
$$
(\lambda_{coord}) \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} 1_{i j}^{obj }\left[(x_{i}-\hat x_{i})^{2}+(y_{i}-\hat y_{i})^{2}\right]
$$
$$
+(\lambda_{coord}) \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} 1_{i j}^{obj}\left[(\sqrt{w_{i}}-\sqrt{\hat w_{i}})^{2}+(\sqrt{h_{i}}- \sqrt{\hat h_{i}})^{2} \right]
$$
$$
+\sum_{i=0}^{S^{2}} \sum_{j=0}^{B} 1_{i j}^{obj}(C_{i}-\hat C_{i})^{2}
$$
$$
+(\lambda_{noobj}) \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} 1_{i j}^{noobj }(C_{i}-\hat C_{i})^{2}
$$
$$
+\sum_{i=0}^{S^{2}} 1_{i}^{obj} \sum_{c}(p_{i}(c)-\hat p_{i}(c))^{2}
$$
其中第一和第二项判断第i个网格的第j个box是否负责这个object，做坐标预测；第三项为含object的box的confidence预测；第四项为不含object的box的confidence预测；第五项判断是否有object中心落在网格i中，是类别预测，c是类别集合。
因此，YOLO的loss函数包含四部分：位置误差、包含object的confidence误差、不包含object的confidence误差、分类误差。几种误差均采用了均方误差函数。
其中yolo中位置误差权重为5，类别误差权重为1。由于我们不是特别关心不包含物体的bounding box，故赋予不包含物体的box的置信度confidence误差的权重为0.5，包含物体的权重则为1。
## YOLO网络的优缺点
**优点:**
* YOLO检测物体非常快，在Titan X的GPU上能达到45 FPS；
* YOLO可以很好的避免背景错误。不像其他物体检测系统使用了滑窗或候选框，分类器只能得到图像的局部信息。YOLO在训练和测试时都能够看到一整张图像的信息，因此YOLO在检测物体时能很好的利用上下文信息，从而不容易在背景上预测出错误的物体信息。和Fast-R-CNN相比，YOLO的背景错误不到Fast-R-CNN的一半；
* YOLO可以学到高度泛化的特征，在迁移学习时的表现比Fast RCNN要好。

**缺点:**
* YOLO的物体检测精度低于two-stage算法，主要是因为正样本与负样本（背景）极其不均衡，导致模型准确度稍低；
* YOLO容易产生物体定位错误；
* YOLO对小物体的检测效果不好，尤其是密集的小物体。

# SSD（Single Shot MultiBox Detector）
论文:SSD：Single Shot MultiBox Detector
论文地址: https://arxiv.org/pdf/1512.02325.pdf 。
SSD网络采取了one stage的思想，并且在网络中融入了Faster R-CNN中的anchors，同时做了特征分层提取并依次计算边框回归和分类的操作，由此可以适应多种尺度目标的训练和检测任务。
## SSD网络结构
SSD网络的主结构是VGG16，前五层VGG16的前五层网络，后面接Conv6和Conv7（卷积核3x3x1024和1x1x1024）是将VGG16的后两层全连接层网络转换而来。在之后并增加了4个卷积层（卷积核1x1x256、1x1x128、1x1x128、1x1x128）来构造网络结构。
## SSD网络新特性
**default box:**
类似于Faster RCNN中的anchor，SSD中的default box长宽比包括1, 2, 3, 1/2, 1/3这几种。当长宽比为1时，作者还额外增加一种default box，一共有6种default box。 
**多尺度的特征图上的目标检测和分类:**
第3、7、8、9、10卷积层和第11pooling层的特征图均输入extra feature层作目标检测和分类，使用一个3x3的卷积核输出每个default box检测到不同类别物体的概率，输出个数为预测类别个数。使用一个3x3的卷积核检测每个default box的位置（x, y, w, h）。需要注意的是各层做目标检测和分类的特征图选定的k值不同，分别是4、6、6、6、6、6。当k=6表示有[1, 2, 3, 1/2, 1/3], 共5种长宽比, 而1又包括了2种输出大小。
SSD网络在特征图的每个像素点预测K个Box。对于每个Box，预测C个类别得分，以及相对于Default Bounding Box的4个偏移值，这样需要（C+4）K个预测器，在MxN的特征图上将产生（C+4）KMN个预测值。
所有特征图上一共8732个default box，计算如下:
$$
\left(38 \times 38 \times 4+19 \times 19 \times 6+10 \times 10 \times 6+5 \times 5 \times 6+3 \times 3 \times 4+1 \times 1 \times 4\right)=8732
$$
## SSD网络的损失函数
$$
L(x, c, l, g)=\frac{1}{N}\left(L_{c o n f}(x, c)+\alpha L_{l o c}(x, l, g)\right)
$$
回归loss smoothL1为:
$$
L_{loc}(x, l, g)=\sum_{i \in \text{Pos}}^{N} \sum_{m \in\{c x, c y, w, h\}} x_{i j}^{k} smooth_{LI}(l_{i}^{m}-\hat g_{j}^{m})
$$
$$
\hat g_{j}^{c x}=(g_{j}^{c x}-d_{i}^{c x}) / d_{i}^{w}
$$
$$
\hat g_{j}^{c y}=(g_{j}^{c y}-d_{i}^{c y}) / d_{i}^{h}
$$
$$
\hat g_{j}^{w}=\log \left(\frac{g_{j}^{w}}{d_{i}^{w}}\right)
$$
$$
\hat g_{j}^{h}=\log \left(\frac{g_{j}^{h}}{d_{i}^{h}}\right)
$$
其中xij为第i个预测框与第j个真实框是否匹配，取值0或1；lm为预测框坐标，gm为真实框坐标。
分类loss为交叉熵损失函数:
$$
L_{c o n f}(x, c)=-\sum_{i \in \text{Pos}}^{N} x_{i j}^{p} \log (\tilde c_{i}^{p})-\sum_{i \in\text{Neg}} \log (\hat c_{i}^{0})
$$
其中
$$
\hat c_{i}^{p}=\frac{\exp (c_{i}^{p})}{\sum_{p} \exp (c_{i}^{p})}
$$
## SSD网络训练技巧
**数据增强:**
随机剪裁，采样一个片段，使剪裁部分与目标重叠分别为0.1, 0.3, 0.5, 0.7, 0.9，剪裁完resize到固定尺寸；然后以0.5的概率随机水平翻转。
**抽样方式:**
正样本:一张图中, 与物体重合最高的anchor, 设为正样本； 另外，物体bbox与anchor的IOU值大于0.5也设为正样本。
负样本:对负样本的loss排序，选择loss最大的样本，并使得正负样本比例为1:3。
**非极大值抑制:**
SSD预测类别和回归框后的default box，先过滤掉类别概率低于阈值的default box，然后也要采用非极大值抑制筛选，筛掉重叠度较高的default box。只不过SSD综合了各个不同feature map上的目标检测输出的default box。
## SSD算法的优缺点
SSD算法的运行速度超过YOLO，对于系数场景的大目标，精度超过Faster-rcnn；
SSD需要人工设置default box的min_size，max_size和aspect_ratio值。网络中default box的基础大小和形状不能直接通过学习获得，而是需要手工设置。而网络中每一层feature使用的default box大小和形状恰好都不一样，导致调试过程非常依赖经验；
SSD虽然采用了多尺度的特征图上的目标检测和分类的思路，但是对小目标的recall依然一般，相对于Faster RCNN的优势不明显。这可能是因为SSD使用conv4_3的低级feature去检测小目标，而低级特征卷积层数少，存在特征提取不充分的问题；
如果场景是密集的包含多个小目标的，建议用Faster-rcnn，针对特定的网络进行优化，也可以继续加速。如果你的应用对速度要求很苛刻，那么首先考虑SSD。
# YOLOv2
论文:YOLO9000: Better, Faster, Stronger
论文地址: https://arxiv.org/pdf/1612.08242.pdf 。
YOLOv1网络主要有两个缺点，一是框位置预测不准确，另一个缺点在于和基于候选框的方法相比召回率较低。因此YOLOv2主要是要在这两方面做提升.另外YOLOv2并不是通过加深或加宽网络达到效果提升，反而是简化了网络。
## 批标准化
YOLOv2网络在每一个卷积层后添加批标准化（batch normalization），通过这一方法，mAP获得了2%的提升。
## 高分辨率分类器
ImageNet分类模型基本采用大小为224x224的图片作为输入，分辨率相对较低，不利于检测模型。所以YOLOv1在采用224x224分类模型预训练后，将分辨率增加至448x448，并使用这个高分辨率在检测数据集上finetune。但是直接切换分辨率，检测模型可能难以快速适应高分辨率。所以YOLOv2增加了在ImageNet数据集上使用448x448输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。
## 引入Anchor
YOLO采用全连接层来直接预测bounding boxes，而Fast R-CNN采用人工选择的bounding boxes。Fast  R-CNN中的 region proposal network仅采用卷积层来预测固定的boxes（anchor boxes）的偏移和置信度。
YOLOv2中作者去除了YOLO的全连接层，也采用anchor boxes来预测bounding boxes。首先，去除了一个pooling层来提高卷积层输出分辨率。然后，修改网络输入尺寸：由448×448改为416，使特征图只有一个中心。物品（特别是大的物品）更有可能出现在图像中心。YOLO的卷积层下采样率为32，因此输入尺寸变为416,输出尺寸为13×13。
对于YOLOv1，每个cell都预测2个boxes，每个boxes包含5个值（x,y,w,h,c）  ，前4个值是边界框位置与大小，最后一个值是置信度（confidence scores）。包含两部分：含有物体的概率以及预测框与ground truth的IOU）。但是每个cell只预测一套分类概率值（class predictions，其实是置信度下的条件概率值）,供2个boxes共享。YOLOv2使用了anchor boxes之后，每个位置的各个anchor box都单独预测一套分类概率值，这和SSD比较类似（但SSD没有预测置信度，而是把background作为一个类别来处理）。
采用anchor  boxes，提升了精确度。YOLO每张图片预测98个boxes，但是采用anchor boxes，每张图片可以预测超过1000个boxes。YOLO模型精确度为69.5mAP，recall为81%；采用anchor boxes方法后，结果为69.2mAP，recall为88%。
## K-means聚类寻找更好的anchor boxes尺寸
在Faster R-CNN和SSD中，anchor boxes的维度（长和宽）都是手动设定的，带有一定的主观性。如果选取的anchor boxes维度比较合适，那么模型更容易学习，从而做出更好的预测。因此，YOLOv2采用k-means聚类方法对训练集中的边界框做了聚类分析。因为设置anchor boxes的主要目的是为了使得预测框与ground truth的IOU更好，所以聚类分析时选用box与聚类中心box之间的IOU值作为距离指标： 
$$
d(\text {box}, \text { centroid })=1-I O U(\text {box}, \text { centroid })
$$
论文中，根据在VOC和COCO数据集上的聚类分析结果，随着聚类中心数目的增加，平均IOU值（各个边界框与聚类中心的IOU的平均值）是增加的，但是综合考虑模型复杂度和召回率，作者最终选取5个聚类中心作为先验框，5个先验框的width和height如下所示（这是针对VOC和COCO数据集的）：
COCO: （0.57273, 0.677385），（1.87446, 2.06253），（3.33843, 5.47434），（7.88282, 3.52778），（9.77052, 9.16828）
VOC: （1.3221, 1.73145），（3.19275, 4.00944），（5.05587, 8.09892），（9.47112, 4.84053），（11.2364, 10.0071）
上面的数字应该是相对于预测的特征图大小13x13。
和使用手工挑选的anchor boxes相比，使用K-means得到的anchor boxes表现更好。使用5个k-means得到的anchor boxes的性能（IOU 61.0）就和Faster R-CNN使用9个手工挑选的anchor boxes的性能（IOU 60.9）相当。这意味着使用k-means获取anchor boxes来预测bounding boxes让模型更容易学习如何预测bounding boxes。 
## 直接位置预测
YOLOv2借鉴Faster RCNN的RPN网络使用anchor boxes来预测边界框相对anchor boxes的偏置。边界框的实际中心位置（x,y） ，需要根据预测的坐标偏移值（tx,ty），anchor boxes的尺度（wa，ha）以及中心坐标（xa，ya）（特征图每个位置的中心点）来计算:
$$
x=\left(t_{x} \times w_{a}\right)-x_{a}
$$
$$
y=\left(t_{y} \times h_{a}\right)-y_{a}
$$
但是上面的公式是无约束的，预测的边界框很容易向任何方向偏移，如当tx=1时边界框将向右偏移anchor boxes一个宽度大小，而当tx=-1时边界框将向左偏移anchor boxes一个宽度大小，因此每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的偏置。
因此，YOLOv2弃用了这种预测方式，而是沿用YOLOv1的方法，就是预测边界框中心点相对于对应cell左上角位置的相对偏移值，为了将边界框中心点约束在当前cell中，使用sigmoid函数处理偏移值，这样预测的偏移值在（0,1）范围内（每个cell的尺度看做1）。根据边界框预测的4个偏置，可以按如下公式计算出边界框实际位置和大小:
$$
b_{x}=\sigma\left(t_{x}\right)+c_{x}
$$
$$
b_{y}=\sigma\left(t_{y}\right)+c_{y}
$$
$$
b_{w}=p_{w} e^{t_{w}}
$$
$$
b_{h}=p_{h} e^{t_{h}}
$$
其中（cx，cy）为cell的左上角坐标，计算时每个cell的尺度为1，当前cell的左上角坐标为（1，1）。由于sigmoid函数的处理，边界框的中心位置会约束在当前cell内部，防止偏移过多。pw和ph是anchor boxes的宽度与长度，它们的值也是相对于特征图大小的，在特征图中每个cell的长和宽均为1。特征图大小为（W,H）（在文中是13x13)，这样我们可以将边界框相对于整张图片的位置和大小计算出来（4个值均在0和1之间）:
$$
b_{x}=\left(\sigma\left(t_{x}\right)+c_{x}\right) / W
$$
$$
b_{y}=\left(\sigma\left(t_{y}\right)+c_{y}\right) / H
$$
$$
b_{w}=p_{w} e^{t_{w}} / W
$$
$$
b_{h}=p_{h} e^{t_{h}} / H
$$
上面4个值分别乘以图片的宽度和长度（像素点值）就可以得到边界框的最终位置和大小。定位预测值被归一化后，参数就更容易得到学习，模型就更稳定。作者使用Dimension Clusters和Direct location prediction这两项anchor boxes改进方法，mAP获得了5%的提升。 
## 新前置网络:Darknet-19
YOLOv2采用了一个新的基础模型（特征提取器），称为Darknet-19，包括19个卷积层和5个maxpooling层。Darknet-19与VGG16模型设计原则是一致的，主要采用3x3卷积，采用2x2的maxpooling层之后，特征图维度降低两倍，而同时将特征图的channles增加两倍。
Darknet-19最终采用global avgpooling做预测，并且在3x3卷积之间使用1x1卷积来压缩特征图channles以降低模型计算量和参数。Darknet-19每个卷积层后面同样使用了batch norm层以加快收敛速度，降低模型过拟合。在ImageNet分类数据集上，Darknet-19的top-1准确度为72.9%，top-5准确度为91.2%，但是模型参数相对小一些。使用Darknet-19之后，YOLOv2的mAP值没有显著提升，但是计算量却可以减少约33%。
## 细粒度特征
YOLOv1是对13×13的feature map进行目标检测。这足以胜任大尺度物体的检测，但是更细粒度的特征可以帮助模型定位较小的目标。
YOLOv2提出了一种passthrough层来利用更精细的特征图。YOLOv2所利用的细粒度特征是26x26x512大小的特征图（Darknet-19最后一个maxpooling层的输入）。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个2x2的局部区域，然后将其转化为channel维度，对于26x26x512的特征图，经passthrough层处理之后就变成了13x13x2048的新特征图（特征图长宽尺寸各降低一半，channles增加4倍），这样就可以与后面的13x13x1024特征图连接在一起形成13x13x3072大小的特征图，然后在此特征图基础上卷积做预测。
另外，作者在后期的实现中借鉴了ResNet网络，不是直接对高分辨特征图处理，而是增加了一个中间卷积层，先采用64个1x1卷积核进行卷积，然后再进行passthrough处理，这样26x26x512的特征图得到13x13x256的特征图。使用细粒度特征之后YOLOv2的性能有1%的提升。
## 多尺度训练
由于YOLOv2模型中只有卷积层和池化层，所以YOLOv2的输入可以不限于  大小的图片。为了增强模型鲁棒性，YOLOv2采用了多尺度输入训练策略，具体来说就是在训练过程中每间隔一定的iterations之后改变模型的输入图片大小。由于YOLOv2的下采样总步长为32，输入图片大小选择一系列为32倍数的值（320,352,...608，32的倍数，因为模型下采样因子为32）。在训练过程，每隔10个iterations随机选择一种输入图片大小，然后只需要修改对最后检测层的处理就可以重新训练。
在小尺寸图片检测中，YOLOv2成绩很好，输入为228x228的时候，帧率达到90FPS，mAP几乎和Faster R-CNN的水准相同。使得其在低性能GPU、高帧率视频、多路视频场景中更加适用。
在大尺寸图片检测中，YOLOv2达到了先进水平，VOC2007上mAP为78.6%，仍然高于平均水准。
## 分类模型训练
作者使用Darknet-19在标准的ImageNet1000类分类数据集上训练了160个迭代，用的随机梯度下降算法，初始学习率为0.1，polynomial rate decay为4，weight decay为0.0005 ，momentum为0.9。训练时用了很多常见的数据扩充方法（data augmentation），包括random crops, rotations, and hue, saturation, and exposure shifts。 
初始训练时网络的输入是224x224，160个迭代后输入的分辨率切换到448x448进行fine tune，fine tune时学习率调整为0.001，训练10个迭代。最终分类网络在ImageNet上top-1准确率76.5%，top-5准确率93.3%。
## 检测模型训练
分类网络训练完后，就该训练检测网络了，作者去掉了原网络最后一个卷积层，转而增加了三个3x3x1024的卷积层（可参考darknet中cfg文件），并且在每一个上述卷积层后面跟一个1x1的卷积层，输出维度是检测所需的数量。对于VOC数据集，预测5种boxes大小，每个box包含5个坐标值和20个类别，所以总共是5x（5+20）=125个输出维度。同时也添加了转移层（passthrough layer ），从最后那个3x3x512的卷积层连到倒数第二层，使模型有了细粒度特征。
作者的检测模型以0.001的初始学习率训练了160次，在60次和90次的时候，学习率减为原来的十分之一。其他的方面，weight decay为0.0005，momentum为0.9，依然使用了类似于Faster-RCNN和SSD的数据扩充（data augmentation）策略。
## stronger:YOLO9000
YOLO9000是在YOLOv2的基础上提出的一种可以检测超过9000个类别的模型，其主要贡献点在于提出了一种分类和检测的联合训练策略。由于检测数据集的标注要比分类数据集打标签繁琐的多，所以ImageNet分类数据集比VOC等检测数据集高出几个数量级。
如何做到9000个类别分类呢？一方面要构造数据集，另一方面要解决模型训练问题，前者采用WordTree解决，后者采用Joint classification and detection。
在YOLO中，边界框的预测其实并不依赖于物体的标签，所以YOLO可以实现在分类和检测数据集上的联合训练。对于检测数据集，可以用来学习预测物体的边界框、置信度以及为物体分类，而对于分类数据集可以仅用来学习分类，但是其可以大大扩充模型所能检测的物体种类。
**如何做数据集的融合？**
作者提出了新的思路：通过ImageNet训练分类，COCO和VOC数据集来训练检测。 通过将两个数据集混合训练，如果遇到来自分类集的图片则只计算分类的Loss，遇到来自检测集的图片则计算完整的Loss。这里面是有问题的，ImageNet对应分类有9000种，而COCO则只提供80种目标检测，这中间如何Match？答案就是multi-label模型，即假定一张图片可以有多个label，并且不要求label间独立。由于ImageNet的类别是从WordNet选取的，作者采用以下策略重建了一个树形结构（称为分层树）：遍历Imagenet的label，然后在WordNet中寻找该label到根节点(指向一个物理对象)的路径；如果路径只有一条，那么就将该路径直接加入到分层树结构中；否则，从剩余的路径中选择一条最短路径，加入到分层树。这个分层树我们称之为 Word Tree，作用就在于将两种数据集按照层级进行结合。
分类时的概率计算借用了决策树思想，某个节点的概率值等于该节点到根节点的所有条件概率之积。
**如何给模型训练？**
文中采用的是Joint classification and detection（联合分类和检测），即在训练期间，我们混合来自检测和分类数据集的图像。 当我们的网络看到标记为检测的图像时，我们可以基于完整的YOLOv2丢失功能进行反向传播。 当它看到分类图像时，我们只反向传播体系结构的分类特定部分的损失。
也就是说，通过这种联合训练，YOLO9000学习使用COCO中的检测数据在图像中查找对象，并学习使用来自ImageNet的数据对各种对象进行分类。YOLO9000的主网络基本和YOLOv2类似，只不过每个grid cell只采用3个box prior。
# YOLOv3
论文:YOLOv3: An Incremental Improvement
论文地址:https://pjreddie.com/media/files/papers/YOLOv3.pdf 。
## 多标签任务
在图片中，一个锚点的感受野肯定会有包含两个甚至更多个不同物体的可能，在之前的方法中是选择和锚点IoU最大的Ground Truth作为匹配类别，用softmax作为激活函数。
YOLOv3多标签模型的提出，对于解决覆盖率高的图像的检测问题效果是十分显著的。YOLOv3模型不仅检测的更精确，最重要的是被覆盖了很多的物体也能很好的在YOLOv3中检测出来。
YOLOv3提供的解决方案是将一个N维的softmax分类器替换成N个sigmoid分类器，这样每个类的输出仍是（0,1）之间的一个值，但是他们的和不再是1。虽然YOLOv3改变了输出层的激活函数，但是其锚点和Ground Truth的匹配方法仍旧采用的是YOLOv1的方法，即每个Ground Truth匹配且只匹配唯一一个与其IoU最大的锚点。但是在输出的时候由于各类的概率之和不再是1，只要置信度大于阈值，该锚点便被作为检测框输出。
## 新骨干网络:Darknet-53
YOLOv3采用了称之为Darknet-53的网络结构（含有53个卷积层），它借鉴了残差网络（residual network）的做法，在一些层之间设置了快捷链路（shortcut connections）。
## Anchor聚类
YOLO2已经开始采用K-means聚类得到anchor boxes的尺寸，作者尝试了折中考虑了速度和精度之后选择的类别数k=5。YOLO3延续了这种方法，为每种下采样尺度设定3种anchor boxes，总共聚类出9种尺寸的anchor boxes。在COCO数据集这9个先验框是：（10x13），（16x30），（33x23），（30x61），（62x45），（59x119），（116x90），（156x198），（373x326）。
分配上，在最小的13x13特征图上（有最大的感受野）应用较大的先验框（116x90），（156x198），（373x326），适合检测较大的对象；中等的26x26特征图上（中等感受野）应用中等的先验框（30x61），（62x45），（59x119），适合检测中等大小的对象；较大的52x52特征图上（较小的感受野）应用较小的先验框（10x13），（16x30），（33x23），适合检测较小的对象。
## 多尺度预测
YOLOv3采用多个尺度融合的方式做预测。原来的YOLO v2有一个转移层（ passthrough layer），假设最后提取的feature map的size是13x13，那么这个层的作用就是将前面一层的26x26的feature map和本层的13x13的feature map进行连接，有点像ResNet。这样的操作也是为了加强YOLO算法对小目标检测的精确度。
这个思想在YOLO v3中得到了进一步加强，在YOLO v3中采用类似FPN的上采样（upsample）和融合做法（最后融合了3个scale，其他两个scale的大小分别是26x26和52x52），在多个scale的feature map上做检测，对于小目标的检测效果提升还是比较明显的。
**注意:**
YOLOv2采用的是降采样的形式进行Feature Map的拼接，YOLOv3则是采用同SSD相同的双线性插值的上采样方法拼接的Feature Map；
每个尺度的Feature Map负责对3个anchor boxes的预测。