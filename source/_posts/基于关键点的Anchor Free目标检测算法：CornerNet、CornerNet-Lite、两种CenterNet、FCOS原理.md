---
title: 基于关键点的Anchor Free目标检测算法：CornerNet、CornerNet-Lite、两种CenterNet、FCOS原理
date: 2019-05-31 18:25:45
tags:
- 目标检测
categories:
- 目标检测
mathjax: true
---

# 基于关键点的Anchor Free目标检测算法
2018到2019年间，出现了许多基于关键点的one stage目标检测算法。这类算法的特点是不使用Anchor  boxes作为先验框，所以又叫做Anchor-Free目标检测算法。
**本文主要介绍五种有代表性的Anchor-Free目标检测算法:**
CornerNet:使用左上角和右下角的两个角点来表示一个目标；
CornerNet-Lite:CornetNet-Lite是对CornetNet进行优化，具体分为两种算法CornerNet-Saccade（高准确率优先）和CornerNet-Squeeze（高实时性优先）；
CenterNet:Keypoint Triplets for Object Detection:使用中心点、左上角点和右下角点三个关键点来表示一个目标；
CenterNet:Objects as Points:用一个中心点+长宽值来表示一个目标；
FCOS:训练集中目标用左上角点和右下角点表示，特征图每个点映射回原图得到原图中的一个点，使用该点+点到框的四个距离来表示一个目标。

注意上面有两个CenterNet，但是它们是两种Anchor-Free目标检测模型，请注意区分。
# CornerNet
论文:CornerNet: Detecting Objects as Paired Keypoints
论文地址:https://arxiv.org/pdf/1808.01244.pdf 。
代码地址:https://github.com/umich-vl/CornerNet 。
这篇文章中提出了一种新的目标检测方法CornerNet，使用单个卷积神经网络将目标边界框检测为一对关键点（即边界框的左上角和右下角）。通过将目标检测为成对关键点，我们消除了现有的one stage检测器设计中对一组anchors的需要。除此之外文章还引入了corner pooling，这是一种新型的池化层，可以帮助网络更好地定位边界框的角点。CornerNet在MS COCO上实现了42.1％的AP，优于所有现有的one stage检测器。
以往的YOLOv3、SSD等one stage检测器使用anchor boxes作为先验框，检测器将anchor boxes密集地分布在图像上，通过对anchor boxes进行评分，并通过回归来改进其坐标来生成最终的边界框预测。
**但anchor boxes的使用有两个缺点:**
* 首先，我们通常需要非常多的anchor boxes以确保与大多数ground truth充分重叠。 但实际只有一小部分anchor boxes与ground truth重叠，这在正负样本之间造成了巨大的不平衡，同时也减慢了训练速度；
* 其次，anchor boxes的使用引入了许多超参数和设计选择。 这些包括多少个box，大小和宽高比。 这些选择主要是通过ad-hoc启发式方法进行的，并且当与多尺度架构相结合时可能会变得更加复杂，其中单个网络在多个分辨率下进行单独预测，每个尺度使用不同的特征和它自己的一组anchor boxes。

CornerNet不使用anchor boxes，而将一个目标的位置检测化为检测边界框的左上角和右下角这对关键点的问题。我们使用单个卷积网络来预测同一物体类别的所有实例的左上角的热图，所有右下角的热图，以及每个检测到的角点的嵌入向量。 嵌入用于对属于同一目标的一对角点进行分组——训练网络以预测它们的类似嵌入。 这种方法极大地简化了网络的输出，并且无需设计anchor boxes。
**作者认为基于关键点的目标检测要优于anchor的检测方法主要有两方面的原因:**
由于框的中心依赖于四个边，很难进行定位。而定位角点只需要定位两条边，同时引入了coner pool的先验，因此，定位更加简单；角点高效的离散了框的解空间，只需要O（wh）的角点可以表示O（w2h2）的anchor box的数量。
## CornerNet网络结构
首先1个7×7的卷积层将输入图像尺寸缩小为原来的1/4（论文中输入图像大小是511×511，缩小后得到128×128大小的输出）。
然后通过Hourglass Network网络进行特征提取，该网络通过串联2个hourglass module组成。每个hourglass module都是先通过一系列的降采样操作缩小输入的大小，然后通过上采样恢复到输入图像大小，因此该部分的输出特征图大小还是128×128，整个hourglass network的深度是104层。
然后将网络得到的特征输入到两个模块Top-left Corner pooling和Bottom-right Corner pooling提取关键点的特征。对于每个Corner Pooling模块，后面接一个预测模块，包括三个部分：目标框的左上角关键点和右下角关键点的类别分类（Heatmaps），每个目标的一对关键点（Embeddings），以及基于坐标回算目标目标位置时的偏置（offsets）。有了两个角点的heatmaps，embeding vectors，及Offset，我们在后面可以通过后处理的方式得到最终的边框。
heatmaps是输出预测角点信息，可以用维度为CHW的特征图表示，其中C表示目标的类别（注意没有背景类），这个特征图的每个通道都是一个mask，mask的每个值范围为0到1，表示该点是角点的分数；embeddings用来对预测的corner点做group，也就是找到属于同一个目标的左上角角点和右下角角点；offsets用来对预测框做微调，这是因为从输入图像中的点映射到特征图时有量化误差，offsets就是用来输出这些误差信息。
```
                                                          -->Heatmaps
									                      |
                               ->Top-left Corner pooling-->->Embeddings
                              /                           |
                             /                            -->Offsets
						    /
输入图片->Hourglass Network-->
							\
							 \                                -->Heatmaps
							  \                               |
							   ->Bottom-right Corner pooling-->->Embeddings
							                                  |
							                                  -->Offsets
```
CornerNet使用Hourglass Network作为CornerNet的骨干网络。 Hourglass Network之后是两个预测模块。 一个模块预测左上角角点，另一个模块预测右下角角点，每个模块都有自己的corner pooling模块，在预测热图、嵌入和偏移之前，池化来自沙漏网络的特征。corner pooling后包括三部分：目标框的左上角关键点和右下角关键点的类别分类（Heatmaps），每个目标的一对关键点（Embeddings），以及基于坐标回算目标目标位置时的偏置（offsets）。 与许多其他探测器不同，我们不使用不同尺度的特征来检测不同大小的物体。 Heatmaps表示不同物体类别的角的位置，一组为左上角角点，另一组为右下角角点。 Embeddings预测每个检测到的角的嵌入向量，使得来自同一目标的两个角的嵌入之间的距离很小。 为了产生更紧密的边界框，网络还预测offsets以稍微调整角的位置。 通过预测的热图（ Heatmaps），嵌入（Embeddings）和偏移（offsets），我们应用一个简单的后处理算法来获得最终的边界框。
## Heatmaps
两个corner pooling模块后各接了一个预测模块。预测模块中包含Heatmaps、Embeddings、offsets三个模块。通过Heatmaps模块，我们预测两组热图，一组用于预测左上角角点，另一组用于预测右下角角点。 每组热图具有C个通道，其中C是分类的类别数，并且大小为H×W。注意没有背景这个类别。这个特征图的每个通道都是一个mask，mask的每个值为0-1之间的值，表示该点是角点的分数。
对于每个角点，有一个ground-truth正位置，其他所有的位置都是负值。 在训练期间，我们没有同等地惩罚负位置，而是减少对正位置半径内的负位置给予的惩罚。 这是因为如果一对假角点检测器靠近它们各自的ground-truth位置，它仍然可以产生一个与ground-truth充分重叠的边界框。我们通过确保半径内的一对点生成的边界框与ground-truth的IoU ≥ t（实验中t=0.7）来确定物体的大小，从而确定半径。给定半径，惩罚的减少量由非标准化的2D高斯函数产生:
$$
e^{-\frac{x^{2}+y^{2}}{2 \sigma^{2}}}
$$
其中心即位置，其σ是半径的1/3。

**Heatmaps的损失函数:**
$$
L_{d e t}=\frac{-1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} \begin{cases}{\left(1-p_{c i j}\right)^{\alpha} \log \left(p_{c i j}\right)} & {\text { if } y_{c i j}=1} \\\\ {\left(1-y_{c i j}\right)^{\beta}\left(p_{c i j}\right)^{\alpha} \log \left(1-p_{c i j}\right)} & {\text { otherwise }}\end{cases}
$$
上式整体上是改良版的focal loss。pcij表示预测的heatmaps在第c个通道（类别c）的（i,j）位置的值，ycij为用非标准化高斯增强的“ground-truth”热图。N表示目标的数量。利用ycij中编码的高斯凸点，（1−ycij）项减少了ground-truth周围的惩罚。
ycij=1时候的损失函数容易理解，就是focal loss，α参数用来控制难易分类样本的损失权重；ycij等于其他值时表示（i,j）点不是类别c的目标角点，照理说此时ycij应该是0（大部分算法都是这样处理的），但是这里ycij不是0，而是用基于ground truth角点的高斯分布计算得到，因此距离ground truth比较近的（i,j）点的ycij值接近1，这部分通过β参数控制权重，这是和focal loss的差别。
**为什么对不同的负样本点用不同权重的损失函数呢？**
这是因为靠近ground truth的误检角点组成的预测框仍会和ground truth有较大的重叠面积，仍能基本框住目标，因此仍然是有效的预测框。
## Offsets
这个值和目标检测算法中预测的offset完全不一样，在目标检测算法中预测的offset是表示预测框和anchor之间的偏置，而这里的offset是表示在取整计算时丢失的精度信息。即下面的公式:
$$
o_{k}=\left(\frac{x_{k}}{n}-\left\lfloor\frac{x_{k}}{n}\right\rfloor, \frac{y_{k}}{n}-\left\lfloor\frac{y_{k}}{n}\right\rfloor\right)
$$
从输入图像到特征图之间尺寸会缩小，假设缩小倍数是n，那么输入图像上的（x,y）点对应到特征图上就如下式：
$$
\left(\left\lfloor\frac{x}{n}\right\rfloor,\left\lfloor\frac{y}{n}\right\rfloor\right)
$$
式中的符号是向下取整，取整会带来精度丢失，这尤其影响小尺寸目标的回归，Faster RCNN中的 ROI Pooling也是有类似的精度丢失问题。我们通过上面计算ok的公式计算offset，然后通过下面的smooth L1损失函数监督学习该参数（使用L1损失与原图标注的位置对其进行修正），和常见的目标检测算法中的回归支路类似。
下式中ok是偏移量，xk和yk是角点k的x和y坐标。我们预测所有类别的左上角共享一组偏移，另一组由右下角共享。 对于训练，我们在ground-truth角点位置应用平滑的L1损失。
$$
L_{o f f}=\frac{1}{N} \sum_{k=1}^{N} SmoothL1Loss\left(o_{k}, \hat o_{k}\right)
$$
## Embeddings
一张图中可能会存在多个目标，因此，可能会检测到多组角点。这里使用embedding模块来确定一组左上角及右下角的点是否是来自同一个目标的边界框。通过基于左上角点的embeding vectors及右下角点embeding vectors的距离来决定是否将两个点进行组合。重要的是二者之间的距离，而向量中的具体数值却不是很重要。本文使用1-D的embeding 向量，etk代表目标k左上角点的embeding ,ebk代表其右下角点的embeding。定义"pull"损失用于组合角点，“push”损失用于分离角点。
**Embeddings的损失函数:**
$$
L_{p u l l}=\frac{1}{N} \sum_{k=1}^{N}\left[\left(e_{t_{k}}-e_{k}\right)^{2}+\left(e_{b_{k}}-e_{k}\right)^{2}\right]
$$
$$
L_{p u s h}=\frac{1}{N(N-1)} \sum_{k=1}^{N} \sum_{j=1 \atop j \neq k}^{N} \max \left(0, \Delta-\left|e_{k}-e_{j}\right|\right)
$$
etk表示属于k类目标的左上角角点的embedding vector，ebk表示属于k类目标的右下角关键点的embedding vector，ek表示etk和ebk的均值。我们在所有实验中将Δ设为1。与偏移损失类似，我们仅在ground-truth角点位置应用损失。
第一个公式用来缩小属于同一个目标（k类目标）的两个关键点的embedding vector（etk和ebk）距离。第二个公式用来扩大不属于同一个目标的两个角点的embedding vector距离。
## Corner Pooling
CornerNet使用了两个Corner Pooling模块，分别是Top-left Corner pooling和Bottom-right Corner pooling，分别预测左上角关键点和右下角关键点。
CornerNet要预测左上角和右下角两个角点，但是这两个角点在不同目标上没有相同规律可循，如果采用普通池化操作，那么在训练预测角点支路时会比较困难。考虑到左上角角点的右边有目标顶端的特征信息（第一张图的头顶），左上角角点的下边有目标左侧的特征信息（第一张图的手），因此如果左上角角点经过池化操作后能有这两个信息，那么就有利于该点的预测，这就有了corner pooling。
每个corner pooling模块有2个输入特征图，特征图的宽高分别用W和H表示，假设接下来要对特征图上（i,j）点做左上角的corner pooling，那么就计算（i,j）到（i,H）的最大值（最大池化）；同时计算（i,j）到（W,j）的最大值（最大池化），然后将这两个最大值相加（就是普通的加法）得到（i,j）点的值。右下角点的corner pooling操作类似，只不过计算最大值变成从（0,j）到（i,j）和从（i,0）到（i,j）。
用公式表示即为：
$$
t_{i j}=\begin{cases}{\max \left(f_{t_{ij}}, t_{(i+1) j}\right)} & {\text { if } i<H} \\\\ {f_{t_{H j}}} & {\text { otherwise }}\end{cases}
$$
$$
l_{i j}=\begin{cases} \max \left(f_{l_{i j}}, l_{i(j+1)}\right) & \text { if } j<W \\\\ f_{l i w} & \text { otherwise } \end{cases}
$$
**Corner Pooling计算举例:**
假如做Top-left Corner pooling:
```
-  -  -  -  -          -  -  -  -  -
2  1  3  0  2          3  3  3  2  2
5  4  1  1  6  ------->6  6  6  6  6
-  -  -  -  -          -  -  -  -  -
-  -  -  -  -          -  -  -  -  -

-  3  1  -  -          -  3  4  -  -
-  1  1  -  -          -  3  4  -  -
-  3  4  -  -  ------->-  3  4  -  - 
-  2  2  -  -          -  2  2  -  -
-  0  2  -  -          -  0  2  -  -
最后将两个最大值矩阵同样位置的元素相加，得:
-  -  -  -  -
-  6  7  -  -
-  9  10 -  -
-  -  -  -  -
-  -  -  -  -
```
## Hourglass Network
CornerNet使用沙漏网络（Hourglass Network）作为其骨干网络。沙漏网络首次被提到是用于人体姿态估计任务。它是一个完全卷积神经网络，由一个或多个Hourglass组成。Hourglass首先通过一系列卷积层和最大池化层对输入特性进行下采样。然后通过一系列的上采样和卷积层将特征上采样回原来的分辨率。由于细节在最大池化层中丢失，因此添加了跳过层用来将细节带回到上采样的特征。沙漏模块在一个统一的结构中捕获全局和局部特征。当多个Hourglass堆积在网络中时，Hourglass可以重新处理特征以获取更高级别的信息。这些特性使沙漏网络成为目标检测的理想选择。事实上，许多现有的检测器已经采用了类似沙漏网络的网络。
我们的沙漏网络由两个Hourglass组成，我们对Hourglass的结构做了一些修改。我们不使用最大池化，而是使用步长2来降低特征分辨率。我们减少了5倍的特征分辨率，并增加了特征通道的数量（256,384,384,384,512）。当我们对特征进行上采样时，我们应用了两个残差模块，然后是一个最近的相邻上采样。每个跳跃连接还包含两个残差模块。沙漏模块中间有4个512通道的残差模块。在沙漏模块之前，我们使用128个通道7×7的卷积模块，步长为2，4倍减少的图像分辨率，后跟一个256个通道，步长为2的残差块。
在沙漏网络基础上，我们还在训练时增加了中间监督。但是，我们没有向网络中添加反向中间预测，因为我们发现这会损害网络的性能。我们在第一个沙漏模块的输入和输出，应用了一个3×3的Conv-BN模块。然后，我们通过元素级的加法合并它们，后跟一个ReLU和一个具有256个通道的残差块，然后将其用作第二个沙漏模块的输入。沙漏网络的深度为104。与许多其他最先进的检测器不同，我们只使用整个网络最后一层的特征来进行预测。
## 训练细节
我们在PyTorch中实现了CornerNet。网络是在默认的PyTorch设置下随机初始化的，没有在任何外部数据集上进行预训练。在训练期间，我们设置了网络的输入分辨率511×511，输出分辨率为128×128。为了减少过拟合，我们采用了标准的数据增强技术，包括随机水平翻转、随机缩放、随机裁剪和随机色彩抖动，其中包括调整图像的亮度，饱和度和对比度。 最后，我们将PCA应用于输入图像。
**完整的损失函数:**
$$
L=L_{d e t}+\alpha L_{p u l l}+\beta L_{p u s h}+\gamma L_{o f f}
$$
其中α，β和γ分别是pull，push和offset的权重。 我们将α和β都设置为0.1，将γ设置为1。我们发现，1或更大的α和β值会导致性能不佳。 我们使用49的batch size，并在10个Titan X（PASCAL）GPU上训练网络（主GPU4个图像，其余GPU每个GPU5个图像）。 为了节省GPU资源，在我们的ablation experiments（即模型简化测试，去掉该结构的网络与加上该结构的网络所得到的结果进行对比）中，我们训练网络，进行250k次迭代，学习率为2.5×10−4。当我们将我们的结果与其他检测器进行比较时，我们额外训练网络，进行250k次迭代，并到最后50k次迭代时，将学习速率降低至2.5×10−5。
## 测试细节
在测试期间，我们使用简单的后处理算法从热图，嵌入和偏移生成边界框。 我们首先通过在角点热图上使用3×3最大池化层来应用非极大值抑制（NMS）。然后我们从热图中选择前100个左上角和前100个右下角。 角点位置由相应的偏移调整。 我们计算左上角和右下角嵌入之间的L1距离。距离大于0.5或包含不同类别的角点对将被剔除。 左上角和右下角的平均得分用作检测分数。
我们不是将图像大小调整为固定大小，而是保持图像的原始分辨率，并在将其输入给CornerNet之前用0填充。 原始图像和翻转图像都用于测试。 我们将原始图像和翻转图像的检测结合起来，并应用soft-max来抑制冗余检测。 仅记录前100个检测项。 Titan X（PASCAL）GPU上的每个图像的平均检测时间为244ms。
# CornerNet-Lite
论文:CornerNet-Lite: Efficient Keypoint Based Object Detection
论文地址:https://arxiv.org/pdf/1904.08900.pdf 。
代码地址:https://github.com/princeton-vl/CornerNet-Lite 。
CornetNet-Lite是对CornetNet的优化，文章中提出了CornerNet的两种改进算法：CornerNet-Saccade和CornerNet-Squeeze。
CornerNet-Saccade引入了Saccade思想，在追求高准确率（mAP）的同时，尽可能提高速度（FPS），即准确率优先，其对标于CornerNet等算法。CornerNet-Squeeze引入SqueezeNet优化思想，在追求高实时性（FPS）的同时，尽可能提高准确率（mAP），即速度优先，其对标于YOLOv3等算法。
CornerNet-Saccade可以用于线下处理，将 CornerNet 的效率提升6倍，将COCO的效率提高1.0％。CornerNet-Squeeze适合用于实时检测，比YOLOv3的效率和准确性更高（CornerNet-Squeenze的AP值是34.4%，速度是34ms，而YOLOv3的AP值是33.0%，速度是39ms）。
CornerNet-Saccade通过减少像素的个数来加速推理的速度。它使用了类似于人眼的注意力机制。首先缩小一幅图像，产生一个attention map，然后再通过模型进一步放大和处理。原始的CornerNet在多个尺度上进行全卷积操作，这和CornerNet-Saccade不同。CornerNet-Saccade 选取若干个高分辨率的裁剪区域来检测，提升检测速度和精度。
CornerNet-Squeeze减少每个像素点上需要处理的步骤，以此来加速推理。它融合了SqueezeNet和 MobileNet的思想，加入了一个精炼的Hourglass主干网络，这个主干网络中大量使用了1×1的卷积，bottleneck层，以及深度可分离卷积。
**我们是否能将CornerNet-Squeeze和CornerNet-Saccade网络结合起来使用提升效率呢？**
文章的实验结果表明是不提倡的：CornerNet-Squeeze-Saccade速度和准确率都要比CornerNet-Squeeze差。这是因为，要用Saccade，网络要能产生足够准确的attention maps，但是我们的CornerNet-Squeeze过于精炼，不具备这样的能力。此外，原来的CornerNet 应用在多个不同的尺度上，提供了足够的空间来进行saccade，减少要处理的像素数量。相反，CornerNet-Squeeze由于推理成本的限制，只能用在单尺度上，供saccade操作的空间就很小。
## 意义和新颖性
上述两个CornerNet-Lite使得基于关键点的检测方法更具竞争力，覆盖了两个主要的使用场景：CornerNet-Saccade用于线下处理，无需牺牲准确率也可提升效率，CornerNet-Squeeze用于实时处理，提升准确率而无需牺牲效率。
这两个变体都很有创新。CornerNet-Saccade 是第一个在基于关键点的目标检测方法中使用saccade的方法。它和之前工作的关键区别就是，每个裁剪区域（像素点或特征图）的处理方式。其它使用了类似saccade机制的工作要么在每个裁剪区域内检测单个物体，要么用一个双阶段网络在每个裁剪区域内产生多个子裁剪区域，然后再产生若干个检测对象。而 CornerNet-Saccade 只使用单阶段网络在每个裁剪区域内输出多个检测对象。
CornerNet-Squeeze是第一篇将SqueezeNet和Hourglass结构整合起来用于目标检测任务的论文。之前用了Hourglass结构的方法优于获取高准确率，但是我们不清楚它是否/如何能获得高效率。我们的设计和结果显示，这也是可能的，尤其在目标检测的任务上。
## CornerNet-Saccade
人类视觉中的Saccades（扫视运动）是指用于固定不同图像区域的一系列快速眼动。在目标检测算法中，我们广义地使用该术语来表示在预测期间选择性地裁剪（crop）和处理图像区域（顺序地或并行地，像素或特征）。
在一张图片中，在物体可能出现的位置附近，CornerNet-Saccade在较小的区域范围内检测物体。它利用缩小后的图片来预测attention maps以及大概的边框，这俩都能提供物体可能的位置。然后CornerNet-Saccade在高分辨率的图像中，在位置中心附近区域内检测物体。它也可以通过调节每张图片上物体位置的最大个数来平衡精度和速度。
**估计目标位置:**
CornerNet-Saccade的第一步就是获取图像中可能的物体位置。我们使用缩小的图像预测attention maps，它能指出物体的位置以及物体大致的尺寸。给定输入图像，我们将图像缩小至长边为255像素或者192像素。192像素的图像在边缘位置填充0，使得它的大小达到255像素，这样它们就可以并行处理。我们使用这样的低分辨率图有2个原因。首先，这一步不应该成为前向推理时的瓶颈。其次，网络应该很容易就可以利用图像的全局信息预测attention maps。
对每一个缩小后的图片，CornerNet-Saccade预测3个 attention maps，一个针对小物体，一个针对中等物体，一个针对大物体。如果一个物体边框的长边小于32像素，它就被认为是小物体，超过96像素的就被认为是大物体，中间的就是中等物体。对不同大小的物体分开预测位置，让我们能更好地控制 CornerNet-Saccade在每个位置应该放大多少。我们在小物体位置可以放大的多一些，在大物体位置可以放大的少一些。
我们利用不同比例的特征图来预测attention maps。从CornerNet-Saccade的主干网络（Hourglass网络）中获取特征图。每一个Hourglass模块都使用了多个卷积和下采样层来降低输入特征图的大小。然后再将特征图通过多个上采样层和卷积层上采样至原始输入的分辨率。上采样层输出的特征图用于预测attention maps。比例较细致的特征图用于预测小物体，而比例较粗糙的用于预测大物体。我们在每个特征图上应用一个3×3卷积-ReLU 模块，后面跟着一个1×1Conv-Sigmoid模块，以此来预测attention maps。在测试时，我们仅处理得分高于阈值t的位置，在所有实验中t=0.3。
当 CornerNet-Saccade处理缩小后的图像，它就可能会检测到图像中的物体并生成边框。从缩小后的图像上获取的边框可能不那么准确。因此，我们需要在高分辨率的图像上再检测一次，来获得更准确的边框。
在训练时，我们将attention map上每个边框对应的中心位置设为正，其余都为负。然后我们再使用α=2的Focal Loss。
**目标的检测:**
CornerNet-Saccade使用缩小后图像上得到的位置来决定到底在哪个位置进行处理。如果我们直接在缩小后的图像上进行裁剪，有一些物体可能就会变得很小，使检测变得不准确。因此，我们应该在更高分辨率的图像上进行检测。
对于从attention maps得到的位置，可以针对不同尺寸的目标设置不同的放大尺寸。Ss代表小目标的缩放尺寸，Sm代表中等目标的缩放尺寸，Sl代表大目标的缩放尺寸。整体三者之间存在一种关系，Ss>Sm>sl，因为，我们需要对小目标进缩放的成都要大一些。本文设置如下,Ss=4,sm=2,sl=1。对于可能存在的位置（x,y），根据大致的目标尺寸，按照si的比例对downsized图片进行放大，然后，将CornerNet-Saccade应用到255x255窗口的中心位置处。
从预测的边界框中得到的位置包含更多目标物的尺寸信息。可以利用得到的边界框的尺寸来确定缩放大小。确定缩放比例后，使小目标的长边为24，中等目标的为64，大目标的为192。
为了让处理更加地高效，我们加了一些重要的实现细节。首先，我们以批次来处理所有区域，这样能更好地利用GPU。其次，我们将原始图片保存在GPU 显存里，在GPU里面进行缩放和裁剪，降低CPU 和 GPU 之间转移图像数据带来的消耗。
在可能的位置处检测到目标物后，基于soft-NMS处理对于的检测结果。在对图片进行裁剪时，裁剪区域的边界可能会包含目标物的部分区域，如下图所示。产生的边界框可能包含目标物很少的区域，而无法被soft-NMS处理掉，因此，删除了距离裁剪边界很近的边界框。训练时，采用与CornerNet相似的损失，用于预测corner heatmaps,embedings及offsets。
**准确率及效率的权衡:**
通过控制每张图片上物体位置的最大个数，我们可以平衡效率和准确率。为了平衡准确率和效率，我们优先处理那些更可能包含物体的位置。因此，当我们得到了物体的位置后，根据它们的得分将它们排序，然后优先处理那些从边框中得到的位置。给定要处理的最大裁剪个数kmax，我们在最高的kmax个位置上进行物体检测。
**抑制冗余对象位置:**
当物体之间距离很近时，我们可能会得到高度重叠的物体。我们并不想把这两个框都检测一遍，因为检测其中一个框时就可能会检测到另一个框中的物体。
我们采取了一个类似于NMS的方法，去除多余的位置。首先，我们将目标物体的位置进行排序，优先处理边框内的位置，然后再处理attention maps上的位置。我们保留最佳的物体位置，去除那些距离最佳位置过于近的位置。我们重复以上操作，直到没有物体位置剩余。
**骨干网络:**
我们设计了一个新的 Hourglass 主干网络，更适合用在CornerNet-Saccade。新的Hourglass网络由3个Hourglass模块组成，深度是54层，而原CornerNet中的Hourglass-104由2个Hourglass模块组成，深度是104层。我们称新的主干网络为Hourglass-54。
Hourglass-54中每一个 Hourglass 模块的参数量更少，也要更浅。按照Hourglass-104中的尺寸缩小策略，我们以步长2来缩小特征图。我们在每个下采样层后面使用了一个残差模块。每个Hourglass 模块降低输入特征尺寸3倍，增加它的通道数（384,384,512 384, 384, 512384,384,512）。在模块的中间位置有一个512通道的残差模块，在每个上采样层后面有一个残差模块。我们同样在Hourglass模块之前降低图像的尺寸2倍。
**训练细节:**
我们使用Adam方法来优化attention maps和目标检测的损失函数，并且使用和CornerNet中一样的训练超参数。网络的输入大小是255×255，这也是测试时的输入大小。我们在4张1080ti GPU上训练，batch size是48。为了避免过拟合，我们使用了CornerNet中的数据增强。当我们要在目标物体附近随机裁剪一个区域，目标物体要么是随机选取的，要么是在中心位置的附近。这确保了训练和测试是一致的，因为网络检测物体的范围是在一个以物体位置为中心的裁剪区域内。
## CornerNet-Squeeze
CornerNet-Saccade是减少要处理像素的数量，而CornerNet-Squeeze则是研究了一个替换方案，降低每个像素点上要处理的成本。在CornerNet中，绝大多数的计算资源都耗费在Hourglass-104上。Hourglass-104由残差模块组成，每个残差模块由2 个3×3的卷积层外加一个skip连接构成。尽管Hourglass-104获得了不错的效果，但是就它的参数个数和推理时间来说，它是非常耗时的。为了降低Hourglass-104的复杂度，我们引入了SqueezeNet和MobileNet中的思想，设计了一个轻量级Hourglass架构。
**来自SqueezeNet和MobileNet中的想法:**
SqueezeNet提出了3个降低网络复杂度的策略：
* 将3×3的卷积核替换为1×1的卷积核；
* 降低输入通道为3×3卷积核；
* 晚点进行下采样；

SqueezeNet中的构建模块fire module，首先通过由1×1卷积组成的squeeze层降低输入通道数。然后将结果送入由1×1和3×3卷积混合组成的expand层。
基于SqueezeNet的想法，我们在CornerNet-Squeeze中使用了fire module，没有用残差模块。而且，受MobileNet的启发，我们将第二层中的标准3×3卷积替换为3×3深度可分离卷积，这进一步加快了推理速度。
我们没有继续探究SqueezeNet中的第三个策略。因为Hourglass网络有一个对称的结构，晚点进行下采样会导致在上采样时得到更高分辨率的特征图。在高分辨率特征图上进行卷积操作，计算成本更高，这就使我们没法进行实时检测。
除了替换残差模块，我们也做了其它的一些改动。我们在Hourglass模块之前增加了一个下采样层，以此降低Hourglass模块特征图的最大分辨率，并且在每个Hourglass模块内去掉了一个下采样层。CornerNet-Squeeze在Hourglass模块前相应地将图像尺寸缩小了3倍，但是CornerNet仅将图像尺寸缩小了2倍。我们在CornerNet的预测模块中，将3×3卷积替换为1×1卷积。最后，我们将hourglass网络中最相邻的上采样层替换为4x4的反卷积。 
**训练细节:**
我们使用了与CornerNet中同样的损失函数和超参数来训练CornerNet-Squeeze。唯一的改变就是 batch size。CornerNet-Squeeze面对同样的图像分辨率，在Hourglass模块之前缩小图像大小可以降低内存消耗4倍。我们在4个1080ti GPU上以batch size=55来训练网络，在主GPU上训练13张图片，剩余的GPU每个训练14张图片。
# CenterNet:Keypoint Triplets for Object Detection
论文:CenterNet: Keypoint Triplets for Object Detection
论文地址:https://arxiv.org/pdf/1904.08189.pdf 。
代码地址:https://github.com/Duankaiwen/CenterNet 。
在目标检测中，由于缺乏对相关剪裁区域的额外监督，基于关键点的方法通常会得到一大堆错误的物体边框。本文提出了一个有效的方法，在每个裁剪区域内以最小的代价去探索它的视觉模式。我们构建了一个单阶段基于关键点的检测器，叫做CornerNet。CornerNet 用每个目标物体的三个关键点来检测，而不是一对关键点，提升识别精度和召回率。因此，本文设计了两个模块，分别是 cascade corner pooling 和 center pooling，前者能丰富左上角和右下角搜集到的信息，后者在中间区域能提供更具辨识度的信息。在MS-COCO数据集上，CenterNet 获得的AP值是47%，比所有的单阶段检测器至少高出4.9%。同时，它的前向推理速度更快，CenterNet 的性能和双阶段检测器相比也很具竞争力。
深度学习出现之后，目标检测得到了明显的提升。目前最流行的方法都是基于ancho 的，在要识别物体上放置预先定义好的anchor boxes，通过ground truth boxes回归出相应的位置。这些方法通常需要一堆anchors来保证预测的边框和ground truth有较高的IoU，anchors的大小、宽高比都需要提前人为设计好。此外，anchors经常会和ground truth边框不一致，降低边框分类的准确率。
为了解决anchor的缺点，人们提出了一个基于关键点的目标检测方法CornerNet。它用一对角点来表示每个物体，无需anchor boxes，在one stage检测器中取得了state of art的检测准确率。但是，CornerNet仍有局限性，就是它缺乏对物体全局信息的参考。也就是说，由于每个物体都是用两个角点表示，算法对识别物体的边界框很敏感，而同时又无法确定哪两个关键点属于同一个物体。因此，经常会产生一些错误的边框，绝大多数都可以很容易地通过辅助信息（如宽高比）去除。
为了解决这个问题，我们让CornerNet可以识别每个候选区域内的视觉模式，这样它就能自己识别每个边框的正确性。在这篇论文中，我们提出了一个低成本但是很高效的办法叫做CenterNet，通过增加一个关键点来探索候选框内中间区域（靠近几何中心的位置）的信息。我们的想法就是，如果一个预测边框和ground truth边框有着很高的 IoU，则该边框的中心关键点预测出相同类别的概率要高，反之亦然。所以，在推理时，通过一对关键点产生了一个边框，如果同类别物体的中心关键点落在该候选框的中心区域，那么我们就认为该候选框包含那个物体。如果目标边框是准确的，那么在其中心区域能够检测到目标物体中心点的概率就会很高。 若有则保留该目标框，若无则删除该目标框。如图1，即使用三个关键点来表示目标物体。
**为了更好的检测中心关键点和角点，我们提出了两个方法来分别增强中心和角点信息:**
* 第一个方法叫center pooling，用于预测中心关键点的分支。Center pooling 有助于中心关键点取得物体内部辨识度更高的视觉信息，让候选框中心部分的感知更简单。实现方式是，在预测中心关键点的特征图上，取中心关键点横向和纵向上响应和的最大值。
* 第二个方法就是cascade corner pooling，增加原始 corner pooling 感知候选框内部信息的功能。实现方式是，在预测角点的特征图上，计算物体边框和内部方向上响应和的最大值。实验证明，这样一个双指向的池化方法面对噪声更加稳定，鲁棒性更强，有助于提升精度和召回。

我们在MS-COCO数据集上评估了CenterNet。在center pooling和cascade corner pooling都使用的情况下，在测试集上AP值能达到47%，超过了现有的单阶段检测器一大截。使用了52层的Hourglass主干网络时，推理时间平均为270毫秒每张图片；使用104层的Hourglass主干网络时，推理时间为340毫秒每张图片。CenterNet效率很高，和现有的two stage检测器相比也不弱。
## 基准和动机
这篇论文使用CornerNet作为基准。为了检测角点，CornerNet产生两个热力图：一个左上角的热力图，一个右下角的热力图。热力图代表不同类别关键点的位置，对每个关键点赋一个置信度分数。此外，CornerNet 也对每个角点预测一个 embedding 和一组偏移量。Embeddings 用于判断两个角点是否来自同一个目标物体。偏移量学习如何将角点从热力图重新映射回输入图像上，为了产生物体的边框，我们依据它们的分数从热力图上分别选取 top−k top-ktop−k 个左上角点和右下角点。然后，我们计算这一对角点的 embedding 向量的距离，以此来判断这一对角点是否属于同一个物体。如果距离小于某阈值，则会生成一个物体边框。该边框会得到一个置信度分数，等于这一对角点的平均分数。
**为了能够量化分析误检问题，研究人员提出了一种新的衡量指标，称为FD（false discovery，错误的检测边框的比例），能够很直观的反映出误检情况。FD的计算方式为:**
$$
F D_{i}=1-A P_{i}
$$
APi表示IOU阈值为i/100时对应的平均准确率。
结果显示在IoU阈值较低时，错误检测边框占了很大的比例，比如当IoU为0.05时，FD rate是32.7%。也就是平均下来，每一百个物体边框，有32.7个边框和ground truth边框的IoU是低于0.05的。小的错误边框就更多了，FD rate是60.3%。一个可能原因是，CornerNet无法深入边框内部一窥究竟。为了让CornerNet感知边框内的视觉信息，一个方案就是将CornerNet改为two stage检测器，使用RoI池化来深入了解边框内的视觉信息。但是，这种操作带来的计算成本很高。
在这篇论文，我们提出了一个非常有效的替代方案CenterNet，可以发掘每个边框内的视觉信息。为了检测物体，我们的方法使用了一个三元组关键点，而非一对关键点。这样做后，我们的方法仍然是一个单阶段检测器，但是部分继承了RoI池化的功能。此方法仅关注中心位置信息，计算成本是很小的。同时通过center pooling和cascade corner pooling，我们在关键点检测过程中进一步加入了物体内部的视觉信息。
## CenterNet的网络结构
```
                 -->cascade corner pooling->corner heatmaps->embeddings and offsets-->
                 |                                                                   |
输入图片->骨干网络-->                                           --------------------------->输出
                 |                                           |       
                 -->center pooling->center heatmap->offsets-->
```
## 基于三元组关键点的目标检测
**抑制误检的原理:**
如果目标框是准确的，那么在其中心区域能够检测到目标中心点的概率就会很高，反之亦然。因此，首先利用左上和右下两个角点生成初始目标框，对每个预测框定义一个中心区域，然后判断每个目标框的中心区域是否含有中心点，若有则保留该目标框，若无则删除该目标框。
**CenterNet的原理:**
我们用一个中心关键点和一对角点来表示每个物体。我们在CornerNet的基础上加入一个中心关键点的 heatmap，同时预测中心关键点的offsets。然后，基于CornerNet提出的方法产生top−k个候选框。同时，为了剔除错误的边框，利用检测到的中心点位置，我们对其按如下过程进行排序操作：
根据它们的分数，选择top−k个中心关键点；
* 使用相应的偏移量将中心关键点重新映射回输入图像中；
* 为每个边框定义一个中心区域，确保该中心区域存在中心关键点。同时确保该中心关键点的类别和边框的类别一致。如果在中心区域检测到中心关键点，我们就保留这个边框。用左上角，右下角和中心关键点分数的平均值更新边框的分数，并保存该边框。如果在该中心区域没有检测到中心关键点，则移除此边框。

**中心区域大小的确定方式:**
中心区域的大小影响会边界框的检测结果。比如，小中心区域对于小的边界框具有较低的召回率，而大区域相对于大的目标造成较低的精度。因此，本文提出了尺度敏感区域用于适应不同尺寸大小的目标物。其一般会生成相对小目标较大，相对大目标较小的中心区域，假设我们需要判断一个边界框I是否需要被保留，tlx,tly代表框左上角的点，brx,bry代表框右下角的点。定义一个中心区域j，定义左上角的点的坐标为(ctlx,ctly),右下角点(cbrx,cbry)。这些参数满足如下定义：
$$
\begin{cases} ctl_{x}=\frac{(n+1)tl_{x}+(n-1)br_{x}}{2n} \\\\ c t l_{y}=\frac{(n+1) t l_{y}+(n-1) b r_{y}}{2 n} \\\\ cbr_{x}=\frac{(n-1) t l_{x}+(n+1) b r_{x}}{2 n} \\\\ c b r_{y}=\frac{(n-1) t l_{y}+(n+1) b r_{y}}{2 n} \end{cases}
$$
n是个奇数，决定中心区域j的大小。在这篇论文中，当边框小于150时，n=3，否则n=5。根据上面的公式，我们可以得到自适应的中心区域，然后在里面检测中心区域是否包含中心关键点。
## 角点及中心点信息的丰富
**Center Pooling:**
物体的几何中心不一定能传达出recognizable的视觉信息（例如，人的头部有很强的视觉模式，但中心关键点通常位于人体的中间）。为了解决这个问题，作者在corner pooling基础上实现了center pooling。为了取得一个方向的最大值比如水平方向上，先使用从左至右的Corner Pooling，再跟着一个从右至左的Corner Pooling，即可直接得到水平方向的最大值。同理，使用top pooling和bottom pooling也可以得到竖直方向的最大值，再将水平Pooling和竖直Pooling的结果相加即可。
**Cascade corner pooling:**
物体bbox的两个角点往往在物体外部，缺少对物体局部特征的描述。因为Corner Pooling目的是找到物体边界部分的最大值来确定bbox，因此其对边缘非常敏感。作者为了解决这个问题，让物体在关注目标边界信息的同时也能关注目标内部信息。首先和Corner Pooling一样，首先沿边界查找边界最大值，然后沿边界最大值的位置查找内部最大值，最后将两个最大值相加。通过这种方式，使得角点可以同时获得物体的边界信息和内部视觉信息。Cascade Top Pooling的具体实现是首先使用Left Corner Pooling，获得每个点右边的最大值，然后使用Top Corner Pooling。
## 训练和推理
输入图片尺寸为511x511,heatmap大小为128x128，训练损失如下，参数定义同CornerNet。
$$
L=L_{det}^{co}+L_{det}^{ce}+\alpha L_{pull}^{co}+\beta L_{push}^{co}+\gamma \left(L_{off}^{co}+L_{off}^{ce}\right)
$$
第一项和第二项分别代表预测角点和中心点的Focal Loss。第三项用于最小化属于同一物体的角点组合向量之间的距离，第四项用于最大化不同物体的角点组合向量之间的距离。第五项括号内的两项都是L1 Loss，用于训练预测角点和中心点的偏移。前向时，选取top-70的角点和中心点，经过soft-nms，最后选取top100的检测框。
我们在8块 Tesla V100 GPUs 上训练，batch size 设为48。Iterations 的最大个数设为48万。在前45万个iterations中，学习率设为2.5×10的-4次方。最后3万个iterations，学习率为2.5×10的−5次方。
接着CornerNet论文，对于单一尺度测试，我们输入原图和经过水平翻转的图片，分辨率与原始分辨率一样。对于多尺度测试，我们输入原图和水平翻转的图片，分辨率分别为0.6、1.2、1.5、1.8。我们从热力图上选取前70个中心关键点，前70个左上角点，前70个右下角点，以此检测边框。在水平翻转的图片上，我们翻转检测到的边框，将它们和原来的边框进行混合。我们也用了 Soft-NMS 来去除多余的边框。最终，根据得分，选择前100个边框作为最终的检测结果。
# CenterNet:Objects as Points
论文:CenterNet:Objects as Points
论文地址:https://arxiv.org/pdf/1904.07850.pdf 。
代码地址:https://github.com/xingyizhou/CenterNet 。
目标检测识别往往在图像上将目标以轴对称的框形式框出。大多成功的目标检测器都先穷举出潜在目标位置，然后对该位置进行分类，这种做法浪费时间，低效，还需要额外的后处理。本文中，我们采用不同的方法，构建模型时将目标作为一个点——即目标BBox的中心点。我们的检测器采用关键点估计来找到中心点，并回归到其他目标属性，例如尺寸，3D位置，方向，甚至姿态。我们基于中心点的方法，称为：CenterNet，相比较于基于BBox的检测器，我们的模型是端到端可微的，更简单，更快，更精确。我们的模型实现了速度和精确的最好权衡，以下是其性能：
MS COCO dataset, with 28:1% AP at 142 FPS, 37:4% AP at 52 FPS, and 45:1% AP with multi-scale testing at 1.4 FPS.
用同个模型在KITTI benchmark 做3D bbox，在COCO keypoint dataset做人体姿态检测。同复杂的多阶段方法比较，我们的取得了有竞争力的结果，而且做到了实时。
**One stage detectors在图像上滑动复杂排列的可能bbox（即锚点）,然后直接对框进行分类，而不会指定框中内容。**
**Two-stage detectors对每个潜在框重新计算图像特征，然后将那些特征进行分类。**

**后处理，即NMS（非极大值抑制），通过计算Bbox间的IOU来删除同个目标的重复检测框。这种后处理很难区分和训练，因此现有大多检测器都不是端到端可训练的。**
本文通过目标中心点来呈现目标，然后在中心点位置回归出目标的一些属性，例如：size, dimension, 3D extent, orientation, pose。 而目标检测问题变成了一个标准的关键点估计问题。我们仅仅将图像传入全卷积网络，得到一个热力图，热力图峰值点即中心点，每个特征图的峰值点位置预测了目标的宽高信息。
**本文的模型训练采用标准的监督学习，推理仅仅是单个前向传播网络，不存在NMS这类后处理。**
**对我们的模型做一些拓展，可在每个中心点输出3D目标框，多人姿态估计所需的结果:**
* 对于3D BBox检测，我们直接回归得到目标的深度信息，3D框的尺寸，目标朝向；
* 对于人姿态估计，我们将关节点（2D joint）位置作为中心点的偏移量，直接在中心点位置回归出这些偏移量的值。

## 相关工作
**本文的CenterNet方法和anchor-based的one-stage目标检测方法类似，中心点可以看做是shape-agnostic（形状未知）的锚点，可以看做是一种隐式的anchors。**
* 第一，我们分配的锚点仅仅是放在位置上，没有尺寸框。没有手动设置的阈值做前后景分类（像Faster RCNN会将与GT IOU >0.7的作为前景，<0.3的作为背景，其他不管）；
* 第二，每个目标仅仅有一个正的锚点，因此不会用到NMS，我们提取关键点特征图上局部峰值点（local peaks）；
* 第三，CenterNet 相比较传统目标检测而言（缩放16倍尺度），使用更大分辨率的输出特征图（缩放了4倍），因此无需用到多重特征图锚点。

**通过关键点估计做目标检测:**
我们并非第一个通过关键点估计做目标检测的。CornerNet将bbox的两个角作为关键点；ExtremeNet 检测所有目标的 最上，最下，最左，最右，中心点；所有这些网络和我们的一样都建立在鲁棒的关键点估计网络之上。但是它们都需要经过一个关键点grouping阶段，这会降低算法整体速度；而我们的算法仅仅提取每个目标的中心点，无需对关键点进行grouping 或者是后处理。
**单目3D目标检测:**
3D BBox检测为自动驾驶赋能。Deep3Dbox使用一个slow-RCNN风格的框架，该网络先检测2D目标，然后将目标送到3D估计网络；3D RCNN在Faster-RCNN上添加了额外的head来做3D projection；Deep Manta使用一个coarse-to-fine的Faster-RCNN，在多任务中训练。而我们的模型同one-stage版本的Deep3Dbox 或3D RCNN相似，同样，CenterNet比它们都更简洁，更快。
## 初步工作
输入图像的宽W，高H。形式如下:
$$
I \in R^{W \times H \times 3}
$$
我们目标是生成关键点热力图:
$$
\hat Y \epsilon[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C}
$$
其中R是下采样因子，这里采用R=4。C表示关键点类别，C=17时为人关节点，用于人姿态估计； C= 80为目标类别数（COCO数据集中），用于目标检测。
在预测的heatmaps中:
$$
\hat Y_{x, y, c}=1
$$
表示检测到的关键点。
$$
\hat Y_{x, y, c}=0
$$
表示为背景。
在整个训练的流程中，CenterNet学习了CornerNet的方法。对于每个标签图(ground truth)中的某一 C 类，我们要将真实关键点p计算出来计算出来用于训练，中心点的计算方式为:
$$
p=\left(\frac{x_{1}+x_{2}}{2}, \frac{y_{1}+y_{2}}{2}\right)
$$
下采样后对应的关键点为:
$$
\tilde{p}=\left\lfloor\frac{p}{R}\right\rfloor
$$
R是上文中提到的下采样因子4。计算出来的是对应低分辨率图像中的中心点。
然后我们利用
$$
Y \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C}
$$
来对图像进行标记。在下采样的[128,128]图像中将ground truth point以
$$
Y \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C}
$$
的形式输入下面的高斯核:
$$
Y_{x y c}=\exp \left(-\frac{\left(x-\tilde p_{x}\right)^{2}+\left(y-\tilde p_{y}\right)^{2}}{2 \sigma_{p}^{2}}\right)
$$
其中σp是一个与目标大小（也就是w和h）相关的标准差。这样我们就将关键点分布到特征图上。如果对于同个类c（同个关键点或是目标类别）有两个高斯函数发生重叠，我们选择元素级最大的。
也就是说，每个点
$$
Y \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C}
$$
的范围是0-1,而1则代表这个目标的中心点，也就是我们要预测要学习的点。
**损失函数:**
在CenterNet中，作者使用了三种全卷积encoder-decoder网络：hourglass，Resnet和DLA来预测
$$
\hat Y
$$
关键点损失函数采用的是像素级逻辑回归的focal loss，如下公式所示:
$$
L_{k}=\frac{1}{N} \sum_{x y c} \begin{cases}{\left(1-\hat Y_{x y c}\right)^{\alpha} \log \left(\hat Y_{x y c}\right)} & {\text { if } Y_{x y c}=1} \\\\ {\left(1-Y_{x y c}\right)^{\beta}\left(\hat Y_{x y c}\right)^{\alpha} \log \left(1-\hat Y_{x y c}\right)} & {\text { otherwise }} \end{cases}
$$
其中α和β是focal loss的超参数，实验中两个数分别设置为2和4， N是图像中的关键点个数，除以N主要为了将所有focal loss归一化。
和Focal Loss类似，对于easy example的中心点，适当减少其训练比重也就是loss值，当
$$
\hat Y_{x y c}=1
$$
时， log前面的括号项就充当了矫正的作用。假如上式接近1的话，说明这个是一个比较容易检测出来的点，那么第二项前面的括号项就相应比较低了。而当上式接近0的时候，说明这个中心点还没有学习到，所以要加大其训练的比重，因此第二项前面的括号项就会很大。
**因为上文中对图像进行了R=4的下采样，这样的特征图重新映射到原始图像上的时候会带来精度误差，因此对于每一个中心点，额外采用了一个local offset来补偿误差:**
$$
\hat O \in \mathcal{R}^{\frac{W}{R} \times \frac{H}{R} \times 2}
$$
所有类 c 的中心点共享同一个offset prediction，这个偏置值(offset)用L1 loss来训练:
$$
L_{o f f}=\frac{1}{N} \sum_{p}\left|\hat O_{\tilde p}-\left(\frac{p}{R}-\tilde p\right)\right|
$$
绝对值项中第一项是我们预测出来的偏置，第二项是在训练过程中提前计算出来的数值。
事实上，这个偏置损失是可选的，我们不使用它也可以，只不过精度会下降一些。
## 目标看作点
令
$$
\left(x_{1}^{(k)}, y_{1}^{(k)}, x_{2}^{(k)}, y_{2}^{(k)}\right)
$$
代表类别为ck的目标k，其中心点为:
$$
p_{k}=\left(\frac{x_{1}^{(k)}+x_{2}^{(k)}}{2}, \frac{y_{1}^{(k)}+y_{2}^{(k)}}{2}\right)
$$
我们使用关键点:
$$
\hat Y
$$
来预测所有中心点。
此外，为每个目标k回归出目标的尺寸
$$
s_{k}=\left(x_{2}^{(k)}-x_{1}^{(k)}, y_{2}^{(k)}-y_{1}^{(k)}\right)
$$
sk这个值是在训练前提前计算出来的，是进行了下采样之后的长宽值。
为了减少计算负担，我们为每个目标种类使用单一的尺寸预测:
$$
\hat S \in R^{\frac{W}{R} \times \frac{H}{R} \times 2}
$$
因此我们可以在中心点位置添加L1 Loss:
$$
L_{s i z e}=\frac{1}{N} \sum_{k=1}^{N}\left|\hat s_{p k}-s_{k}\right|
$$
我们不将scale进行归一化，直接使用原始像素坐标。
**整个训练的损失函数如下:**
$$
L_{d e t}=L_{k}+\lambda_{s i z e} L_{s i z e}+\lambda_{o f f} L_{o f f}
$$
整体的损失函数为物体损失、大小损失与偏置损失的和，每个损失都有相应的权重。在论文中 λsize=0.1 ，λoff=1 ，论文中所使用的backbone都有三个head layer，分别产生[1,80,128,128]、[1,2,128,128]、[1,2,128,128]，也就是每个坐标点产生C+4个数据（即关键点类别C, 偏移量的x,y，尺寸的w,h）。
**从点到回归框:**
在预测阶段，我们首先针对一张图像进行下采样，随后对下采样后的图像进行预测，对于每个类在下采样的特征图中预测中心点，然后将输出图中的每个类的热点单独地提取出来。如何提取呢？我们将热力图上的所有响应点与其连接的8个临近点进行比较，如果该点响应值大于或等于其八个临近点值则保留，最后我们保留所有满足之前要求的前100个峰值点。
假设
$$
\hat  P_{c}
$$
表示类别c的n个中心点的集合。
$$
\hat P
$$
代表c类中检测到的一个点。
每个关键点的位置用整型坐标表示:
$$
\left(x_{i}, y_{i}\right)
$$
使用
$$
\hat Y_{x_{i} y_{i} c}
$$
表示当前点的置信度。
然后我们使用下面的公式来产生目标框:
$$
(\hat x_{i}+\delta \hat x_{i}-\hat w_{i} / 2, \hat y_{i}+\delta \hat y_{i}-\hat h_{i} / 2, \hat x_{i}+\delta \hat x_{i}+\hat w_{i} / 2, \hat y_{i}+\delta \hat y_{i}+\hat h_{i} / 2)
$$
其中:
$$
\left(\delta \hat x_{i}, \delta \hat y_{i}\right)=\hat O \hat x_{i}, \hat y_{i}
$$
是当前点对应原始图像的偏置点。
$$
\left(\hat w_{i}, \hat h_{i}\right)=\hat S \hat x_{i}, \hat y_{i}
$$
是预测出来当前点对应目标的长宽。
最终根据模型预测出来的
$$
\hat Y \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times C }
$$
即就是当前中心点存在物体的概率值，代码中设置的阈值为0.3，也就是从上面选出的100个结果中选出大于该阈值的中心点作为最终的结果。
**3D检测:**
3D检测是对每个目标进行3维bbox估计，每个中心点需要3个附加信息：depth，3D dimension，orientation（深度、3D维度和方向）。因此对其中每个部分增加一个单独的部分检测头。
 对于每个中心点，深度值depth是一个维度的。然后depth很难直接回归，我们参考Depth map prediction from a single image using a multi-scale deep network这篇文章对输出做了变换:
$$
d=1 / \sigma(\hat d)-1
$$
其中σ是sigmoid函数。
我们在特征点估计网络上添加了一个深度估计器:
$$
\hat D \in[0,1]^{\frac{W}{R} \times \frac{H}{R}}
$$
该通道使用了两个卷积层，然后做ReLU 。我们用L1 loss来训练该深度估计器。
目标的3D维度是三个标量值。我们直接回归出它们（长宽高）的绝对值，单位为米。
$$
\hat \Gamma \in[0,1]^{\frac{W}{R} \times \frac{H}{R} \times 3}
$$
该3D维度估计器也使用L1 loss来训练。
方向默认是单标量的值，然而其也很难回归。我们参考3d bounding box estimation using deep learning and geometry这篇文章的做法，使用两个bins来呈现方向，且i做n-bin回归。方向用8个标量值来编码的形式，每个bin有4个值。对于一个bin,两个值用作softmax分类，其余两个值回归到在每个bin中的角度。
**人体姿态估计:**
人体姿态估计估计出每个人体的k个关节点位置（COCO数据集中k=17，即每个人有17个关节点），我们令中心点的姿态是2k维的，然后将每个关键点（关节点对应的点）参数化为相对于中心点的偏移（L1 Loss）:
$$
\hat J=R^{\frac{W}{R} \times \frac{H}{R} \times 2}
$$
为了精修关键点，估计k个人体部位热力图 （Focal Loss）:
$$
\hat \Phi \in R^{\frac{W}{R} \times \frac{H}{R} \times k}
$$
我们使用标准的bottom-up多人体姿态估计，我们训练人的关节点热力图使用focal loss和像素偏移量，这块的思路和中心点的训练雷同。
我们将初始预测映射到特征图上检测到的最近的关键点。然后将中心偏移作为分组标志，将各个检测到的关键点分配给最近的人体。
具体来说，令:
$$
(\hat x, \hat y)
$$
代表检测到的中心点。
第一次回归得到的关节点为:
$$
l_{j}=(\hat x, \hat y)+\hat J_{\hat x \hat y \hat j}(j=1,2, \ldots, k)
$$
我们提取到所有关键点（关节点，此处是类似中心点检测用热力图回归得到的，对于热力图上值小于0.1的直接略去），然后将每个回归（第一次回归，通过偏移方式）位置lj与最近的检测关键点（关节点）进行分配:
$$
\arg \min_{l \in L_{j}} (l-l_{j})^{2}
$$
## 实现细节
我们实验了4个结构：ResNet-18, ResNet-101, DLA-34， Hourglass-104。我们用deformable卷积层来更改ResNets和DLA-34，按照原样使用Hourglass网络。
**Hourglass:**
堆叠的Hourglass网络通过两个连续的hourglass模块对输入进行了4倍的下采样，每个hourglass模块是个对称的5层下和上卷积网络，且带有skip连接。该网络较大，但通常会生成最好的关键点估计。
**ResNet:**
Xiao等人对标准的ResNet做了3个up-convolutional网络来dedao更高的分辨率输出（最终stride为4）。为了节省计算量，我们改变这3个up-convolutional的输出通道数分别为256,128,64。up-convolutional核初始为双线性插值。
**DLA:**
即Deep Layer Aggregation（DLA），是带多级跳跃连接的图像分类网络，我们采用全卷积上采样版的DLA，用deformable卷积来跳跃连接低层和输出层；将原来上采样层的卷积都替换成3x3的deformable卷积。在每个输出head前加了一个3x3x256的卷积，然后做1x1卷积得到期望输出。
**Training:**
训练输入图像尺寸：512x512；输出分辨率：128x128（即4倍stride）；采用数据增强方式：随机flip, 随机scaling（比例在0.6到1.3），裁剪，颜色jittering；采用Adam优化器。

在3D估计分支任务中未采用数据增强（scaling和crop会影响尺寸）。
## 总结
CenterNet通过预测一个中心点解决了2D目标检测，3D目标检测以及姿态估计任务，完美的把这三个任务统一到一起；
与另一篇CenterNet:Keypoint Triplets for Object Detection相比较，Keypoint Triplets则是通过中心点预测抑制CornerNet中存在的大量误检，而本篇则是使用目标中心点来预测目标；
从使用关键点来进行的目标检测方法在coco上训练耗时来看，这类方法普遍存在训练耗时较长的问题。
在实际训练中，如果在图像中两个目标经过下采样后中心点重叠了，那么CenterNet会将这两个物体的当成一个物体来训练（因为只有一个中心点）。同理，在预测过程中，如果两个同类的物体在下采样后的中心点也重叠了，那么CenterNet也只能检测出一个中心点。
# FCOS
论文:FCOS: Fully Convolutional One-Stage Object Detection
论文地址:https://arxiv.org/pdf/1904.01355.pdf 。
代码地址:https://github.com/tianzhi0549/FCOS 。
本文提出了一种全卷积的one-stage目标检测器（FCOS），以逐像素预测方式解决目标检测，类似于语义分割。相比于RetinaNet，SSD，YOLOv3和Faster R-CNN等依赖于预定义的锚框（anchor boxes）的目标检测模型，FCOS不需要锚框，因此FCOS完全避免了与锚框相关的复杂计算，且避免了与锚框相关的所有超参数，这些参数通常对最终检测性能非常敏感。FOCS凭借唯一的后处理非极大值抑制（NMS），达到了比以往one-stage检测模型更好的性能。
## 介绍
全卷积网络FCN在密集预测的任务如语义分割，深度估计，关键点检测上取得了不错的效果。但是在以往的目标检测任务中，由于anchor boxes的使用，并没有应用全卷积逐像素点预测的目标检测模型。如果我们能像语义分割中的FCN那样使用逐像素点预测的方式来解决目标检测问题，那么视觉任务中所有的问题都可以用一个框架来解决。这篇论文证明这是可以的。而且，和基于anchors的检测器相比，基于FCN的检测器更加简单，也可以取得更好的效果。
以往的DenseBox和UnitBox等方法也曾尝试将全卷积框架用于目标检测。这些基于FCN的框架在每层特征图的每个空间位置上，直接预测一个4D向量加一个类别。4D向量表示像素点到4个边框的距离，这样每一个像素点都要预测一个4D向量。但是为了应付不同大小的边框，DenseBox将图像缩放到一个固定尺寸。这样DenseBox不得不在图像金字塔上进行检测，这和FCN的一次处理完所有卷积的方式相悖，而且这些方法应用在通用目标检测任务上效果不会很好，往往它们具有高度重叠的边框。高度重叠的边框会造成界限不明确，很难理清重叠部分的像素点应该属于哪一个边框。
接下来我们证明了FPN能有效地解决这种不明确的问题。我们发现FCOS会在远离目标物体中心的位置上产生一些效果不好的预测边框。为了降低这些不好的检测结果，我们引入了一个 “center-ness” 分支（只有一层），预测像素点到目标边框中心的距离。这个分数然后用于降低效果不好的边框的权重，然后用NMS将检测结果合并。Center-ness分支很简单，也很有效。有了它，在同样的训练和测试环境下，我们基于FCN的检测器就能超过基于anchor的检测器。
**FCOS与anchor-based检测器的区别:**
* anchor-based算法将输入图像上的位置作为锚框的中心店，并且对这些锚框进行回归。FCOS直接对feature map中每个位置对应原图的边框都进行回归，换句话说FCOS直接把每个位置都作为训练样本，这一点和FCN用于语义分割相同。FCOS算法feature map中位置与原图对应的关系，如果feature map中位置为（x,y），映射到输入图像的位置是:
$$
\left(\left\lfloor\frac{s}{2}\right\rfloor+ x s,\left\lfloor\frac{s}{2}\right\rfloor+ y s\right)
$$
* 在训练过程中，anchor-based算法对样本的标记方法是，如果anchor对应的边框与真实边框（ground truth）的IOU大于一定阈值，就设为正样本，并且把交并比最大的类别作为这个位置的类别。而在FCOS中，如果位置 (x,y) 落入任何真实边框，就认为它是一个正样本，它的类别标记为这个真实边框的类别。
* 以往的anchor-based算法训练一个多元分类器，而FCOS训练C个二元分类器（C是类别数）。
* 在多尺度FPN的检测上，anchor-based算法将不同尺寸的锚框分配到不同级别的特征层，而FCOS通过直接限定不同特征级别的边界框的回归范围来进行分配。

**FCOS模型的优点:**
FCOS模型使用语义分割的思想来解决目标检测问题，它摒弃了目标检测中常见的anchor boxes和object proposal，使得不需要调优涉及anchor boxes和object proposal的超参数；
在训练过程中避免大量计算GT boxes和anchor boxes 之间的IoU，使得训练过程占用内存更低；
FCOS可以作为two stage检测器的区域建议网络（RPN），其性能明显优于基于锚点的RPN算法；
FCOS可以经过最小的修改便可扩展到其他的视觉任务，包括实例分割、关键点检测。
**FCOS算法步骤:**
对输入的图片进行预处理操作；
搭建下图所示的网络架构，将输入数据送入backbone网络中获取输入数据的feature_map，在feature_map的每一点上面进行回归操作，进行网络训练获取网络模型；
将预训练的网络模型应用到测试图片中，从特征金字塔的多个Head中获得预测的结果；
使用NMS等后处理操作获得最终的结果。
## FCOS网络结构
```
输入
 |
C1
 |
C2
 |
C3——>P3——>head
 |   |
C4——>P4——>head  
 |   |
C5——>P5——>head
     |
     P6——>head
     |
     P7——>head
每一个head的结构:
                                                -->Classification HxWxC
                                                |
      -->HxWx256-->HxWx256-->HxWx256-->HxWx256-->
      |                                         |
      |                                         -->Center-ness HxWx1
      |
head-->
      |
      -->HxWx256-->HxWx256-->HxWx256-->HxWx256-->Regression HxWx4
```
P3-P7是FPN特征金字塔，head是一个三分支的头检测网络。
## 全卷积one stage检测器
设
$$
F_{i} \in \mathbb{R}^{H \times W \times C}
$$
为CNN的第i层特征图，s是该层之前的总步长。
输入图像的ground truth边框定义为:
$$
B_{i}=\left(x_{0}^{(i)}, y_{0}^{(i)}, x_{1}^{(i)} y_{1}^{(i)}, c^{(i)}\right) \in R^{4} \times\{1,2, \ldots, C\}
$$
上式中:
$$
\left(x_{0}^{(i)}, y_{0}^{(i)}\right) 和 \left(x_{1}^{(i)} y_{1}^{(i)}\right)
$$
分别表示边框左上角和右下角的坐标。
$$
c^{(i)}
$$
表示边框中目标的类别。
C是类别的总数，对于COCO数据集而言，C=80。
对特征图Fi上的每个位置（x,y），我们可以将其映射回输入图像:
$$
\left(\left\lfloor\frac{s}{2}\right\rfloor+ x s,\left\lfloor\frac{s}{2}\right\rfloor+ y s\right)
$$
基于anchor的检测器将输入图像上的位置作为anchor boxes的中心，然后对这些anchor boxes回归出目标边框。而我们的方法是直接在特征图的每个位置上回归出目标边框。也就是说，我们的检测器直接将每个点看作训练样本，而不是将anchor boxes看作训练样本，这和语义分割中的FCN一样。
如果（x,y）落入一个ground truth边框内，它就被标注为正样本，该位置的标签
$$
c^{\ast}
$$
就是Bi的标签。否则它就是负样本（背景类），有:
$$
c^{\ast}=0
$$
除了分类的标签外我们也有一个4D的向量:
$$
t^{\ast}=\left(l^{\ast}, t^{\ast}, r^{\ast}, b^{\ast}\right)
$$
作为每一个样本回归的目标。四个变量分别代表该位置到边框四条边的距离。如果一个点落入多个边框之中，它就被视作模糊样本。就目前来说，我们只选取最小面积的边框作为回归的目标。如果位置（x,y）与边框Bi相关联，该位置的回归目标可定义为:
$$
l^{\ast}=x-x_{0}^{(i)}, t^{\ast}=y-y_{0}^{(i)}, r^{\ast}=x_{1}^{(i)}-x, b^{\ast}=y_{1}^{(i)}-y
$$
FCOS模型能够利用尽可能多的前景样本来训练回归器。这和基于anchor boxes的检测器不同，它们只将那些和ground truth边框IOU足够高的anchor boxes当作正样本。我们认为也许这是FCOS比基于 anchor的检测器效果好的原因之一。
**网络输出:**
与训练目标对应，最后一层预测一个类别标签的80维的向量p，以及一个4维的向量t=（l,t,r,b），即中心点距离边框的left、top、right和bottom边之间的距离。我们训练C个二元分类器，而不是一个多类别分类器。我们在主干网络特征图之后增加4个卷积层，分别对应分类和回归分支。而且，由于回归目标通常是正的，我们在回归分支上面用exp（x）将任意实数映射到（0,正无穷）之内。FCOS的参数个数要比基于anchor的检测器少9倍，因为一般基于anchor的方法在每个位置上会有9个anchor boxes。
**损失函数:**
$$
 L\left(p_{x, y}t_{x, y}\right) =\frac{1}{N_{pos}} \sum_{x, y} L_{cls}\left(p_{x, y}, c_{x, y}^{\ast}\right) +\frac{\lambda}{N_{pos}} \sum_{x, y} 1_{(c_{x, y}^{\ast}>0)} L_{reg}\left(t_{x, y},t_{x, y}^{\ast}\right)
$$
该loss函数包含两部分，Lcls表示分类loss，本文使用的是Focal_loss（类别损失）；Lreg表示回归loss，本文使用的是IOU loss。
px,y表示特征图的位置（x,y）处的分类概率，tx,y表示特征图的位置（x,y）处的回归坐标。Npos表示正样本的个数，在这篇论文中λ=1用于平衡Lreg的权重。对特征图Fi上的各个位置的结果进行求和。
$$
1_{c^{\ast}>0}
$$
是指标函数，当
$$
c_{i}^{\ast}>0
$$
指标函数值为1，否则为0。
**前向推理:**
FCOS 的前向推理很直接。给定输入图片，前向通过整个网络，获得特征图Fi上每个位置的分类得分px,y以及回归预测tx,y。如果一个位置的px,y>0.05，则它被列为正样本，然后通过上面的公式计算l，t，r，b获得预测边框。
## 用FPN对FCOS进行多尺度的预测
**FCOS可能遇到的两个问题:**
* 最后一个特征图上较大的步长可能导致低召回率。对于基于anchor的检测器，因步长较大而导致召回率低的问题，可以通过降低判断正样本的IOU阈值来弥补。对于FCOS，我们证明即使步长很大，基于FCN的FCOS检测器也能产生足够好的召回率。而且，它甚至要比基于anchor的RetinaNet要好。而且，利用多层级FPN预测，能够进一步提升召回率；
  能被进一步提升，达到 RetinaNet 的最好成绩。
* Ground truth边框的重叠区域会造成训练中的不明确，到底重叠区域内的位置应该回归到哪个边框里去？这个问题会导致基于FCN的检测器性能降低。我们证明这种不明确问题可以通过FPN多层级预测解决，和基于anchor的检测器相比较，基于FCN的检测器能取得更优的成绩。

在FCOS中，我们在特征图的不同层级上检测不同大小的物体，我们使用了特征图的5种层级，即P3、P4、P5、P6、P7。P3、P4、P5是通过CNN 的特征图C3、C4、C5后接一个1×1的卷积层而产生。P6、P7通过在P5、P6上分别应用一个步长为2的卷积层而得到。特征层P3、P4、P5、P6、P7的步长分别为8、16、32、64、128。
基于anchor的检测器在不同特征层上分配不同大小的anchor boxes，而我们直接限定边框回归的范围。更具体点，我们首先在所有特征层上的每个位置计算回归目标:
$$
l^{\ast}, t^{\ast}, r^{\ast}, b^{\ast}
$$
如果一个位置满足:
$$
\max \left(l^{\ast}, t^{\ast}, r^{\ast}, b^{\ast}\right)>m_{i} 或\max \left(l^{\ast}, t^{\ast}, r^{\ast},    b^{\ast}\right)<m_{i-1}
$$
那么它就被设为负样本，就不需要回归边框。mi是第i个特征层需要回归的最大距离。在论文中，m2、m3、m4、m5、m6、m7分别被设为0、64、128、256、512、∞。因为不同大小的物体被分配到不同的特征层，而绝大多数的重叠物体彼此间的大小很不一样，多层级预测能极大地缓解前面提到的重叠区域模糊问题，因而提升FCN检测器的精度。
最后，我们在不同的特征层级间共享 heads，提升了检测器的效率和性能。但是我们发现不同特征层级需要回归不同的大小范围（比如对P3是[0,64]，对P4是[64,128] ），因而对不同的特征层级使用一样的heads是不合理的。所以，除了使用标准的exp（x），我们也使用exp（six） ，其中si是一个可训练的标量，自动调节特征层Pi的指数函数的底数，从而提升性能。
## Center-ness
通过FPN多尺度预测之后发现FCOS和基于anchor的检测器之间仍然存在着一定的性能差距，主要原因是距离目标中心较远的位置产生很多低质量的预测边框。
Center-ness的作用就是用来很好的抑制这些低质量的预测边框的产生，它的优点是比较简单。不需要引入其它的超参数。它的位置是在Head网络的分类网络分支下面，与分类分支平行。对于给定的一个位置的回归目标:
$$
l^{\ast}, t^{\ast}, r^{\ast}, b^{\ast}
$$
center-ness目标的定义如下:
$$
centerness ^{\ast}=\sqrt{\frac{\min \left(l^{\ast}, r^{\ast}\right)}{\max \left(l^{\ast}, r^{\ast}\right)} \times \frac{\min \left(t^{\ast}, b^{\ast}\right)}{\max \left(t^{\ast}, b^{\ast}\right)}}
$$
使用根号是为了降低centerness衰减的速度。center-ness（中心度）取值为0到1之间，通过二元交叉熵损失来训练。并把这个损失加入前面提到的损失函数中。测试时，将预测的center-ness和对应的分类得分相乘，得到最终的得分，再用这个得分对检测边框进行排名。因此center-ness可以降低那些远离物体中心边框的得分。在最后的NMS过程中，这些低质量的边框就会很大概率上被剔除，从而显着提高了检测性能。
基于anchor的检测器使用2个IOU阈值Tlow,Thigh来将anchor box标为负样本、忽略和正样本。而 center-ness可以看作为一个soft阈值。Center-ness通过模型训练来学习，而无需手动去调。而且依据此方法，我们的检测器仍可以将任意落入ground truth边框的点看作正样本，除了那些在多层级预测中已经被标注为负样本的点，在回归器中就可以使用尽可能多的训练样本。
## 算法细节
在训练阶段，文中使用ResNet-50作为backbone网络，使用SGD优化器，初始学习率为0.01，batch_size=16，在迭代60K和80K时的weight_decay分别为0.0001和0.9，使用ImagNet预训练权重进行初始化，将输入图片裁剪为短边不小于800，长边不小于1333大小。整个网络是在COCO数据集上面训练得到的。
## 总结
在COCO数据集上，各种目标检测算法PK。FCOS已经超越 Two-stage的Faster R-CNN，还超越了 One-stage的YOLOv2、SSD、RetinaNet，以及很新的CornerNet。
相比于YOLOV1算法，YOLOV1只利用了目标的中心区域的点做预测，因此召回率较低。而FCOS利用了目标的整个区域内的点，召回率和基于anchor-based的算法相当；尽管centerness确实带来效果上的明显提升，但是缺乏理论可解释性；作为一种新的one stage算法，论文中未题算法的推理的速度，可见该算法的速度并不算快。