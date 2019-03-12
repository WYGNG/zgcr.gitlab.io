---
title: 梯度提升树（GBDT）原理
date: 2019-03-12 13:04:18
tags:
- 机器学习原理推导
categories:
- 机器学习原理推导
mathjax: true
---

# 梯度提升树介绍
基于梯度提升算法的学习器叫做 GBM（Gradient Boosting Machine）。理论上，GBM可以选择各种不同的学习算法作为基学习器。但在现实中，用得最多的基学习器是决策树。当使用的基学习器是决策树时，我们称之为梯度提升树（Gradient Boosting Decision Tree）。
回顾Adaboost算法，我们是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，同时迭代思路和Adaboost也有所不同。
梯度提升树的主要思想是，每一次建立模型是在之前建立模型损失函数的梯度下降方向，即利用了损失函数的负梯度在当前模型的值作为回归问题提升树算法的残差近似值，去拟合一个回归树。损失函数用来评价模型性能，损失函数越小，性能越好。只要让损失函数持续下降，就能使模型不断提高性能，而让损失函数快速下降的最好方法就是使损失函数沿着梯度方向下降（因为梯度方向上下降最快）。
# 加法模型（additive model）
GBDT算法可以看成是由K棵树组成的加法模型：
$$
\hat y_{i}=\sum_{k=1}^{K} f_{k}(x_{i}), f_{k} \in F
$$
其中F是指所有基模型组成的函数空间（即深度学习中的假设空间）。
一般化的损失函数是预测值与真实值之间的关系，如我们前面的平方损失函数，那么对于n个样本来说，则可以写成：
$$
L=\sum_{i=1}^{n} l\left(y_{i}, \hat y_{i}\right)
$$
最小化损失函数就相当于最小化模型的偏差，但同时我们也需要兼顾模型的方差，所以目标函数还包括抑制模型复杂度的正则项，因此目标函数可以写成：
$$
O b j=\sum_{i=1}^{n} l\left(y_{i}, \hat y_{i}\right)+\sum_{k=1}^{K} \Omega\left(f_{k}\right)
$$
其中Ω代表了基模型的复杂度，若基模型是树模型，则树的深度、叶子节点数等指标可以反应树的复杂程度。
**如何学习加法模型？**
使用前向分布算法（forward stagewise algorithm），因为学习的是加法模型，如果能够从前往后，每一步只学习一个基函数及其系数（结构），逐步逼近优化目标函数，那么就可以简化复杂度。这一学习过程称之为Boosting。具体地，我们从一个常量预测开始，每次学习一个新的函数，过程如下:
$$
\begin{aligned} \hat y_{i}^{0} &=0 \\\\ \hat y_{i}^{1} &=f_{1}\left(x_{i}\right)=\hat y_{i}^{0}+f_{1}\left(x_{i}\right) \\\\ \hat y_{i}^{2} &=f_{1}\left(x_{i}\right)+f_{2}\left(x_{i}\right)=\hat y_{i}^{1}+f_{2}\left(x_{i}\right) \\\\ & \cdots \\\\ \hat y_{i}^{t} &=\sum_{k=1}^{t} f_{k}\left(x_{i}\right)=\hat y_{i}^{t-1}+f_{t}\left(x_{i}\right) \end{aligned}
$$
在每一步中如何决定加入哪一个函数f呢？指导原则还是最小化目标函数。 
在第t步，模型对xi的预测为:
$$
\hat y_{i}^{t}=\hat y_{i}^{t-1}+f_{t}\left(x_{i}\right)
$$
其中ft（xi）为这一轮我们要学习的函数（决策树）。则目标函数可写为:
$$
\begin{aligned} O b j^{(t)} &=\sum_{i=1}^{n} l\left(y_{i}, \hat y_{i}^{t}\right)+\sum_{i=i}^{t} \Omega\left(f_{i}\right) \\\\ &=\sum_{i=1}^{n} l\left(y_{i}, \hat y_{i}^{t-1}+f_{t}\left(x_{i}\right)\right)+\Omega\left(f_{t}\right)+\text {constant } \end{aligned}
$$
假如损失函数为平方损失，则目标函数为:
$$
\begin{aligned} O b j^{(t)} &=\sum_{i=1}^{n}\left(y_{i}-\left(\hat y_{i}^{t-1}+f_{t}\left(x_{i}\right)\right)\right)^{2}+\Omega\left(f_{t}\right)+\text { constant } \\\\ &=\sum_{i=1}^{n}\left[2\left(\hat y_{i}^{t-1}-y_{i}\right) f_{t}\left(x_{i}\right)+f_{t}\left(x_{i}\right)^{2}\right]+\Omega\left(f_{t}\right)+\text {constant } \end{aligned}
$$
其中:
$$
\left(\hat y_{i}^{t-1}-y_{i}\right)
$$
称之为残差。因此，使用平方损失函数时，GBDT算法的每一步在生成决策树时只需要拟合前面的模型的残差。
由泰勒公式:
$$
f(x+\Delta x) \approx f(x)+f^{\prime}(x) \Delta x+\frac{1}{2} f^{\prime \prime}(x) \Delta x^{2}
$$
目标函数是关于变量:
$$
\hat y_{i}^{t-1}+f_{t}\left(x_{i}\right)
$$
的函数，若把变量:
$$
\hat y_{i}^{t-1}
$$
看成是泰勒公式中的x，把变量ft（xi）看成是泰勒公式中的Δx，那么上面的目标函数可以展开为:
$$
O b j^{(t)}=\sum_{i=1}^{n}\left[l\left(y_{i}, \hat y_{i}^{t-1}\right)+g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right)+\text{constant}
$$
$$
g_{i}=\partial_{\hat y^{t-1}} l\left(y_{i}, \hat y^{t-1}\right)
$$
$$
h_{i}=\partial_{\hat y^{t-1}}^{2} l\left(y_{i}, \hat y^{t-1}\right)
$$
分别是损失函数的一阶导数和二阶导数。假如损失函数为平方损失函数,则:
$$
g_{i}=\partial_{\hat y^{t-1}}\left(\hat y^{t-1}-y_{i}\right)^{2}=2\left(\hat y^{t-1}-y_{i}\right)
$$
$$
h_{i}=\partial_{\hat y^{t-1}}^{2}\left(\hat y^{t-1}-y_{i}\right)^{2}=2
$$
将上两式代入泰勒公式展开后的目标函数，由于函数中的常量在函数最小化的过程中不起作用，因此我们还可以移除掉常量项，得:
$$
O b j^{(t)} \approx \sum_{i=1}^{n}\left[g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right)
$$
由于要学习的函数仅仅依赖于目标函数，从上式可以看出只需为学习任务定义好损失函数，并为每个训练样本计算出损失函数的一阶导数和二阶导数，通过在训练样本集上最小化上式即可求得每步要学习的函数，根据最初的加法模型公式可得最终要学习的模型。
# GBDT算法
一颗生成好的决策树，假设其叶子节点个数为T，该决策树是由所有叶子节点对应的值组成的向量w。
把特征向量映射到叶子节点所引的函数:
$$
q : R^{d} \rightarrow\{1,2, \cdots, T\}
$$
那么决策树可表示为
$$
f_{t}(x)=w_{q(x)}
$$
决策树的复杂度可以由下面的正则项定义:
$$
\Omega\left(f_{t}\right)=\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2}
$$
即决策树模型的复杂度由生成的树的叶子节点数量和叶子节点对应的值向量的L2范数决定。
定义集合
$$
I_{j}=\{i | q\left(x_{i}\right)=j\}
$$
为所有被划分到叶子节点j的训练样本的集合。则目标函数为:
$$
\begin{aligned} O b j^{(t)} & \approx \sum_{i=1}^{n}\left[g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right) \\\\ &=\sum_{i=1}^{n}\left[g_{i} w_{q\left(x_{i}\right)}+\frac{1}{2} h_{i} w_{q\left(x_{i}\right)}^{2}\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\\\ &=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T \end{aligned}
$$
定义
$$
G_{j}=\sum_{i \in I_{j}} g_{i}
$$
$$
H_{j}=\sum_{i \in I_{j}} h_{i}
$$
其中gi为一阶导数，hi为二阶导数。
则目标函数可写成:
$$
O b j^{(t)}=\sum_{j=1}^{T}\left[G_{i} w_{j}+\frac{1}{2}\left(H_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
$$
假设树的结构是固定的，即q（x）固定，令上面目标函数的一阶导数为0（对wj求导），则叶子节点j对应的值为：
$$
w_{j}^{*}=-\frac{G_{j}}{H_{j}+\lambda}
$$
则目标函数的值为:
$$
O b j=-\frac{1}{2} \sum_{j=1}^{T} \frac{G_{j}^{2}}{H_{j}+\lambda}+\gamma T
$$
**总结一下单棵决策树的生成过程:**
* 枚举所有可能的树结构q；
* 用上面最后的目标函数表达式为每个q计算其对应的目标函数分数，分数越小说明对应的树结构越好；
* 根据上一步的结果，找到最佳的树结构，用上面的wj表达式为树的每个叶子节点计算预测值。

**然而，可能的树结构数量是无穷的，实际上我们不可能枚举所有可能的树结构。通常情况下，我们采用贪心策略来生成决策树的每个节点:** 
* 从深度为0的树开始，对每个叶节点枚举所有的可用特征 
* 针对每个特征，把属于该节点的训练样本根据该特征值升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的最大收益（采用最佳分裂点时的收益），其实这个就是决策树中遇到连续值特征时找到最佳划分点的做法。每个特征的特征值排序的时间复杂度为O（nlogn），假设公用K个特征，那么生成一颗深度为K的树的时间复杂度为O（Knlogn）。
* 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，把该节点生长出左右两个新的叶节点，并为每个新节点关联对应的样本集；
* 回到第1步，递归执行到满足特定条件为止。

上面的步骤就是一个标准的特征值为连续值的决策树的生成过程，但收益计算公式与标准决策树的最大增益、最大增益比、基尼指数不同。
假设当前节点记为C，分裂之后左孩子节点记为L，右孩子节点记为R，则该分裂获得的收益定义为当前节点的目标函数值减去左右两个孩子节点的目标函数值之和:
$$
G a i n=O b j_{C}-O b j_{L}-O b j_{R}
$$
即:
$$
\operatorname{Gain}=\frac{1}{2}\left[\frac{G_{L}^{2}}{H_{L}+\lambda}+\frac{G_{R}^{2}}{H_{R}+\lambda}-\frac{\left(G_{L}+G_{R}\right)^{2}}{H_{L}+H_{R}+\lambda}\right]-\gamma
$$
其中，-γ项表示因为增加了树的复杂性（该分裂增加了一个叶子节点）带来的惩罚。
**总结GBDT的学习算法过程:**
* 算法每次迭代生成一颗新的决策树；
* 每次迭代开始之前，计算损失函数在每个训练样本点的一阶导数和二阶导数；
* 通过贪心策略生成新的决策树，通过wj表达式计算每个叶节点对应的预测值；
* 把新生成的决策树添加到模型中：
$$
\hat y_{i}^{t}=\hat y_{i}^{t-1}+f_{t}\left(x_{i}\right)
$$

通常在最后一步，我们把模型更新公式替换为:
$$
\hat y_{i}^{t}=\hat y_{i}^{t-1}+\epsilon f_{t}\left(x_{i}\right)
$$
其中ε称之为步长或者学习率。增加因子的目的是为了避免模型过拟合。