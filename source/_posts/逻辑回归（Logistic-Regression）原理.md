---
title: 逻辑回归（Logistic Regression）原理
date: 2019-05-02 18:02:39
tags:
- 机器学习原理推导
categories:
- 机器学习原理推导
mathjax: true
---

# sigmoid函数
**sigmoid函数公式:**
$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$
当x趋近于负无穷时，函数值趋近于0；当x趋近于正无穷时，y趋近于1；当x= 0时，y=0.5。这种特征使得sigmoid函数很适合用来做二分类。
**sigmoid函数求导:**
$$
\begin{aligned} \sigma^{\prime}(x) &=\left(\frac{1}{1+e^{-x}}\right)^{\prime} \\\\ &=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}} \\\\ &=\frac{1}{\left(1+e^{-x}\right)}\frac{e^{-x}}{\left(1+e^{-x}\right)} \\\\ &=\sigma(x)(1-\sigma(x)) \end{aligned}
$$
# 逻辑回归原理推导
普通的线性回归的模型是求出输出特征向量Y和输入样本矩阵X之间的线性关系系数W和b，满足Y=WX+b。此时我们的Y是连续的，所以是回归模型。
如果我们想要Y是离散的话，怎么办呢？我们就用上面的sigmoid函数对Y再做一次变换，得到g（Y）。，我们把g（Y）大于等于0.5的看作类别1，小于0.5的看作类别0，这样得到了一个二分类模型。这就是逻辑回归模型。
sigmoid函数公式:
$$
g(z)=\frac{1}{1+e^{-z}}
$$
如果我们令
$$
z=wx+b
$$
**可得逻辑回归模型的一般形式:**
$$
h_{w,b}(x)=\frac{1}{1+e^{-(wx+b)}}
$$
下面我们进一步给出更加准确的逻辑回归定义。
**二项逻辑回归模型定义为具有如下条件概率分布的模型:**
$$
P(Y=1 | x)=\frac{\exp (w \cdot x+b)}{1+\exp (w \cdot x+b)}
$$
$$
P(Y=0 | x)=\frac{1}{1+\exp (w \cdot x+b)}
$$
x是输入（特征空间）,Y是输出标签，w和b为模型参数。注意看上面两式似乎和我们一开始介绍的不太一样？实际上我们将P（Y=1 | x）除以
为了方便表达，将b扩充入w ，同时输入x后加入一项全为1的列，使得
$$
w \cdot x+b \Rightarrow w \cdot x
$$
此时有
$$
w=w_{1}, w_{2}, \cdots, w_{n}, b, x=x_{1}, x_{2}, \cdots, x_{n}, 1
$$
**此时逻辑回归的条件概率分布如下:**
$$
P(Y=1 | x)=\frac{\exp (w \cdot x)}{1+\exp (w \cdot x)}
$$
$$
P(Y=0 | x )=\frac{1}{1+\exp (w \cdot x+b)}
$$
**一个事件发生的几率指该事件发生的概率与不发生的概率的比值。**
如果某一事件发生的概率为 p, 那么它发生的几率为
$$
\frac{p}{1-p}
$$
该事件发生的对数几率函数为
$$
logit(p)=\log \frac{p}{1-p}
$$
对于P（Y=1|x），其对数几率函数为
$$
\log \frac{P(Y=1 | x)}{1-P(Y=1 | x)}=w \cdot x
$$
也就是说，在逻辑回归模型中，输出Y=1的对数几率是输入x的线性函数。
或者说，通过逻辑回归模型条件概率分布式可以将线性函数wx转换为概率。线性函数wx的值越接近正无穷，概率值就越接近1；线性函数wx的值越接近负无穷，概率值就越接近0。
# 极大似然估计法与梯度下降法估计模型参数
## 极大似然估计
在逻辑回归中我们可以认为样本是伯努利分布（n重二项分布）。对于给定的训练集T:
$$
T=\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots ,\left(x_{N}, y_{N}\right)
$$
设
$$
P(Y=1 | x)=\pi(x), \quad P(Y=0 | x)=1-\pi(x)
$$
似然函数为:
$$
\prod_{i=1}^{N}\left[\pi\left(x_{i}\right)\right]^{y_{i}}\left[1-\pi\left(x_{i}\right)\right]^{1-y_{i}}
$$
**对数似然函数为:**
$$
\begin{aligned} L(w) &=\sum_{i=1}^{N}\left[y_{i} \log \pi\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-\pi\left(x_{i}\right)\right)\right] \\\\ &=\sum_{i=1}^{N}\left[y_{i} \log \frac{\pi\left(x_{i}\right)}{1-\pi\left(x_{i}\right)}+\log \left(1-\pi\left(x_{i}\right)\right)\right] \\\\ &=\sum_{i=1}^{N}\left[y_{i}\left(w \cdot x_{i}\right)-\log \left(1+\exp \left(w \cdot x_{i}\right))\right]\right.\end{aligned}
$$
上面L（w）的第一步公式就是深度学习中常用的交叉熵损失函数形式。
**这样我们只需要对L（w）求极大值，从而得到w的估计值即可。**
如果w只有一个参数，我们可以用对L求w的导数，再令导数为0的方法，求出w的值，此时L（w）是一个极大值点。
## 梯度下降法
在深度学习模型中我们往往有多个w，此时用上面的方法很难解出w，此时我们往往采用梯度下降法或拟牛顿法进行多轮迭代寻找w的估计值:
先求L对某个wi的梯度:
$$
\frac{\partial L(w)}{\partial w_{i}}=y_{i} x_{i}-\frac{\exp \left(w \cdot x_{i}\right) \cdot x_{i}}{1+\exp \left(w\cdot x_{i}\right)} =\left(y_{i}-\frac{1}{1+\exp (-w \cdot x)}\right) x_{i}
$$
wi参数更新公式:
$$
w_{i}=w_{i}-\eta \frac{\partial L(w)}{\partial w_{i}}=w_{i}-\eta \left(y_{i}-\frac{1}{1+\exp (-w \cdot x)}\right) x_{i},(i=1,2, \cdots, n)
$$
# 逻辑回归与SVM的异同
**相同点:**
* lr和SVM都是分类算法;
* 如果不考虑核函数，则lr和SVM都是线性分类算法；
* lr和SVM都是监督学习算法；
* lr和SVM都是判别模型，计算的是P（Y|X）；

**不同点:**
* lr和SVM的损失函数不同。逻辑回归方法基于概率理论，假设样本为1的概率可以用sigmoid函数来表示，然后通过极大似然估计的方法估计出参数的值；支持向量机基于几何间隔最大化原理，认为存在最大几何间隔的分类面为最优分类面；
* SVM只考虑边界线附近的少数点（支持向量），而逻辑回归考虑全局所有点（远离的点对边界线的确定也起作用）。影响SVM决策面的样本点只有少数的支持向量，当在支持向量外添加或减少任何样本点对分类决策面没有影响；而在lr中，每个样本点都会影响决策面的结果。因此线性SVM不直接依赖于数据分布，分类平面不受某类点影响；lr则受所有数据点的影响，如果数据不同类别十分不平衡，则一般需要先对数据做平衡采样。
* 解决非线性问题时，支持向量机采用核函数的机制，而lr通常不采用核函数的方法。因为模型训练的过程就是决策面的计算过程，计算决策面时，SVM算法中只有少数几个代表支持向量的样本参与了计算，即只有少数几个样本需要参与核计算（即kernal machine解的系数是稀疏的）；而lr算法中，每个样本点都必须参与决策面的计算过程，假如我们在lr里也运用核函数的原理，那么每个样本点都必须参与核计算，计算量十分大。因此在具体应用时，lr很少运用核函数机制。
* 线性SVM依赖数据表达的距离测度，所以需要对数据先做归一化；而lr计算的是概率，归一化可以加快模型收敛速度，如果不归一化，只要训练的足够充分，其最终结果也先相同。
* SVM的损失函数中自带了正则项（1/2||w||2），而lr本身不带正则项，我们必须在损失函数上另外添加正则项。

