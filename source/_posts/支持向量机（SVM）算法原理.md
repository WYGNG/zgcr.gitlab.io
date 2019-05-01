---
title: 支持向量机（SVM）算法原理
date: 2019-03-08 19:31:28
tags:
- 机器学习原理推导
categories:
- 机器学习原理推导
mathjax: true
---

# 支持向量机（SVM）算法介绍
支持向量机（support vector machines, SVM）是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器。它与感知机的最大区别是学习策略为间隔最大化。
SVM可分类为三类：线性可分支持向量机（数据线性可分）、线性支持向量机（数据近似线性可分）及非线性支持向量机（数据线性不可分）。
# 线性可分支持向量机
## 线性可分支持向量机定义
给定一个线性可分的训练数据集，通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面为
$$
wx+b=0
$$
以及相应的分类决策函数
$$
f(x)=sign(wx+b)
$$
就称为线性可分支持向量机。距离超平面最近的样本点就称为支持向量，显然正类和负类各有一个支持向量。

**如何求解得到间隔最大化的超平面呢？**
## 函数间隔、几何间隔与硬间隔最大化
### 函数间隔
**超平面关于样本点（xi，yi）的函数间隔为：**
$$
\hat\gamma_{i}=y_{i}(wx_{i}+b)
$$
通过观察wx+b的符号与类标记y的符号是否一致可判断分类是否正确，为了统一，我们把y乘以wx+b，这样无论是正样本点还是负样本点结果都是正的。
**对于数据集T，定义超平面对T的函数间隔为超平面关于数据集T中所有样本点的函数间隔值之中的最小值：**
$$
\hat\gamma=\min_{i=1, \cdots, N} \hat\gamma_{i}
$$
当我们选择超平面时，只有函数间隔还不够，因为只要成比例地改变w和b, 例如将它们改为2w和2b, 超平面并没有改变， 但函数间隔却成为原来的2倍。
**我们可以对分离超平面的法向量w加某些约束， 比如对函数间隔除以W的L2范数，此时函数间隔就变为几何间隔。几何间隔才是点到超平面的距离。**
### 几何间隔
**定义超平面关于样本点（xi,yi）的几何间隔为：**
$$
\gamma_{i}=y_{i}\left(\frac{w}{\|w\|} \cdot x_{i}+\frac{b}{\|w\|}\right)
$$
**超平面对数据集T的几何间隔为超平面关于数据集T中所有样本点的几何间隔值之中的最小值：**
$$
\gamma=\min_{i=1, \cdots, N} \gamma_{i}
$$
 我们可以发现函数间隔和几何间隔有如下关系：
$$
\gamma=\frac{\hat\gamma}{||w||}
$$
**当我们取分母即W的L2范数为1时，函数间隔和几何间隔相等**。
如果超平面参数w和b成比例地改变（超平面没有改变），函数间隔也按此比例改变，而几何间隔不变。
### 硬间隔最大化
我们寻找的超平面应当满足：这个超平面不仅可以将数据集中的所有正负实例点分开，而且对最难区分的实例点（离超平面最近的点，也就是支持向量）也有足够大的间隔将它们分开。
**寻找几何间隔最大的分离超平面问题可表示为下面的约束最优化问题:**
$$
\max_{w, b} \gamma
$$
$$
s.t. \quad y_{i}(\frac{w}{||w||} x_{i}+\frac{b}{||w||}) \geqslant \gamma, \quad i=1,2, \cdots, N
$$
即我们要最大化超平面关于数据集T的几何间隔γ，约束条件则表示训练集中每个点与超平面的几何间隔都大于等于γ。
事实上我们就是要找到一对支持向量（一正例一负例两个样本点，超平面在这两个点的正中间，使得超平面对两个点的距离最远），同时这个超平面对于更远处的其他所有正负实例点都能正确区分。
**我们可将上式的几何间隔用函数间隔表示，这个问题可改写为:**
$$
\max_{w, b} \frac{\hat\gamma}{||w||}
$$
$$
s.t. \quad y_{i}(wx_{i}+b) \geqslant \hat\gamma, \quad i=1,2, \cdots, N
$$
 此时如果我们将w和b放大λ倍，那么右边的函数间隔也会放大λ倍。也就是说约束条件左右两边都整体放大了λ倍，对不等式约束没有影响。因此，我们不妨取:
$$
\hat\gamma=1
$$
取1是为了方便推导和优化，且这样做对目标函数的优化没有影响。
**那么上面的最优化问题变为:**
$$
\max_{w, b} \frac{1}{||w||}
$$
$$
s.t. \quad y_{i}(wx_{i}+b) \geqslant 1, \quad i=1,2, \cdots, N
$$
最大化1/||w||与最小化（1/2）||w||2是等价的，乘以1/2是为了后面求导时计算方便。
**则我们可以得到下面的线性可分支持向量机学习的最优化问题:**
$$
\min_{w, b} \frac{1}{2}||w||^{2}
$$
$$
s.t. \quad y_{i}(wx_{i}+b)-1 \geqslant 0, \quad i=1,2, \cdots, N
$$
## 线性可分支持向量机的原始问题转化为对偶问题
### 原始问题
**上面的最优化问题我们称之为原始问题。**
我们可以列出一个拉格朗日函数来求解原始问题。我们定义拉格朗日函数为:
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}(w x_{i}+b)+\sum_{i=1}^{N} \alpha_{i}
$$
$$
\alpha_{i} \geqslant 0, i=1,2, \cdots, N
$$
**原始问题是一个极小极大问题。**
$$
\min_{w, b} \max_{\alpha_{i} \geq 0} L(w, b, \alpha)
$$
直接求原始问题并不好解，我们可以应用拉格朗日对偶性，通过求解对偶问题来得到原始问题的最优解。
### 对偶问题及求解过程
**根据拉格朗日对偶性，对偶问题是极大极小问题，即求:**
$$
\max_{\alpha_{i} \geq 0} \min_{w, b} L(w, b, \alpha)
$$
**对偶问题的求解过程:**
先对L（w,b,α）求对w、b的极小。我们需要求L对w、b的偏导数并令其为0。
$$
\nabla_{w} L(w, b, \alpha)=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0
$$
$$
\nabla_{b} L(w, b, \alpha)=\sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
得到两个等式:
$$
w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
$$
$$
\sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
将上面两式代入拉格朗日函数，得:
$$
\min_{w, b}L(w, b, \alpha)=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j}\right) \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}
$$
$$
=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
再求min L（w,b,α）对α的极大，即:
$$
\max_{\alpha}-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
$$
\quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
$$
\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
$$
**我们将上面第一个式子乘以-1，即可得到等价的对偶最优化问题:**
$$
\min_{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$
$$
\quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
$$
\alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
$$
**我们可以求出上面问题对α的解向量（α即为所有拉格朗日乘子）为:**
$$
(\alpha_{1},\alpha_{2}, \cdots,\alpha_{N})^{T}
$$
**则按上面求L对w、b的偏导数并令其为0得到的等式，我们可得:**
$$
w=\sum_{i=1}^{N} \alpha_{i}y_{i}x_{i}
$$
$$
b=y_{j}-\sum_{i=1}^{N} \alpha_{i}y_{i}(x_{i} \cdot x_{j})
$$
**那么超平面就为**
$$
\sum_{i=1}^{N} \alpha_{i}y_{i}(x \cdot x_{i})+b=0
$$
**决策函数为**
$$
f(x)=sign(\sum_{i=1}^{N} \alpha_{i}y_{i}x \cdot x_{i}+b)
$$
我们注意到对原始的最优化问题:
$$
\min_{w, b} \frac{1}{2}||w||^{2}
$$
$$
s.t. \quad y_{i}(wx_{i}+b)-1 \geqslant 0, \quad i=1,2, \cdots, N
$$
**有不等式约束，因此上述过程必须满足KKT条件:**
$$
\alpha_{i} \geqslant 0
$$
$$
y_{i} f\left(x_{i}\right)-1 \geqslant 0
$$
$$
\alpha_{i}\left(y_{i} f\left(x_{i}\right)-1\right)=0
$$
计算实例在《统计学习方法》P107-P108。
# 线性支持向量机
我们经常遇到这样的情况：数据集T中大部分数据线性可分，但有少部分数据线性不可分，这时使用上面的线性可分支持向量机并不可行。这时我们就要使用学习策略为软间隔最大化的线性支持向量机。
## 软间隔最大化
所谓软间隔最大化，就是在硬间隔最大化的基础上，对每个样本点（xi,yi）引进一个松弛变量ξi，于是我们的约束不等式变为:
$$
y_{i}(wx_{i}+b) \geqslant 1-\xi_{i}
$$
同时，我们的目标函数中对每个松弛变量ξi需要支付一个代价ξi。
**即目标函数变为:**
$$
\frac{1}{2}||w||^{2}+C \sum_{j=1}^{N} \xi_{i}
$$
这里C>0称为惩罚参数。C较大时对误分类的惩罚增大，C较小时对误分类的惩罚减小。
上面的目标函数包含两层含义:一是使间隔尽量大，二是使误分类点的个数尽量小。
松弛变量ξi表示xi在超平面下不符合真实分类，但是与超平面距离<ξi，因此我们将其归到另一类（也就是真实分类）。松弛变量用来解决异常点，它可以"容忍"异常点在错误的区域，使得支持向量机计算出的超平面不至于产生过拟合的问题，但同时，我们又不希望松弛变量太大导致超平面分类性能太差。
## 线性支持向量机的原始问题转化为对偶问题
### 原始问题
$$
\min_{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}
$$
$$
\quad y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N
$$
$$
\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
$$
### 对偶问题及解题过程
**原始问题的对偶问题为:**
$$
\min_{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$
$$
\quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
$$
0 \leqslant \alpha_{i} \leqslant C, \quad i=1,2, \cdots, N
$$
**拉格朗日函数为:**
$$
L(w, b, \xi, \alpha, \mu) \equiv \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}-\sum_{i=1}^{N} \alpha_{i}\left(y_{i}\left(w \cdot x_{i}+b\right)-1+\xi_{i}\right)-\sum_{i=1}^{N} \mu_{i} \xi_{i}
$$
$$
\alpha_{i} \geqslant 0, \mu_{i} \geqslant 0
$$
类似的，先求L对w、b、ξ的极小，求L对w、b、ξ的偏导数并令偏导数为0，最后得:
$$
w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
$$
$$
\sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
$$
C-\alpha_{i}-\mu_{i}=0
$$
上面三式代回L，得:
$$
\min_{w, b, \xi} L(w, b, \xi, \alpha, \mu)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
再对上式求α的极大，得到对偶问题:
$$
\max_{\alpha}-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
$$
\quad \sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$
$$
C-\alpha_{i}-\mu_{i}=0
$$
$$
\begin{array}{l}{\alpha_{i} \geqslant 0} \\ {\mu_{i} \geqslant 0, \quad i=1,2, \cdots, N}\end{array}
$$
设
$$
\alpha=(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{N})^{T}
$$
是上面对偶问题的拉格朗日乘子的解，那么有:
$$
w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
$$
$$
b=y_{j}-\sum_{i=1}^{N} y_{i} \alpha_{i}(x_{i} \cdot x_{j})
$$
**那么超平面为:**
$$
\sum_{i=1}^{N} \alpha_{i} y_{i}(x \cdot x_{i})+b=0
$$
**决策函数为:**
$$
f(x)=sign(\sum_{i=1}^{N} \alpha_{i} y_{i}(x \cdot x_{i})+b)
$$
# 非线性支持向量机、核函数与核技巧
如果我们的数据集样本是非线性可分的，我们可以使用一个映射将样本从原始空间映射到一个合适的高维空间，在这个高维空间中数据集线性可分。
## 核函数
设X是输入空间，H是特征空间（即高维空间）,如果存在一个映射:
$$
\phi(x) : X \rightarrow H
$$
使得
$$
K(x, z)=\phi(x) \cdot \phi(z)
$$
$$
x, z \in \mathcal{X}
$$
那么称K（x,z）为核函数。Φ（x）为映射函数，等式右边是内积。
## 核技巧
现在假设我们用映射:
$$
x \rightarrow \phi(x)
$$
将超平面从原本低维空间的形式变成高维空间的形式:
$$
f(x)=\omega^{T} x+b
$$
$$
f(x)=\omega^{T} \phi(x)+b
$$
 这时我们原本在低维的非线性边界在高维就变成了一个超平面，在高维中数据集变得线性可分了。
 但是如果我们先将数据集从低维映射到高维，再去求超平面，由于高维空间中维数要多的多，计算量会变得很大。
**核技巧是一种方法，我们用它可以简化计算。它就是用低维空间的内积来求高维空间的内积，省去了先做映射变换再在高维空间中求内积的麻烦。** 

**举例:**
以多项式核为例，若样本x有d个分量:
$$
x=\left(x_{1}, x_{2}, \ldots . . x_{d}\right)
$$
我们做一个二维的多项式变换，将d维扩展:
$$
\phi(x)=(1 ; x_{1}, x_{2}, \ldots x_{d} ; x_{1}^{2}, x_{1} x_{2} \ldots x_{1} x_{d} ; \ldots ; x_{d} x_{1},x_{d} x_{2} \ldots x_{d}^{2})
$$
注意上面的式子中有很多重复项可以合并。
我们对变换后的样本作内积:
$$
\phi(x)^{T} \phi(y)=1+\sum_{i=1}^{d} x_{i} y_{i}+\sum_{i=1}^{d} \sum_{j=1}^{d} x_{i} x_{j} y_{i} y_{j}
$$
 其中，最后一项可以拆成:
$$
\sum_{i=1}^{d} \sum_{j=1}^{d} x_{i} x_{j} y_{i} y_{j}=\sum_{i=1}^{d} x_{i} y_{i} \sum_{j=1}^{d} x_{j} y_{j}
$$
又低维空间的内积可表示为:
$$
x^{T} y=\sum_{i=1}^{d} x_{i} y_{i}
$$
所以上面变换后的多项式内积可以写成:
$$
\phi(x)^{T} \phi(y)=1+x^{T} y+\left(x^{T} y\right)^{2}
$$
我们就可以不先进行高维变换，再在高维空间做内积，而是直接利用低维空间的内积计算即可。

**SVM中的核技巧:**
不使用核函数时支持向量机的最优化问题为:
$$
\max \frac{2}{|| \omega||}
$$
$$
y_{i}(\omega^{T} x_{i}+b) \geq 1
$$
使用核函数将数据集样本从低维映射到高维空间后最优化问题变为:
$$
\max \frac{2}{||\omega||}
$$
$$
y_{i}\left(\omega^{T} \phi\left(x_{i}\right)+b\right) \geq 1
$$
于是我们可以得到类似的对偶问题:
$$
\max_{\alpha} \sum_{i=1} \alpha_{i}-\sum_{i} \sum_{j} \alpha_{i} \alpha_{j} y_{i} y_{j} \phi\left(x_{i}\right)^{T} \phi\left(x_{j}\right)
$$
$$
\sum_{i} \alpha_{i} y_{i}=0, \alpha_{i} \geq 0
$$
我们在计算其中的
$$
\phi\left(x_{i}\right) \phi\left(x_{j}\right)
$$
内积时就可以使用核技巧。使用核函数表示对偶问题为:
$$
W(\alpha)=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} K\left(x_{i}, x_{j}\right)-\sum_{i=1}^{N} \alpha_{i}
$$
分类决策函数表示为:
$$
f(x)=sign(\sum_{i=1}^{N_{s}} a_{i}y_{i} \phi(x_{i}) \cdot \phi(x)+b)=sign(\sum_{i=1}^{N_{s}} a_{i}y_{i}K(x_{i}, x)+b)
$$
