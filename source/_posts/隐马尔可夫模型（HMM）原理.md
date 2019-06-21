---
title: 隐马尔可夫模型（HMM）原理
date: 2019-06-21 12:41:56
tags:
- 机器学习原理推导
categories:
- 机器学习原理推导
mathjax: true
---

# 概率图模型与隐马尔可夫模型
概率图模型是一类用图来表示变量相关关系的模型。可以分为两类:一类是用有向无环图表示变量间的依赖关系，称为有向图模型；另一类是使用无向图表示变量间的相关关系，称为无向图模型。
隐马尔可夫模型（HMM）是一种有向图模型，它是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。隐藏的马尔可夫链随机生成不可观测的状态的序列称为状态序列；每个状态生成一个观测，再由此产生的观测的随机序列，称为观测序列。序列的每一个位置可以看作是一个时刻。
# 隐马尔可夫模型原理
## 隐马尔可夫模型定义
隐马尔可夫模型由初始概率分布、状态转移概率分布、观测概率分布确定。设Q是所有可能的状态的集合，V是所有可能的观测的集合，即:
$$
Q=(q_{1}, q_{2}, \cdots, q_{N}), \quad V=(v_{1}, v_{2}, \cdots, v_{M})
$$
其中，N是可能的状态数，M是可能的观测数。
I是长度为T的状态序列，O是对应的观测序列，即:
$$
I=\left(i_{1}, i_{2}, \cdots, i_{T}\right), \quad O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)
$$
A为状态转移概率矩阵（NXN矩阵）:
$$
A=\left[a_{i j}\right]
$$
其中:
$$
a_{i j}=P\left(i_{t+1}=q_{j} | i_{t}=q_{i}\right), \quad i=1,2, \cdots, N ; j=1,2, \cdots, N
$$
即在时刻t处于状态qi的条件下在时刻t+1转移到状态qj的概率。
B为观测概率矩阵（NXM矩阵）:
$$
B=\left[b_{j}(k)\right]
$$
其中:
$$
b_{j}(k)=P\left(o_{t}=v_{k} | i_{t}=q_{j}\right), \quad k=1,2, \cdots, M ; j=1,2, \cdots, N
$$
是在时刻t处于状态qj的条件下生成观测vk的概率。
记π为初始状态概率向量:
$$
\pi=\left(\pi_{i}\right)
$$
其中:
$$
\pi_{i}=P\left(i_{1}=q_{i}\right), \quad i=1,2, \cdots, N
$$
表示时刻t=1处于状态qi的概率。
因此，HMM模型λ可以用三元符号表示，即：
$$
\lambda=(A, B, \pi)
$$
A,B,π称为HMM模型的三要素。
**举例:**
假设有4个盒子，每个盒子都有红白两种颜色的球，球的数量如下:
```
盒子        1        2        3        4
红球数      5        3        6        8
白球数      5        7        4        2
```
按下面的方法抽取球:
开始时，从4个盒子中等概率地抽取一个，再从盒子中随机抽一个球，记录颜色后放回。然后从当前盒子转移到下一个盒子，如果当前为盒子1，下一个盒子一定是2；如果当前为盒子2或3，以概率0.4和0.6转移到左边或右边的盒子；如果当前为盒子4，各以0.5概率停留在盒子4或转移到盒子3。转移后，再从盒子中随机抽一个球，记录颜色后放回。
现在假设我们要连续地抽5次。抽取结果如下:
$$
O=(红,红,白,白,红)
$$
这个例子中有两个随机序列:
盒子序列（状态序列）和球颜色序列（观测序列）。前者是隐藏的，后者是可观测的。
则状态集合Q和观测集合V为:
$$
Q=(盒子1,盒子2,盒子3,盒子4), \quad V=(红,白)
$$
状态序列和观测序列长度T=5。
开始时，从4个盒子中等概率地抽取一个，则初始概率分布π为:
$$
\pi=(0.25,0.25,0.25,0.25)^{\mathrm{T}}
$$
状态转移概率分布A为（由盒子转移规则得出）:
$$
A=\left[\begin{array}{cccc}{0} & {1} & {0} & {0} \\\\ {0.4} & {0} & {0.6} & {0} \\\\ {0} & {0.4} & {0} & {0.6} \\\\ {0} & {0} & {0.5} & {0.5}\end{array}\right]
$$
观测概率分布B为（由每个盒子红白球比例计算得出）:
$$
B=\left[\begin{array}{ll}{0.5} & {0.5} \\\\ {0.3} & {0.7} \\\\ {0.6} & {0.4} \\\\ {0.8} & {0.2}\end{array}\right]
$$
## 两个基本假设和三个基本问题
**隐马尔可夫模型做了两个基本假设:**
* 齐次马尔可夫性假设，即假设隐藏的马尔可夫链在任意时刻t的状态只依赖于其前一时刻的状态，与其他时刻的状态及观测无关，也与时刻t无关，即:
$$
P\left(i_{t} | i_{t-1}, o_{t-1}, \cdots, i_{1}, o_{1}\right)=P\left(i_{t} | i_{t-1}\right), \quad t=1,2, \cdots, T
$$
* 观测独立性假设，即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态，与其他的观测和状态无关，即:
$$
P\left(o_{t} | i_{T}, o_{T}, i_{T-1}, o_{T-1}, \cdots, i_{t+1}, o_{t+1}, i_{t-1}, i_{t-1}, \cdots, i_{1}, o_{1}\right)=P\left(o_{t} | i_{t}\right)
$$

**隐马尔可夫模型有3个基本问题:**
1. 概率计算问题。给定模型λ和观测序列O，计算在模型λ下观测序列O出现的慨率P（O|λ）。
2. 学习问题。已知观测序列O，估计模型λ的参数，使得在该模型下观测序列概率P（O|λ）最大。即用极大似然估计的方法估计参数。
3. 预测问题，也称为解码问题。已知模型λ和观测序列O，求对给定观测序列条件概率P（I|O）最大的状态序列。即给定观测序列，求最有可能的对应的状态序列。

## 概率计算问题:P（O|λ）的计算方法
给定模型λ和观测序列O，计算在模型λ下，观测序列O出现的概率P（O|λ）。
### 直接计算方法（概念上可行，计算上不可行）
列举所有可能的长度为T的状态序列I
$$
I=\left(i_{1}, i_{2}, \cdots, i_{T}\right)
$$
求各个状态序列I和给定的观测序列O的联合概率P（O,I∣λ），然后对所有可能的状态序列求和，得到P（O|λ）。
对某个状态序列I的概率为:
$$
P(I | \lambda)=\pi_{i_{1}} a_{i_{1} i_{2}} a_{i_{2} i_{3}} \cdots a_{i_{T-1} i_{T}}
$$
对上面的状态序列I，输入的观测序列O的概率P（O|I,λ）:
$$
P(O | I, \lambda)=b_{i_{1}}\left(o_{1}\right) b_{i_{2}}\left(o_{2}\right) \cdots b_{i_{T}}\left(o_{T}\right)
$$
O和I同时出现的l联合概率为:
$$
P(O, I | \lambda)=P(O | I, \lambda) P(I | \lambda)=\pi_{i_{1}} b_{i_{1}}\left(o_{1}\right) a_{i_{1} i_{2}} b_{i_{2}}\left(o_{2}\right) \cdots a_{i_{i-1} i_{\tau}} b_{i_{\tau}}\left(o_{T}\right)
$$
然后，对所有可能的状态序列I求和，得到观测序列O的概率P（O|λ）:
$$
P(O | \lambda)=\sum_{I} P(O | I, \lambda) P(I | \lambda)=
\sum_{i_{1}, i_{2}, \cdots, i_{T}} \pi_{i_{1}} b_{i_{1}}\left(o_{1}\right) a_{i_{1} i_{2}} b_{i_{2}}\left(o_{2}\right) \cdots a_{i_{i-1} i_{\tau}} b_{i_{\tau}}\left(o_{T}\right)
$$
通过这种计算方式的计算量非常大，其复杂度为:
$$
O\left(T N^{T}\right)
$$
实际是不可行的。在真实情况下，一般采用更有效的算法，即前向-后向算法。
### 前向算法
给定隐马尔可夫模型λ和观测序列O，定义到时刻t部分观测序列为:
$$
o_{1}, o_{2}, \cdots, o_{t}
$$
且状态为qi的概率为前向概率，记作:
$$
\alpha_{t}(i)=P\left(o_{1}, o_{2}, \cdots, o_{t}, i_{t}=q_{i} | \lambda\right)
$$
**下面要计算观测序列概率P（O∣λ）。**
计算初值:
$$
\alpha_{1}(i)=\pi_{i} b_{i}\left(o_{1}\right), \qquad i=1,2, \cdots, N
$$
递推，对t=1,2,⋯,T−1，有
$$
\alpha_{t+1}(i)=\left[\sum_{j=1}^{N} \alpha_{t}(j) a_{j i}\right] b_{i}\left(o_{t+1}\right), \quad i=1,2, \cdots, N
$$
终止:
$$
P(O | \lambda)=\sum_{i=1}^{N} \alpha_{T}(i)
$$
该算法时间复杂度为:
$$
O\left(N^{2} T\right)
$$
比直接计算法小很多。
**计算实例:**
现在有盒子和球模型λ=（A,B,π），状态集合Q=（1,2,3），观测集合V=（红，白）。
状态转移概率分布A、观测概率分布B、初始概率分布π为:
$$
A=\left[\begin{array}{lll}{0.5} & {0.2} & {0.3} \\\\ {0.3} & {0.5} & {0.2} \\\\ {0.2} & {0.3} & {0.5}\end{array}\right]
$$
$$
B=\left[\begin{array}{ll}{0.5} & {0.5} \\\\ {0.4} & {0.6} \\\\ {0.7} & {0.3}\end{array}\right]
$$
$$
\pi=(0.2,0.4,0.4)^{\mathrm{T}}
$$
设T=3, O=（红,白,红），试用前向算法计算P（O|λ）。
计算初值:
$$
\alpha_{1}(1)=\pi_{1} b_{1}\left(o_{1}\right)=0.10 \\\\ \alpha_{1}(2)=\pi_{2} b_{2}\left(o_{1}\right)=0.16 \\\\ \alpha_{1}(3)=\pi_{3} b_{3}\left(o_{1}\right)=0.28
$$
递推计算:
$$
\alpha_{2}(1)=\left[\sum_{i=1}^{3} \alpha_{1}(i) a_{i 1}\right] b_{1}\left(o_{2}\right)=0.154 \times 0.5=0.077 \\\\ \alpha_{2}(2)=\left[\sum_{i=1}^{3} \alpha_{1}(i) a_{i 2}\right] b_{2}\left(o_{2}\right)=0.184 \times 0.6=0.1104 \\\\ \alpha_{2}(3)=\left[\sum_{i=1}^{3} \alpha_{1}(i) a_{13}\right] b_{3}\left(o_{2}\right)=0.202 \times 0.3=0.0606
$$
$$
\alpha_{3}(1)=\left[\sum_{i=1}^{3} \alpha_{2}(i) a_{i 1}\right] b_{1}\left(o_{3}\right)=0.04187 \\\\ \alpha_{3}(2)=\left[\sum_{i=1}^{3} \alpha_{2}(i) a_{i 2}\right] b_{2}\left(o_{3}\right)=0.03551 \\\\  
\alpha_{3}(3)=\left[\sum_{i=1}^{3} \alpha_{2}(i) a_{i 3}\right] b_{3}\left(o_{3}\right)=0.05284
$$
终止:
$$
P(O | \lambda)=\sum_{i=1}^{3} \alpha_{3}(i)=0.13022
$$
### 后向算法
给定隐马尔可夫模型λ和观测序列O，定义在时刻t状态为qi的条件下，从t+1到T的部分观测序列为:
$$
o_{t+1}, o_{t+2}, \cdots, o_{T}
$$
的概率为后向概率，记作:
$$
\beta_{t}(i)=P\left(o_{t+1}, o_{t+2}, \cdots, o_{T} | i_{t}=q_{i}, \lambda\right)
$$
**下面要计算观测序列概率P（O∣λ）。**
初值:
$$
\beta_{T}(i)=1, \quad i=1,2, \cdots, N
$$
递推，对t=1, 2,...,T-1，有:
$$
\beta_{t}(i)=\sum_{j=1}^{N} a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j), \quad i=1,2, \cdots, N
$$
终止:
$$
P(O | \lambda)=\sum_{i=1}^{N} \pi_{i} b_{i}\left(o_{1}\right) \beta_{1}(i)
$$
**利用前向概率和后向概率的定义可以将观测序列概率P（O∣λ）统一写成:**
$$
P(O | \lambda)=\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j), \quad t=1,2, \cdots, T-1
$$
此式当t=1和t=T-1时分别为前向算法和后向算法的终止公式。
### 一些概率值与期望的计算
**利用前向概率和后向慨率，可以得到关于单个状态和两个状态概率的计算公式。**
1. 给定模型λ和观测O，在时刻t处于状态qi的概率，记为
$$
\gamma_{t}(i)=P\left(i_{t}=q_{i} | O, \lambda\right)
$$
可以通过前向后向概率计算。即:
$$
\gamma_{t}(i)=P\left(i_{t}=q_{i} | O, \lambda\right)=\frac{P\left(i_{t}=q_{i}, O | \lambda\right)}{P(O | \lambda)}
$$
由前向概率αt（i）和后向概率βt（i）定义可知:
$$
\alpha_{t}(i) \beta_{t}(i)=P\left(i_{t}=q_{i}, O | \lambda\right)
$$
故有:
$$
\gamma_{t}(i)=\frac{\alpha_{t}(i) \beta_{t}(i)}{P(O | \lambda)}=\frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}
$$
2. 给定模型A和观测序列O，在时刻t处于状态qi且在时刻t+1处于状态qj的概率，记为
$$
\xi_{t}(i, j)=P\left(i_{t}=q_{i}, i_{t+1}=q_{j} | O, \lambda\right)
$$
可以通过前向后向概率计算:
$$
\xi_{i}(i, j)=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}{P(O | \lambda)}=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}{\sum_{i=1}^{N} \sum_{j=1}^{N} P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)}
$$
又
$$
P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O | \lambda\right)=\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)
$$
于是化简得
$$
\xi_{t}(i, j)=\frac{\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)}
$$
3. 将γt（i）和ξt（i,j）对各个时刻t求和，可以得到一些有用的期望值:
  在观测O下状态i出现的期望值:
$$
\sum_{t=1}^{T} \gamma_{t}(i)
$$
在观刻O下由状态i转移的期望值:
$$
\sum_{t=1}^{T-1} \gamma_{t}(i)
$$
在观测O下由状态i转移到状态j的期望值:
$$
\sum_{i=1}^{T-1} \xi_{t}(i, j)
$$

##  学习问题:监督学习方法和非监督学习方法（Baum-Welch算法）
已知观测序列O，估计模型λ的参数，使得在该模型下观测序列概率P（O|λ）最大。即用极大似然估计的方法估计参数。
### 监督学习方法
假设已给训练数据集包含S个长度相同的观测序列和对应的状态序列
$$
(\left(O_{1}, I_{1}\right),\left(O_{2}, I_{2}\right), \cdots,\left(O_{S}, I_{S}\right))
$$
下面利用极大似然估计法来估计隐马尔可夫模型的参数。
1. 转移概率aij的估计:
  设样本中时刻t处于状态i时刻t+1转移到状态j的频数为Aij，那么状态转移概率aij的估计是
$$
\hat a_{i j}=\frac{A_{j j}}{\sum_{j=1}^{N} A_{i j}}, \quad i=1,2, \cdots, N ; j=1,2, \cdots, N
$$
2. 观测概率bj（k）的估计:
  设样本中状态为j并观测为k的频数是Bjk，那么状态为j观测为k的概率bj（k）的估计是:
$$
\hat b_{j}(k)=\frac{B_{j k}}{\sum_{k=1}^{M} B_{j k}}, \quad j=1,2, \cdots, N_{i} \quad k=1,2, \cdots, M
$$
3. 初始状态概率π的估计πi为S个样本中初始状态为qi的频率。

由于监督学习需要使用训练数据， 而人工标注训练数据往往代价很高，有时就会利用非监督学习的方法。
### 非监督学习方法——Baum-Welch算法
由于监督学习需要大量的标注数据，需要耗费很多的人力物力，因此，有时会采用非监督学习方法来进行参数估计。假设给定训练数据集只包含S个长度为T的观测序列而没有对应的状态序列
$$
(O_{1}, O_{2}, \cdots, O_{s})
$$
我们的目标是学习隐马尔可夫模型λ=（A,B,π）的参数。我们将观测序列数据看作观测数据O，状态序列数据看作不可观测的隐数据I，那么隐马尔可夫模型实际上是一个含有隐变量的概率模型:
$$
P(O | \lambda)=\sum_{I} P(O | I, \lambda) P(I | \lambda)
$$
它的参数学习可以由EM算法实现。
1. 确定完全数据的对数似然函数:
  所有观测数据写成:
$$
O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)
$$
所有隐数据写成:
$$
I=\left(i_{1}, i_{2}, \cdots, i_{T}\right)
$$
完全数据是:
$$
(O, I)=\left(o_{1}, o_{2}, \cdots, o_{T}, i_{1}, i_{2}, \cdots, i_{T}\right)
$$
完全数据的对数似然函数是:
$$
\log P(O, I | \lambda)
$$
2. EM算法的E步:求Q函数
$$
Q(\lambda, \overline{\lambda})=\sum_{I} \log P(O, I | \lambda) P(O, I | \overline{\lambda})
$$
左边等式中第一个λ是要极大化的隐马尔可夫模型参数，第二个λ是隐马尔可夫模型参数的当前估计值。
$$
P(O, I | \lambda)=\pi_{i_{1}} b_{i_{1}}\left(o_{1}\right) a_{i_{1} i_{2}} b_{i_{2}}\left(o_{2}\right) \cdots a_{i_{1-1} i_{T}} b_{i_{T}}\left(o_{T}\right)
$$
于是函数Q可以写成:
$$
Q(\lambda, \overline{\lambda})=\sum_{I} \log \pi_{i1} P(O, I | \overline{\lambda}) \\\\
+\sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i, t+1}\right) P(O, I | \overline{\lambda})+\sum_{I}\left(\sum_{t=1}^{T} \log b_{i_{i}}\left(o_{t}\right)\right) P(O, I | \overline{\lambda})
$$
式中求和都是对所有训练数据的序列总长度T进行的。
3. EM 算法的M步:极大化Q函数，求模型参数A、B、π。
  由于要极大化的参数在上式中单独地出现在3个项中，所以只需对各项分别极大化。
  第一项可写为:
$$
\sum_{I} \log \pi_{i_{0}} P(O, I | \overline{\lambda})=\sum_{i=1}^{N} \log \pi_{i} P\left(O, i_{1}=i | \overline{\lambda}\right)
$$
πi满足约束条件:
$$
\sum_{i=1}^{N} \pi_{i}=1
$$
利用拉格朗日乘子法，写出拉格朗日函数:
$$
\sum_{i=1}^{N} \log \pi_{i} P\left(O, i_{1}=i | \overline{\lambda}\right)+\gamma\left(\sum_{i=1}^{N} \pi_{i}-1\right)
$$
对其求偏导数并令结果为0:
$$
\frac{\partial}{\partial \pi_{i}}\left[\sum_{i=1}^{N} \log \pi_{i} P\left(O, i_{1}=i | \overline{\lambda}\right)+\gamma\left(\sum_{i=1}^{N} \pi_{i}-1\right)\right]=0
$$
得:
$$
P\left(O, i_{1}=i | \overline{\lambda}\right)+\gamma \pi_{i}=0
$$
对i求和得到γ:
$$
\gamma=-P(O | \overline{\lambda})
$$
代回偏导数为0的式子中，得
$$
\pi_{i}=\frac{P\left(O, i_{1}=i | \overline{\lambda}\right)}{P(O | \overline{\lambda})}
$$
第二项可写为:
$$
\sum_{I}\left(\sum_{t=1}^{T-1} \log a_{i_{t}i_{t+1}}\right) P(O, I | \overline{\lambda})=\sum_{i=1}^{N} \sum_{j=1}^{N} \sum_{t=1}^{T-1} \log a_{i j} P\left(O, i_{t}=i, i_{t+1}=j | \overline{\lambda}\right)
$$
类似第一项，应用具有约束条件
$$
\sum_{j=1}^{N} a_{i j}=1
$$
的拉格朗日乘了法可以求出
$$
a_{i j}=\frac{\sum_{i=1}^{T-1} P\left(O, i_{t}=i, i_{t+1}=j | \overline{\lambda}\right)}{\sum_{t=1}^{T-1} P\left(O, i_{t}=i | \overline{\lambda}\right)}
$$
第三项可写为:
$$
\sum_{I}\left(\sum_{t=1}^{T} \log b_{i_{t}}\left(o_{t}\right)\right) P(O, I | \overline{\lambda})=\sum_{j=1}^{N} \sum_{t=1}^{T} \log b_{j}\left(o_{t}\right) P\left(O, i_{t}=j | \overline{\lambda}\right)
$$
同样用拉格朗日乘子法，约束条件是
$$
\sum_{k=1}^{M} b_{j}(k)=1
$$
注意只有在ot=vk时bj（ot）对bj（k）的偏导数才不为0，以I（ot=vk）表示。求得:
$$
b_{j}(k)=\frac{\sum_{t=1}^{T} P\left(O, i_{t}=j | \overline{\lambda}\right) I\left(o_{t}=v_{k}\right)}{\sum_{t=1}^{T} P\left(O, i_{t}=j | \overline{\lambda}\right)}
$$
将上面第三步中三项最终推出的公式中的各概率分别用γt（i），ξt（i,j）表示，则可将相应的公式写成:
$$
a_{i j}=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)}
$$
$$
b_{j}(k)=\frac{\sum_{t=1,o_{t}=v_{k}}^{T} \gamma_{t}(j)}{\sum_{t=1}^{T} \gamma_{t}(j)}
$$
$$
\pi_{i}=\gamma_{1}(i)
$$
上面三式就是Baum-Welch算法。


**Baum-Welch算法的流程如下:**
* 初始化，对n=0，选取aij（0），bj（k）（0），πi（0），得到模型
$$
\lambda^{(0)}=\left(A^{(0)}, B^{(0)}, \pi^{(0)}\right)
$$
* 递推。对n=1,2,...，
$$
a_{i j}^{(n+1)}=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)}
$$
$$
b_{j}(k)^{(n+1)}=\frac{\sum_{t=1, o_{t}=v_{k}}^{T} \gamma_{t}(j)}{\sum_{t=1}^{T} \gamma_{t}(j)}
$$
$$
\pi_{i}^{(n+1)}=\gamma_{1}(i)
$$
右端各值按
$$
O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)
$$
$$
\lambda^{(n)}=\left(A^{(n)}, B^{(n)}, \pi^{(n)}\right)
$$
计算。
* 终止。得到模型参数:
$$
\lambda^{(n+1)}=\left(A^{(n+1)}, B^{(n+1)}, \pi^{(n+1)}\right)
$$

## 预测问题（解码问题）:近似算法和维特比（Viterbi）算法
已知模型λ和观测序列O，求对给定观测序列条件概率P（I|O）最大的状态序列。即给定观测序列，求最有可能的对应的状态序列。
### 近似算法
近似算法的思想是，在每个时刻t选择在该时刻最有可能出现的状态it，从而得到一个状态序列
$$
I^{\ast}=\left(i_{1}^{\ast}, i_{2}^{\ast}, \cdots, i_{T}^{\ast}\right)
$$
将它作为预测的结果。
给定隐马尔可夫模型λ和观测序列O，在时刻t处于状态qi的概率为：
$$
\gamma_{t}(i)=\frac{\alpha_{t}(i) \beta_{t}(i)}{P(O | \lambda)}=\frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}
$$
在每一时刻t最有可能的状态it*是
$$
i_{t}^{\ast}=\arg \max_{1 \leqslant i \leqslant N}\left[\gamma_{t}(i)\right], \quad t=1,2, \cdots, T
$$
从而得到状态序列I。
虽然近似计算思想简单，但是预测的序列可能有实际不发生的部分，即有可能出现转移概率为0的相邻状态，没法保证整体上的状态序列是最有可能的。
### 维特比（Viterbi）算法
维特比算法则通过动态规划求概率最大的路径（最优路径），这时每一条路径即对应着一个状态序列。维特比算法从时刻t=1开始，递推地计算在时刻t状态为i的各条部分路径的最大概率，直到得到时刻t=T状态为i的各条路径的最大概率，时刻t=T的最大概率记为最优路径的概率P，最优路径的终结点iT也同时得到，之后，从终结点开始，由后向前逐步求得结点
$$
i_{T-1}^{\ast}, \cdots, i_{1}^{\ast}
$$
最终得到最优状态序列（最优路径）:
$$
I^{\ast}=\left(i_{1}^{\ast}, i_{2}^{\ast}, \cdots, i_{T}^{*}\right)
$$
首先定义两个变量δ和ψ，定义在时刻t状态为i的所有单个路径
$$
\left(i_{1}, i_{2}, \cdots, i_{t}\right)
$$
中概率最大值为
$$
\delta_{t}(i)=\max _{i_{1}, i_{2}, \cdots, t-1} P\left(i_{t}=i, i_{t-1}, \cdots, i_{1}, o_{t}, \cdots, o_{1} | \lambda\right), \quad i=1,2, \cdots, N
$$
由定义可得δ的递推公式：
$$
\delta_{t+1}(i)=\max_{i_{1}, i_{2}, \cdots, i_{t}} P\left(i_{t+1}=i, i_{t}, \cdots, i_{1}, o_{t+1}, \cdots, o_{1} | \lambda\right) \\\\
=\max_{1 \leqslant j \leqslant N}\left[\delta_{t}(j) a_{j i}\right] b_{i}\left(o_{t+1}\right), \quad i=1,2, \cdots, N ; t=1,2, \cdots, T-1
$$
定义在时刻t状态为i的所有单个路径
$$
\left(i_{1}, i_{2}, \cdots, i_{t-1}, i\right)
$$
中概率最大的路径的第t−1个结点为:
$$
\psi_{t}(i)=\arg \max_{1 \leqslant j \leqslant N}\left[\delta_{t-1}(j) a_{j i}\right], \quad i=1,2, \cdots, N
$$
**维特比算法流程如下:**
* 输入模型λ和观测O；
* 初始化
$$
\delta_{1}(i)=\pi_{i} b_{i}\left(o_{1}\right), \qquad i=1,2, \cdots, N \\\\
\psi_{1}(i)=0, \qquad i=1,2, \cdots, N
$$
* 递推，对t=2,3,···,T
$$
\delta_{t}(i)=\max_{1 \leq j \leq N}\left[\delta_{t-1}(j) a_{j i}\right] b_{i}\left(o_{t}\right), \quad i=1,2, \cdots, N \\\\
\psi_{t}(i)=\arg \max_{1 \leqslant j \leqslant N}\left[\delta_{t-1}(j) a_{j i}\right], \quad i=1,2, \cdots, N
$$
* 终止
$$
P^{\ast}=\max_{1 \leq i \leqslant N} \delta_{T}(i) \\\\
i_{T}^{\ast}=\arg \max_{1 \leq i \leqslant N}\left[\delta_{T}(i)\right]
$$
* 最终路径回溯，对t=T-l，T-2，...，1
$$
i_{t}^{\ast}=\psi_{t+1}\left(i_{t+1}^{\ast}\right)
$$
得到最优路径
$$
I^{\ast}=\left(i_{1}^{\ast}, i_{2}^{\ast}, \cdots, i_{T}^{\ast}\right)
$$

**计算举例:**
假设隐马尔可夫模型为λ=（A,B,π）:
$$
A=\left[\begin{array}{lll}{0.5} & {0.2} & {0.3} \\\\ {0.3} & {0.5} & {0.2} \\\\ {0.2} & {0.3} & {0.5}\end{array}\right]
$$
$$
B=\left[\begin{array}{ll}{0.5} & {0.5} \\\\ {0.4} & {0.6} \\\\ {0.7} & {0.3}\end{array}\right]
$$
$$
\pi=(0.2,0.4,0.4)^{\mathrm{T}}
$$
已知观测序列O=（红，白，红），试求最优状态序列。
初始化，t=1时，对每个状态i，i=1，2，3，求状态为i观测o1为红的概率:
$$
\delta_{1}(i)=\pi_{i} b_{i}\left(o_{1}\right)=\pi_{i} b_{i}(红), \quad i=1,2,3
$$
代入数据，得
$$
\delta_{1}(1)=0.10, \quad \delta_{1}(2)=0.16, \quad \delta_{1}(3)=0.28
$$
记
$$
\psi_{1}(i)=0, \quad i=1,2,3
$$
在t=2时，对每个状态i，i=1，2，3，求在t=1时状态为j观测为红并在t=2时状态为i观测o2为白的路径的最大概率:
$$
\delta_{2}(i)=\max_{1 \leq j \leq 3}\left[\delta_{1}(j) a_{j i}\right] b_{i}\left(o_{2}\right)
$$
同时，对每个状态i，i=1，2，3，记录慨率最大路径的前一个状态j:
$$
\psi_{2}(i)=\arg \max_{1 \leqslant j \leqslant 3}\left[\delta_{1}(j) a_{j i}\right], \quad i=1,2,3
$$
代入实际数据，得
$$
 \delta_{2}(1) =\max_{1 \leqslant j \leqslant 3}\left[\delta_{1}(j) a_{j 1}\right] b_{1}\left(o_{2}\right) =\max_{j}(0.10 \times 0.5,0.16 \times 0.3,0.28 \times 0.2) \times 0.5=0.028 
$$
$$
\psi_{2}(1)=3 \\\\
\delta_{2}(2)=0.0504, \quad \psi_{2}(2)=3 \\\\
\delta_{2}(3)=0.042, \quad \psi_{2}(3)=3
$$
同样，在t=3时:
$$
\delta_{3}(i)=\max_{1 \leqslant j \leqslant 3}\left[\delta_{2}(j) a_{j i}\right] b_{i}\left(o_{3}\right)
$$
$$
\psi_{3}(i)=\arg \max_{1 \leqslant j \leqslant 3}\left[\delta_{2}(j) a_{j i}\right]
$$
$$
\delta_{3}(1)=0.00756, \quad \psi_{3}(1)=2 \\\\ \delta_{3}(2)=0.01008, \quad \psi_{3}(2)=2 \\\\ \delta_{3}(3)=0.0147, \quad \psi_{3}(3)=3
$$
以P表示最优路径的概率，则
$$
P^{\ast}=\max_{1 \leqslant i \leqslant 3} \delta_{3}(i)=0.0147
$$
最优路径的终点:
$$
i_{3}^{\ast}=\arg \max_{i}\left[\delta_{3}(i)\right]=3
$$
由最优路径的终点，逆向寻找前面的路径点:
在t=2时
$$
i_{2}^{\ast}=\psi_{3}\left(i_{3}^{\ast}\right)=\psi_{3}(3)=3
$$
在t=1时
$$
i_{1}^{\ast}=\psi_{2}\left(i_{2}^{\ast}\right)=\psi_{2}(3)=3
$$
于是求得最优路径，即最优状态序列:
$$
I^{*}=\left(i_{1}^{\ast}, i_{2}^{\ast}, i_{3}^{\ast}\right)=(3,3,3)
$$
