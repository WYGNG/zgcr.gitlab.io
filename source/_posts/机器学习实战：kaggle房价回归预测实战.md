---
title: 机器学习实战：kaggle房价回归预测实战
date: 2019-03-27 18:06:32
tags:
- 机器学习算法实践
categories:
- 机器学习算法实践
---

**数据和特征决定了机器学习的上限,而模型和算法只是逼近这个上限而已。**
# 特征工程sklearn实践
这里主要以iris数据集为例，展示了数据预处理方法、三大类特征选择方法、降维方法。
**数据预处理方法:**
* z_score标准化；
* min_max标准化；
* 数据单位向量化；
* 数据二值化；
* 数据转为one_hot编码形式；
* 缺失值填充；
* 特征多项式变换；
* 特征使用自定义函数变换。

**三大类特征选择方法:**
* (Filter)过滤法:特征打分，按阈值保留特征。包括方差选择特征法、卡方检验法、皮尔森相关系数法；
* (Wrapper)包装法:根据目标函数(通常是预测效果评分)，每次选择若干特征，或者排除若干特征。包括递归特征消除法等。
* (Embedded)嵌入法:使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。包括基于惩罚项的特征选择法和基于树模型的特征选择法。

**降维方法:**
* PCA；
* LDA。

**代码实现如下:**
```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 以iris数据集为例
iris = load_iris()
x, y = iris.data, iris.target
# 训练集与测试集7:3划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train[0])

# 数据z_score标准化,x减均值再除以标准差
ss = StandardScaler()
x_z_score = ss.fit_transform(x_train)
# print(x_z_score[0])

# 数据min_max标准化,(x-xmin)/(xmax-xmin)
mm = MinMaxScaler()
x_min_max = mm.fit_transform(x_train)
# print(x_min_max[0])

# 数据单位化,norm可以为l1、l2或max，默认为l2
# 为l1时，样本各个特征值除以各个特征值的绝对值之和;为l2时，样本各个特征值除以各个特征值的平方之和;为max时，样本各个特征值除以样本中特征值最大的值
nm = Normalizer(norm="l2")
x_normalize = nm.fit_transform(x_train)
# print(x_normalize[0])

# 数据二值化，设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
x_binary = Binarizer(3, ).fit_transform(x_train)
# print(x_binary[0:2])

# 将特征转为one_hot编码(也可用pandas.get_dummies函数)
ohe = OneHotEncoder(categories='auto', sparse=False)
x_one_hot = ohe.fit_transform([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# print(x_one_hot[0])

# 填充缺失值(也可用pandas.fillna函数),用均值填充平均值
# iris没有缺失值数据行,这里添加一行缺失值
x_missing = np.append(iris.data, [["Nan", "Nan", "Nan", "Nan"]], axis=0)
# print(x_missing[-1])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x_fill_nan = imp.fit_transform(x_missing)
# print(x_fill_nan[-1])

# 多项式变换(对行变量处理)
# 假如一个输入样本是２维的,形式如[a,b] ,则二阶多项式的特征集包括(a+b)(a+b)合并后的每一项
# iris数据集有四项特征,(a+b+c+d)(a+b+c+d)=a^2+2ab+2ac+2ad+b^2+2bc+2bd+c^2+2cd+d^2,再加上本身a,b,c,d四项特征和1这项特征,共15项
# interaction_only=False表示有a平方和b平方这样的项,include_bias=True表示有第一项1
pf = PolynomialFeatures(2, interaction_only=False, include_bias=True)
x_pf = pf.fit_transform(x)
# print(x_pf[0])

# 自定义函数变换特征,以log函数为例,loglp即ln(x+1)
ft = FunctionTransformer(np.log1p, validate=False)
x_log = ft.fit_transform(x)
# print(x_log[0])

# (Filter)过滤法主要思想:
# 按照发散性或者相关性对各个特征进行对每一维的特征“打分”
# 即给每一维的特征赋予权重，这样的权重就代表着该维特征的重要性，设定阈值或者待选择阈值的个数，选择特征
# (Filter)过滤法包括方差选择特征法、卡方检验法、皮尔森相关系数法

# 用方差选择特征,选择方差大于阈值的特征
vt = VarianceThreshold(threshold=0.3)
x_variance_select = vt.fit_transform(x_train)
# print(x_variance_select[0])

# 通过卡方检验,选择K个与标签最相关的特征,k即选取得分最大的前topk个特征
# 卡方检验用来找出特征对类别的相关性
# 我们总是假设H0:观察频数与期望频数没有差别,即某个特征对类别的频数没有影响
# 举例:
#                  体重下降 体重未下降 合计 体重下降率
# 吃晚饭组          123     467        590  20.85%
# 不吃晚饭组        45      106        151  29.80%
# 合计              168     573        741  22.67%
# 这里类别有两个,即体重下降和体重不下降;特征也有两个,即吃晚饭和不吃晚饭
# 建立假设检验
# H0：r1＝r2，不吃晚饭对体重下降没有影响，即吃不吃晚饭的体重下降率相等；
# H1：r1≠r2，不吃晚饭对体重下降有显著影响，即吃不吃晚饭的体重下降率不相等,α=0.05
# 如果不吃饭玩对体重下降没有影响,即假设H0,查表知道总体样本中体重下降率为22.67%
# 则理论值为
#                  体重下降 体重未下降 合计
# 吃晚饭组         133.765  456.234    590
# 不吃晚饭组       34.2348  116.765    151
# 合计             168      573        741
# 如果不吃饭玩与体重下降真的是独立无关的,那么四格表里的理论值和实际值差别应该会很小
# 卡方检验计算公式:x^2=对所有类别求和:(该类别样本数-该类别理论上的样本数)^2/该类别理论上的样本数
# 于是上面例子的卡方值为:
# 卡方值=5.498
# 求自由度:理论频次项加观察频次项=2,类别数k,自由度=(2-1)*(k-1),显然上面例子的自由度为1
# 根据自由度和分位数查卡方分布表,自由度为1,分位数为α=0.05,查表得到值3.84
# 而计算得卡方值5.498＞3.84,查表可看出α在0.02到0.01之间,所以能够以95%的概率拒绝H0假设(否定HO假设),可以认为不吃晚饭对体重下降有显著影响。
sk = SelectKBest(chi2, k=2)
# chi2,卡方统计量，X中特征取值必须非负,卡方检验用来测度随机变量之间的依赖关系。
# 通过卡方检验得到的特征之间是最可能独立的随机变量(卡方值越小变量越独立),因此这些特征的区分度很高。
x_kbest_select = sk.fit_transform(x_train, y_train)
# 每个特征的卡方值,卡方值越小得分越高
# print(sk.pvalues_)
# 每个特征的得分
# print(sk.scores_)
# print(x_kbest_select[0])

# Pearson相关系数(适用于回归问题)
# 相关系数是两个随机变量的协方差与两个随机变量的标准差之积的比值
# 皮尔森相关系数方法衡量的是变量之间的线性相关性,结果的取值区间为[-1，1]，-1表示完全的负相关，+1表示完全的正相关，0表示没有线性相关。
# Pearson相关系数的一个明显缺陷是，作为特征排序机制，他只对线性关系敏感
# 如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0
# Scipy的pearsonr方法能够同时计算相关系数和p-value
# 固定种子
np.random.seed(0)
x = np.random.normal(0, 1, 300)
# pearsonr(x, y)的输入为特征矩阵和目标向量
# np.random.normal(0, 1, 100) 创建100个均值为0，方差为1的高斯随机数
# 输出为二元组(sorce, p-value)的数组
print("低噪音时:", pearsonr(x, x + np.random.normal(0, 1, 300)))
print("高噪音时:", pearsonr(x, x + np.random.normal(0, 10, 300)))


# (Wrapper)包装法选择特征主要思想:
# 根据目标函数(通常是预测效果评分)，每次选择若干特征，或者排除若干特征。
# 也可以将特征子集的选择看作是一个搜索寻优问题，生成不同的组合，对组合进行评价，再与其他的组合进行比较。
# 这样就将子集的选择看作是一个是一个优化问题
# (Wrapper)包装法包括递归特征消除法

# 递归特征消除法来选择特征，这里选择逻辑回归作为基模型，n_features_to_select为保留的特征个数
# 递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练
# 以经典的SVM-RFE算法中来讨论此算法。首先，在每一轮训练过程中，会选择所有特征来进行训练，继而得到了分类的超平面w*x+b=0
# 如果有n个特征，那么SVM-RFE会选择出w中分量的平方值最小的那个序号i对应的特征，将其删除
# 在第二类的时候，特征数就剩下了n-1个，继续用这n-1个特征和输出值来训练SVM
# 同样的，继续去掉w中分量的平方值最小所对应的特征。以此类推，直到剩下的特征数满足我们的要求为止。
# 具体到SVM在sklearn中应用时，可以通过学习器返回的coef_属性或feature_importance_属性来获得每个特征的重要程度
# 然后从当前的特征集合中移除不重要的特征。在特征集合上不断重复上述过程，直到最终达到所需要的特征数量为止。
rfe = RFE(estimator=LogisticRegression(multi_class="auto", solver="liblinear"), n_features_to_select=2)
x_rfe = rfe.fit_transform(x_train, y_train)
# print(x_rfe[0])

# (Embedded)嵌入法选择特征的主要思想:
# 使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。
# 类似于Filter方法，但是是通过训练来确定特征的优劣,挑选出那些对模型的训练有重要意义的属性。
# (Embedded)嵌入法包括基于惩罚项的特征选择法和基于树模型的特征选择法

# 基于惩罚项的特征选择法,这里选择带L1惩罚项的逻辑回归作为基模型,可以输入一个阈值过滤不重要的特征
# 如果使用参数惩罚设置为L1，则使用的阈值为1e-5，否则默认使用mean
sf_lr = SelectFromModel(LogisticRegression(penalty="l1", C=0.1, multi_class="auto", solver="liblinear"), 0.3)
x_sf_lr = sf_lr.fit_transform(x_train, y_train)
# print(x_sf_lr[0])

# 基于树模型的特征选择法,这里选择GBDT模型作为基模型
# GradientBoostingClassifier为GBDT的分类器
sf_gbdt = SelectFromModel(GradientBoostingClassifier())
x_sf_gbdt = sf_gbdt.fit_transform(x_train, y_train)
# print(x_sf_gbdt[0])

# 两种降维方法PCA和LDA。PCA是一种无监督的数据降维方法，LDA是一种有监督的数据降维方法

# PCA降维,参数n_components为降维后的维数
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_train)
# 打印保留的维度的方差
# print(pca.explained_variance_ratio_)
# print(x_pca[0])

# 线性判别分析(LDA)降维,参数n_components为降维后的维数
# PCA是将数据投影到方差最大的几个相互正交的方向上，以期待保留最多的样本信息
# LDA希望投影后相同类别的组内方差小,而组间方差大,使投影后使得同类样本尽可能近,不同类样本尽可能远
lda = LDA(n_components=2)
x_lda = lda.fit_transform(x_train, y_train)
# print(x_lda[0])
```
# 特征工程数据可视化
我们使用kaggle著名的House Prices: Advanced Regression Techniques案例数据集。数据集下载地址:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data 。
**代码实现如下:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

df_train = pd.read_csv('./house_prices/train.csv')
df_test = pd.read_csv('./house_prices/test.csv')
# print(df_train.columns)
print(df_train.shape, df_test.shape)

# 最小价格大于0,数据均有效
# print(df_train['SalePrice'].describe())

# # 价格分布向左偏移,显然不满足正态分布
# sns.distplot(df_train['SalePrice'])
# plt.show()
# plt.close()

# 求价格的偏度和峰度
# 偏度是样本的三阶标准化矩,即((X-μ)/σ)^3的期望
# 峰度是四阶累积量除以二阶累积量的平方,即(X-μ)^4的期望为分子,((X-μ)^2的期望)^2为分母
# print("Skewness:{},Kurtosis:{}".format(df_train['SalePrice'].skew(), df_train['SalePrice'].kurt()))

# # 'GrLivArea'与SalePrice两变量的图像,可以看出这两个变量有明显的线性关系
# data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# plt.show()
# plt.close()

# # 'TotalBsmtSF'与SalePrice两变量的图像,显然这两个变量也有明显的线性关系
# data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
# data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
# plt.show()
# plt.close()

# # 'OverallQual'与SalePrice两变量的图像,销售价格随着商品整体质量的提高而升高
# data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.show()
# plt.close()

# # 'YearBuilt'与SalePrice两变量的图像,显然人们倾向于在新物品上花更多的钱
# data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
# f_2, ax_2 = plt.subplots(figsize=(16, 8))
# fig_2 = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
# fig_2.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)
# plt.show()
# plt.close()

# # 上面我们只是根据这几个特征的含义推断出它们之间可能有关联,通过画图分析我们证实了它们的关系
# # 现在我们要针对有所特征画热力图形式的矩阵来判断任意两两特征之间的相关性
# # data.corr()即相关系数矩阵，给出了任意两个变量之间的相关系数
# # 颜色越浅代表相关性越强，可以看到'TotalBsmtSF'和'1stFlrSF'特征的相关性非常之强
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# # vmax、vmin即热力图颜色取值的最大值和最小值,默认会从data中推导
# # square=True会将单元格设为正方形
# sns.heatmap(corrmat, square=True, ax=ax, cmap='Blues')
# plt.show()
# plt.close()

# # 打印相关系数矩阵,只显示大于0.5的值的项,其他均为NaN
# print(df_train.corr()[df_train.corr() > 0.5])
# # 查看SalePrice对其他变量的相关系数,按从大到小排序
# corr = df_train.corr()['SalePrice']
# print(corr[np.argsort(corr, axis=0)[::-1]])

# # 画出指定特征个数的热力图皮尔森相关系数矩阵,这里指定k=10
# k = 10
# corrmat = df_train.corr()
# # nlargest()的第一个参数就是截取的行数,这里就是从相关系数矩阵中截取相关性最高的前10个特征,第二个参数就是依据的列名
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# # np.corrcoef()计算皮尔逊积矩相关系数
# cm = np.corrcoef(df_train[cols].values.T)
# # 设置字体大小
# sns.set(font_scale=1)
# # annot=True在每个方格中写入数据，square=True会将单元格设为正方形
# # cbar是否在热力图侧边绘制颜色刻度条，默认值是True
# sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
#             xticklabels=cols.values, cmap='Blues')
# plt.show()
# plt.close()

# # 我们还可以一次性画出多个变量中两两变量之间的散点图来查看它们之间有无线性关系
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], height=2)
# plt.show()
# plt.close()

# # 我们也可以一次性对多个变量中画每个变量的数据的直方图来查看数据分布
# # bins指一张图中有几个条形
# df_train.hist(bins=20, figsize=(20, 15))
# plt.show()
# plt.close()

# 缺失值占总样本数百分比统计
total = df_train.isnull().sum().sort_values(ascending=False)
# 缺失值数量/总样本数量,ascending=False降序排列
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data)

# 我们将删除缺失值超过15%的变量,如'PoolQC', 'MiscFeature', 'Alley','Fence','FireplaceQu','LotFrontage'
#  'MasVnrArea'和'MasVnrType'特征与'YearBuilt'和'OverallQual'有很强的相关性,因此我们删除'MasVnrArea'和'MasVnrType'
# Garage的几个变量缺失的都是同样行的数据,因为关于车库的最重要信息是GarageCars,所以我们删除这几个特征
# 同理,我们删除Bsmt几个特征
# Electrical特征只有一项缺失值,我们删除有缺失值这一行数据即可
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# 检查还有无缺失值数据
# print(df_train.isnull().sum().max())
df_train = pd.get_dummies(df_train, drop_first=True)

# # 下面分析异常值,如SalePrice不应该小于0
# # 先把数据进行标准化
# # 新建一列存放标准化后的SalePrice
# # 提取最低的10个数和最高的10个数,可以发现最低的数它们的距离都较近，离0也较近；而最高的10个数里有两个大于7的值，疑似离散点
# sale_price_scaled = StandardScaler().fit_transform((df_train['SalePrice'][:, np.newaxis]).astype(float))
# # y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
# low_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][:10]
# high_range = sale_price_scaled[sale_price_scaled[:, 0].argsort()][-10:]
# # print(low_range)
# # print(high_range)

# # 前面我们已经进行过双变量分析,比如下面两个变量
# # 右下角有两个GrLivArea值很大的点,是异常点
# 右上角靠左也有两个点,这两点就是上面SalePrice中两个大于7的值,注意这两个点不是异常点,先保留，有待观察
# data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0, 800000))
# plt.show()
# plt.close()

# 删除上面两个异常点
df_train_outlier = df_train.sort_values(by='GrLivArea', ascending=False)[:2]
# print(df_train_outlier)
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# # 再观察下面这两个变量,可以看到有一些TotalBsmtSF>3000的离群点,不过我们先保留这些点,先不要删除它们
# data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
# data.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000))
# plt.show()
# plt.close()

# # 下面探究变量是否服从正态分布,显然SalePrice偏离了正态分布的中心
# # displot()集合了matplotlib的hist()与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新用途
# # 核密度估计是在概率论中用来估计未知的密度函数
# # fit=norm即指图上黑色的标准正态分布曲线
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# # 计算概率图的分位数
# # plot如果给出,则根据给出的数据绘制分位数和最小二乘拟合
# res = stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()
# plt.close()

# # 我们把SalePrice值取log对数后再画图看看,现在SalePrice值看起来服从正态分布了
# df_train['SalePrice'] = np.log(df_train['SalePrice'])
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# stats.probplot(df_train['SalePrice'], plot=plt)
# plt.show()
# plt.close()

# # GrLivArea的值也取log对数,这样它的值也从左偏离正态分布变得服从正态分布了
# df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)
# plt.show()
# plt.close()

# # 一组随机变量具备同方差即指线性回归的最小二乘法的残值服从均值为0，方差为σ^2的正态分布，即其干扰项必须服从随机分布
# # 测试两个度量变量的同方差的最佳方法是图形化，先对特征'GrLivArea'和'SalePrice'
# plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
# plt.show()
# plt.close()

# # 再观察'TotalBsmtSF'(只看大于0的值)和'SalePrice',注意这里TotalBsmtSF的非零值已经取过对数
# plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
# plt.show()
# plt.close()
```