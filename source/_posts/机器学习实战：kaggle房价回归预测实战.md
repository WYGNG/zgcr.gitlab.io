---
title: 机器学习实战：kaggle房价回归预测实战
date: 2019-03-27 18:06:32
tags:
- 机器学习算法实践
categories:
- 机器学习算法实践
---

# 项目介绍
项目地址:https://www.kaggle.com/c/house-prices-advanced-regression-techniques 。
该项目数据集中包含79个特征，最后预测出房价特征。
我们进入上面的项目地址，点击Join competition。就可以参与项目竞赛。点击Data项，可以下载训练集train.csv、测试集test.csv、提交样本sample_submission.csv以及一个数据集描述文件data_description.txt。
我们自己对数据集进行预处理，并自己学习一个模型，用来预测官方测试集，最终将预测结果保存为与提交样本sample_submission.csv同样格式的.csv文件，然后点击submit Prediction提交预测结果文件即可(注意提交次数是有限制的)。
# 房价回归Bagging方法预测
这是一个比较基础的预测。首先我们要对数据集进行预处理。主要包括以下几步:
* 为了使房价数据更加平滑和更加服从高斯分布，我们将房价用log1p函数(即log(x+1))处理，并创建一个新的dataframe对象将原始房价数据和处理后的房价数据做直方图对比，可以发现处理后的房价数据更加服从高斯分布；
* 将训练集的房价特征先单独拿出来，然后将训练集79个特征的所有行数据与测试集数据合并，一块儿处理；
* MSSubClass特征是一个类别，但pandas会将其默认处理成数字，我们将其转化为str，然后用get_dummies方法将79个特征中凡是属于类别的数据都转化成one-hot编码格式(比如某个特征中有16个类，那么get_dummies方法会将该特征处理成16列的特征)；
* 将所有缺失值都用对应列的平均值填充；
* 所有非one_hot编码的特征列全部归一化；
* 最后将dataframe对象中所有值转换成numpy array形式。

我们使用岭回归和随机森林作为bagging方法的两个个体学习器，最终预测结果取它们的算数平均值。
分别对岭回归模型的alpha值(正则化项前面系数)和随机森林中决策树所取特征占所有特征得百分比做网格搜索，以负均方误差的平方根作为评价指标，指标越低越好。最终得到alpha为15时最好，决策树特征=0.3(随机取30%的特征构建决策树)时最好。
分别学习alpha=15的岭回归模型和mat_feat=0.3的随机森林模型，分别得到它们对测试集的预测值，最后取算数平均值作为最终预测值。
**代码如下:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# index_col=0即第0号列作为index值,一般这一列都是编号,对预测没有什么作用
train_data = pd.read_csv("./input/train.csv", index_col=0)
test_data = pd.read_csv("./input/test.csv", index_col=0)
# print(train_data.iloc[0].values)
# print(train_data.head())
print(train_data.shape)

# log1p=log(x+1),这行代码用原始数据中的房价和取对数后的房价创建了个新的DataFrame
# 在数据预处理时首先可以对偏度比较大的数据用log1p函数进行转化,使其更加服从高斯分布
# log1p可以避免出现负数结果,如果用log(x+1)后的价格做模型,预测时预测结果也不要忘记反向操作,反向函数就是expm1
# expm1()=exp(x)-1
prices = pd.DataFrame({"price": train_data["SalePrice"], "log(price + 1)": np.log1p(train_data["SalePrice"])})
prices.hist()
plt.show()
plt.close()

# 把train_data的特征SalePrice先pop出来,然后将train_data和test_data拼合起来一起处理其他特征
y_train = np.log1p(train_data.pop("SalePrice"))
# print(y_train.head())
all_data = pd.concat((train_data, test_data), axis=0)
print(all_data.shape)
# MSSubClass值是一个类别,它们之间没有大小关系,但是pandas默认将其处理成数字,我们要将其转换成str
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
# print(all_df["MSSubClass"].value_counts())
# 使用get_dummies方法将MSSubClass转成one-hot形式编码
# prefix可以是字符串或字符串列表,这样就把MSSubClass这列扩展成16列的one-hot编码
# print(pd.get_dummies(all_df["MSSubClass"], prefix="MSSubClass").head())
# 我们可以将all_df中所有表示类别的特征都转成one-hot形式编码
all_dummy_data = pd.get_dummies(all_data)
# [5 rows x 303 columns]
# print(all_dummy_df.head())
# isnull()判断哪一列有缺失值,.sum()统计缺失值有多少个,ascending=False表示降序排列
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
# 求所有列平均值
mean_cols = all_dummy_data.mean()
# print(mean_cols.head(10))
# 缺失值用对应列的均值填充
all_dummy_data = all_dummy_data.fillna(mean_cols)
# 检查一下,现在没有缺失值了
# print(all_dummy_df.isnull().sum().sum())
# 我们还要对数据进行归一化,注意one_hot编码形式的数据不需要归一化
numeric_cols = all_data.columns[all_data.dtypes != "object"]
# 得到需要进行归一化的特征列表
# print(numeric_cols)
# 求均值和标准差,然后进行标准化
numeric_col_means = all_dummy_data.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_data.loc[:, numeric_cols].std()
all_dummy_data.loc[:, numeric_cols] = (all_dummy_data.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
# 数据重新分回训练集和测试集
dummy_train_data = all_dummy_data.loc[train_data.index]
dummy_test_data = all_dummy_data.loc[test_data.index]
print(dummy_train_data.shape, dummy_test_data.shape)

# .values将dataframe对象转换成numpy array形式
X_train = dummy_train_data.values
X_test = dummy_test_data.values

# np.logspace创建等比数列,-3和2表示开始时是10的-3次方,结束时是10的2次方(包含),一共取50个数
alphas = np.logspace(-3, 2, 50)
# 以均方误差作为性能度量
test_scores = []
# 网格搜索来寻找最佳alpha
# 使用ridge regression岭回归方法做预测
for alpha in alphas:
   # alpha值越大则岭回归的正则化项越大,必须是正浮点数,alpha对应于其他线性模型(如Logistic回归或LinearSVC)中的C^-1
   clf = Ridge(alpha)
   # cross_val_score即交叉验证方法,cv=10代表10折,使用上面定义的clf模型,neg_mean_squared_error即负均方误差
   test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
   # 记录每个alpha值使用交叉验证时得到的test_score平均值
   test_scores.append(np.mean(test_score))


plt.plot(alphas, test_scores)
plt.xlabel("alphas")
plt.ylabel("test scores")
plt.show()
plt.close()

# 设定随机森林中的决策树使用的特征占比
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
# 网格搜索来寻找最佳max_feat
# 使用随机森林模型预测
for max_feat in max_features:
   # n_estimators为最大弱学习器的个数(决策树的个数),max_features=max_feat即决策树使用的特征占所有特征的比例
   clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
   # cross_val_score即交叉验证方法,cv=5代表5折,使用上面定义的clf模型,neg_mean_squared_error即负均方误差
   test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring="neg_mean_squared_error"))
   # 记录每个alpha值使用交叉验证时得到的test_score平均值
   test_scores.append(np.mean(test_score))

plt.plot(max_features, test_scores)
plt.xlabel("max_features")
plt.ylabel("test_scores")
plt.show()
plt.close()

# 根据上面网格搜索,使用岭回归时最佳alpha为15,使用随机森林时最佳max_features=0.3
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)

# 分别学习岭回归和随机森林模型
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
# 分别用岭回归和随机森林模型做预测
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
# bagging方法，回归问题最后取两个学习器(岭回归和随机森林)的算术平均值作为预测值
y_final = (y_ridge + y_rf) / 2
# submission_data即最后的预测结果
submission_data = pd.DataFrame(data={"Id": test_data.index, "SalePrice": y_final})
# print(submission_data.head())

submission_data.to_csv("./input/submission_0.csv", index=False)
```
# 房价回归使用XGboost预测
windows下要先在Python中安装xgboost库。在这个网址:https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost 中下载xgboost‑0.82‑cp36‑cp36m‑win_amd64.whl文件(如果你的Python是64位3.6版本)。然后将该文件放到Python文件夹下，在该文件夹下按住shift鼠标右键打开cmd.exe安装。
```python
python -m pip install xgboost-0.82-cp36-cp36m-win_amd64.whl
```
代码如下:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# index_col=0即第0号列作为index值,一般这一列都是编号,对预测没有什么作用
train_data = pd.read_csv("./input/train.csv", index_col=0)
test_data = pd.read_csv("./input/test.csv", index_col=0)
print(train_data.shape, test_data.shape)

# log1p=log(x+1),这行代码用原始数据中的房价和取对数后的房价创建了个新的DataFrame
# 在数据预处理时首先可以对偏度比较大的数据用log1p函数进行转化,使其更加服从高斯分布
# log1p可以避免出现负数结果,如果用log(x+1)后的价格做模型,预测时预测结果也不要忘记反向操作,反向函数就是expm1=exp(x)-1
prices = pd.DataFrame({"price": train_data["SalePrice"], "log(price + 1)": np.log1p(train_data["SalePrice"])})
prices.hist()
plt.show()
plt.close()

# 把train_data的特征SalePrice先pop出来,然后将train_data和test_data拼合起来一起处理其他特征
y_train = np.log1p(train_data.pop("SalePrice"))
all_data = pd.concat((train_data, test_data), axis=0)
# MSSubClass值是一个类别,它们之间没有大小关系,但是pandas默认将其处理成数字,我们要将其转换成str
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
# 使用get_dummies方法可将all_data中特征值是字符串的特征转成one-hot形式编码
# [5 rows x 303 columns]
all_dummy_data = pd.get_dummies(all_data)
# 将所有缺失值用本列平均值填充
mean_cols = all_dummy_data.mean()
all_dummy_data = all_dummy_data.fillna(mean_cols)
# 对所有数据进行归一化,注意one_hot编码形式的数据不需要归一化
numeric_cols = all_data.columns[all_data.dtypes != "object"]
numeric_col_means = all_dummy_data.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_data.loc[:, numeric_cols].std()
all_dummy_data.loc[:, numeric_cols] = (all_dummy_data.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
# 数据重新分回训练集和测试集
dummy_train_data = all_dummy_data.loc[train_data.index]
dummy_test_data = all_dummy_data.loc[test_data.index]
print(dummy_train_data.shape, dummy_test_data.shape)
# .values将dataframe对象转换成numpy array形式
X_train = dummy_train_data.values
X_test = dummy_test_data.values

# # 前面网格搜索已知alpha=15时岭回归模型性能最好
# ridge = Ridge(15)
# # 网格搜索对于回归问题采用bagging方法的最佳param,param即个体学习器数量,个体学习器即采用岭回归模型
# params = [1, 10, 15, 20, 25, 30, 40]
# test_scores = []
# for param in params:
#  clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
#  # cross_val_score即交叉验证方法,cv=10代表10折,使用上面定义的clf模型,neg_mean_squared_error即负均方误差平方根
#  test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
#  test_scores.append(np.mean(test_score))
#
# plt.plot(params, test_scores)
# plt.xlabel("params")
# plt.ylabel("test scores")
# plt.show()
# plt.close()
# # 最佳param为15
#
# params = [10, 15, 20, 25, 30, 40, 50, 60, 70, 100]
# test_scores = []
# # 网格搜索对于回归问题采用bagging方法的最佳param,param即个体学习器数量,个体学习器采用默认的决策树
# for param in params:
#  clf = BaggingRegressor(n_estimators=param)
#  # cross_val_score即交叉验证方法,cv=10代表10折,使用上面定义的clf模型,neg_mean_squared_error即负均方误差平方根
#  test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
#  test_scores.append(np.mean(test_score))
#
# plt.plot(params, test_scores)
# plt.xlabel("params")
# plt.ylabel("test scores")
# plt.show()
# plt.close()
# # 最佳param为50
#
# params = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# test_scores = []
# for param in params:
#  clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
#  test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
#  test_scores.append(np.mean(test_score))
#
# plt.plot(params, test_scores)
# plt.xlabel("params")
# plt.ylabel("test scores")
# plt.show()
# plt.close()
# # 最佳param为40
#
# params = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# test_scores = []
# # 网格搜索对于回归问题采用bagging方法的最佳param,param即个体学习器数量,个体学习器采用默认的决策树
# for param in params:
#  clf = BaggingRegressor(n_estimators=param)
#  test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
#  test_scores.append(np.mean(test_score))
#
# plt.plot(params, test_scores)
# plt.xlabel("params")
# plt.ylabel("test scores")
# plt.show()
# plt.close()
# # 最佳param为50
#
# 上面BaggingRegressor尝试了个体学习器为决策树和岭回归模型的组合
# params尝试了[1, 10, 15, 20, 25, 30, 40],[10, 15, 20, 25, 30, 40, 50, 60, 70, 100]
# params尝试了[10, 15, 20, 25, 30, 35, 40, 45, 50]一共三种组合

params = [1, 2, 3, 4, 5, 6]
test_scores = []
for param in params:
   # xgboost是梯度提升树的实现,XGBRegressor默认使用CART回归树,max_depth是回归树的最大深度,我们这里通过网格搜索找出最佳max_depth
   clf = XGBRegressor(max_depth=param)
   test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
   test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.xlabel("params")
plt.ylabel("test scores")
plt.show()
plt.close()
# 最佳param为5

# 最后学习一个XGBRegressor模型
xgboost_model = XGBRegressor(max_depth=5)
xgboost_model.fit(X_train, y_train)
y_pred = np.expm1(xgboost_model.predict(X_test))
# submission_data即最后的预测结果
submission_data = pd.DataFrame(data={"Id": test_data.index, "SalePrice": y_pred})

submission_data.to_csv("./input/submission_1.csv", index=False)
```
# 特征贡献度可视化:以随机森林模型为例
sklearn的各类模型都有一个.feature_importances_属性，用来表示各个特征对模型的贡献度。我们可以使用argsort对其进行排序，由于argsort排序是从小到大的，因此要用[::-1]进行倒序，得到从大到小的排序，返回值是特征的index下标，我们只需要查询特征名表，根据下标找到对应的特征名就可以将特征贡献度可视化了。
这里以随机森林模型为例，展示特征贡献度可视化。由于处理时使用了one_hot编码等方法，最终产生了303个特征，我们这里只取贡献度最大的前30个特征来可视化。
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# index_col=0即第0号列作为index值,一般这一列都是编号,对预测没有什么作用
train_data = pd.read_csv("./house_prices/train.csv", index_col=0)
test_data = pd.read_csv("./house_prices/test.csv", index_col=0)
# print(train_data.iloc[0].values)
# print(train_data.head())
print(train_data.shape)

# log1p=log(x+1),这行代码用原始数据中的房价和取对数后的房价创建了个新的DataFrame
# 在数据预处理时首先可以对偏度比较大的数据用log1p函数进行转化,使其更加服从高斯分布
# log1p可以避免出现负数结果,如果用log(x+1)后的价格做模型,预测时预测结果也不要忘记反向操作,反向函数就是expm1
# expm1()=exp(x)-1
prices = pd.DataFrame({"price": train_data["SalePrice"], "log(price + 1)": np.log1p(train_data["SalePrice"])})
# prices.hist()
# plt.show()
# plt.close()

# 把train_data的特征SalePrice先pop出来,然后将train_data和test_data拼合起来一起处理其他特征
y_train = np.log1p(train_data.pop("SalePrice"))
# print(y_train.head())
all_data = pd.concat((train_data, test_data), axis=0)
print(all_data.shape)
# MSSubClass值是一个类别,它们之间没有大小关系,但是pandas默认将其处理成数字,我们要将其转换成str
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)
# print(all_df["MSSubClass"].value_counts())
# 使用get_dummies方法将MSSubClass转成one-hot形式编码
# prefix可以是字符串或字符串列表,这样就把MSSubClass这列扩展成16列的one-hot编码
# print(pd.get_dummies(all_df["MSSubClass"], prefix="MSSubClass").head())
# 我们可以将all_df中所有表示类别的特征都转成one-hot形式编码
all_dummy_data = pd.get_dummies(all_data)
# [5 rows x 303 columns]
# print(all_dummy_df.head())
# isnull()判断哪一列有缺失值,.sum()统计缺失值有多少个,ascending=False表示降序排列
# print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
# 求所有列平均值
mean_cols = all_dummy_data.mean()
# print(mean_cols.head(10))
# 缺失值用对应列的均值填充
all_dummy_data = all_dummy_data.fillna(mean_cols)
# 检查一下,现在没有缺失值了
# print(all_dummy_df.isnull().sum().sum())
# 我们还要对数据进行归一化,注意one_hot编码形式的数据不需要归一化
numeric_cols = all_data.columns[all_data.dtypes != "object"]
# 得到需要进行归一化的特征列表
# print(numeric_cols)
# 求均值和标准差,然后进行标准化
numeric_col_means = all_dummy_data.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_data.loc[:, numeric_cols].std()
all_dummy_data.loc[:, numeric_cols] = (all_dummy_data.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
# 数据重新分回训练集和测试集
dummy_train_data = all_dummy_data.loc[train_data.index]
dummy_test_data = all_dummy_data.loc[test_data.index]
print(dummy_train_data.shape, dummy_test_data.shape)

# .values将dataframe对象转换成numpy array形式
X_train = dummy_train_data.values
X_test = dummy_test_data.values

# # 设定随机森林中的决策树使用的特征占比
# max_features = [.1, .3, .5, .7, .9, .99]
# test_scores = []
# # 网格搜索来寻找最佳max_feat
# # 使用随机森林模型预测
# for max_feat in max_features:
#  # n_estimators为最大弱学习器的个数(决策树的个数),max_features=max_feat即决策树使用的特征占所有特征的比例
#  clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#  # cross_val_score即交叉验证方法,cv=5代表5折,使用上面定义的clf模型,neg_mean_squared_error即负均方误差
#  test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring="neg_mean_squared_error"))
#  # 记录每个alpha值使用交叉验证时得到的test_score平均值
#  test_scores.append(np.mean(test_score))
#
# plt.plot(max_features, test_scores)
# plt.xlabel("max_features")
# plt.ylabel("test_scores")
# plt.show()
# plt.close()

# 根据上面网格搜索,使用随机森林时最佳max_features=0.3
rf = RandomForestRegressor(n_estimators=200, max_features=.3)
rf.fit(X_train, y_train)
y_rf = np.expm1(rf.predict(X_test))
submission_data = pd.DataFrame(data={"Id": test_data.index, "SalePrice": y_rf})
# print(submission_data.head())
submission_data.to_csv("./house_prices/submission.csv", index=False)

# 特征贡献度可视化
features = dummy_test_data.columns.values
# print(features)
feature_importances = rf.feature_importances_
# print(feature_importances)
# y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
indices = np.argsort(feature_importances)[::-1]
# 只取贡献度最高的30个特征来作图
show_indices = indices[0:30]
# print(show_indices)
num_show_features = len(show_indices)
# print(num_show_features)
# 将前30个特征重要度以柱状图展示
plt.figure(figsize=(12, 9))
plt.bar(range(num_show_features), feature_importances[show_indices], color="g", align="center")
plt.xticks(range(num_show_features), [features[i] for i in show_indices], rotation='45')
plt.xlim([-1, num_show_features])
plt.show()
plt.close()
# 输出各个特征的贡献度
for i in show_indices:
   print("特征{}贡献度为:{:.3f}".format(features[i], feature_importances[i]))
```