---
title: Kaggle Lending Club Loan Data数据可视化分析与不良贷款预测
date: 2019-06-21 14:14:07
tags:
- 机器学习算法实践
categories:
- 机器学习算法实践
---

# 数据集介绍
该数据集地址: https://www.kaggle.com/wendykan/lending-club-loan-data 。
该数据集是一个单一的CSV文件（loan.csv），解压后总大小1.22G，共有2260668笔Lending Club平台2012-2018年的贷款数据，文件的每一行就是一条贷款数据，每一行中包含了145个特征。
另外，数据集中有一个LCDataDictionary.xlsx文件，专门介绍每个特征的含义。
**我的完整分析代码已经上传到我的kaggle kernel中:**
https://www.kaggle.com/zgcr654321/data-analysis-visualization-and-loan-prediction 。
# 数据可视化分析前的数据预处理
## 引入包和数据集
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from pyecharts import Bar3D, Line3D
from pyecharts import WordCloud
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# Data preprocessing before Data analysis visualization
loan_data = pd.read_csv("loan.csv", low_memory=False)
print(loan_data.shape)


# (2260668, 145)

```
## 对特征缺失值的处理
计算特征缺失值比例的函数:
```python
# calculate the missing value percent of features
def draw_missing_data_table(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    missing_data.reset_index(inplace=True)
    missing_data.rename(columns={"index": "feature_name"}, inplace=True)

    return missing_data
```
保存所有特征缺失值比例的计算结果:
```python
# save missing value percent of features
missing_data_count = draw_missing_data_table(loan_data)
missing_data_count.to_csv("missing_data_count.csv")
missing_data_count = pd.read_csv("missing_data_count.csv", header=0, index_col=0)
missing_data_count = missing_data_count[missing_data_count["Percent"] > 0.0]
print(missing_data_count.head())
#                                  feature_name    Total   Percent
# 0                                          id  2260668  1.000000
# 1                                   member_id  2260668  1.000000
# 2                                         url  2260668  1.000000
# 3  orig_projected_additional_accrued_interest  2252242  0.996273
# 4                         hardship_start_date  2250055  0.995305
```
画缺失值比例图，只画出缺失值比例大于0.03的特征:
```python
# draw a graph of missing value percent of features(percent>0.03)
missing_data_count_show = missing_data_count[missing_data_count["Percent"] > 0.03]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=missing_data_count_show["Percent"], y=missing_data_count_show["feature_name"], ax=ax)
ax.set_title("Missing value percent for each feature", fontsize=16)
ax.set_xlabel("missing percent", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/Missing value percent for each feature.jpg", dpi=200, bbox_inches="tight")
```
删除缺失值比例大于0.15的特征:
```python
# delete features that missing value percent more than 0.15
for index, feature_count_null in missing_data_count.iterrows():
	if feature_count_null["Percent"] > 0.15:
		drop_feature_name = feature_count_null["feature_name"]
		loan_data.drop([drop_feature_name], axis=1, inplace=True)

missing_data_count = missing_data_count[missing_data_count["Percent"] <= 0.15]
print(missing_data_count.head())
#              feature_name   Total   Percent
# 58  mths_since_recent_inq  295435  0.130685
# 59              emp_title  166969  0.073858
# 60       num_tl_120dpd_2m  153657  0.067970
# 61             emp_length  146907  0.064984
# 62     mo_sin_old_il_acct  139071  0.061518
```
对于缺失值比例小于0.04的特征，删除含有这些特征的缺失值的行数据:
```python
# delete rows which contain missing value for features that  missing value precent less than 0.04
for index, feature_count_null in missing_data_count.iterrows():
	if feature_count_null["Percent"] < 0.04:
		drop_feature_name = feature_count_null["feature_name"]
		drop_index = loan_data[loan_data[drop_feature_name].isnull().values == True].index
		loan_data.drop(index=drop_index, axis=0, inplace=True)

print(loan_data.shape)
# (2137073, 87)
```
再计算此时数据集特征的缺失值比例:
```python
# calculate the missing value percent of features again,save missing value percent of features
missing_data_count_2 = draw_missing_data_table(loan_data)
missing_data_count_2.to_csv("missing_data_count_2.csv")
missing_data_count_2 = missing_data_count_2[missing_data_count_2["Percent"] > 0.0]
print(missing_data_count_2)
#             feature_name   Total   Percent
# 0  mths_since_recent_inq  235741  0.110310
# 1              emp_title  154722  0.072399
# 2             emp_length  137175  0.064188
# 3       num_tl_120dpd_2m   81243  0.038016
# 4     mo_sin_old_il_acct   66915  0.031312
```
剩余5个有缺失值的特征，其中3个用0填充，有两个特征先不填充，等到模型训练前再处理。
```python
# fill missing value of mths_since_recent_inq/num_tl_120dpd_2m/mo_sin_old_il_acct by mean value of each feature
# don"t fill emp_title and emp_length
loan_data["mths_since_recent_inq"].fillna(loan_data["mths_since_recent_inq"].mean(), inplace=True)
loan_data["num_tl_120dpd_2m"].fillna(loan_data["num_tl_120dpd_2m"].mean(), inplace=True)
loan_data["mo_sin_old_il_acct"].fillna(loan_data["mo_sin_old_il_acct"].mean(), inplace=True)
# Convert the value of feature:"term" from category to numeric
term_dict = {" 36 months": 36, " 60 months": 60}
loan_data["term"] = loan_data["term"].map(term_dict)
loan_data["term"] = loan_data["term"].astype("float")
```
再计算此时数据集规模和数据集特征的缺失值比例:
```python
# calculate the missing value percent of features the three times,save missing value percent of features
missing_data_count_3 = draw_missing_data_table(loan_data)
missing_data_count_3.to_csv("missing_data_count_3.csv")
missing_data_count_3 = missing_data_count_3[missing_data_count_3["Percent"] > 0.0]
print(missing_data_count_3)
print(loan_data.shape)
#   feature_name   Total   Percent
# 0    emp_title  154722  0.072399
# 1   emp_length  137175  0.064188
# (2137073, 87)
```
## 保存处理好的数据集
```python
# save the dataset after all missing value operation
loan_data.to_csv("loan_clean_data.csv", index=None)
loan_data = pd.read_csv("loan_clean_data.csv", low_memory=False)
print(loan_data.shape)
# (2137073, 87)
```
**注意:**
如果我们想直接进行模型训练和预测，那么在进行模型训练和预测前的数据集预处理前读入loan_clean_data.csv文件即可。

# 数据可视化分析
## 申请贷款金额和实际贷款金额的数据分布
按照数据集中LCDataDictionary.xlsx文件给出的含义，特征loan_amnt和funded_amnt分别代表每笔贷款申请的贷款金额和实际贷款的金额。这两个特征的数值应当是一致的。为了验证一下这个设想是否正确，我们单独提取出这两个特征，分别画出每个特征的数据分布图。我们可以看到特征分布确实是一致的，说明这两个数据分布一致。后面在计算皮尔森相关系数矩阵时我们也发现这两个特征之间的相关系数为+1，这证明了我们之前的设想是正确的。我们注意到大部分贷款额度都在20000以下。
```python
# Data distribution of the loan amount and actual loan amount
sns.set_style("whitegrid")
f_loan, ax_loan = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(loan_data["loan_amnt"], ax=ax_loan[0, 0], color="#F7522F")
sns.violinplot(y=loan_data["loan_amnt"], ax=ax_loan[0, 1], inner="quartile", palette="Reds")
sns.distplot(loan_data["funded_amnt"], ax=ax_loan[1, 0], color="#2F8FF7")
sns.violinplot(y=loan_data["funded_amnt"], ax=ax_loan[1, 1], inner="quartile", palette="Blues")
ax_loan[0, 0].set_title("Loan amount distribution", fontsize=16)
ax_loan[0, 1].set_title("Loan amount distribution", fontsize=16)
ax_loan[1, 0].set_title("Funded amount distribution", fontsize=16)
ax_loan[1, 1].set_title("Funded amount distribution", fontsize=16)
ax_loan[0, 0].set_xlabel("loan amount", fontsize=16)
ax_loan[1, 0].set_xlabel("loan amount", fontsize=16)
ax_loan[0, 1].set_ylabel("loan amount", fontsize=16)
ax_loan[1, 1].set_ylabel("loan amount", fontsize=16)
plt.show()
plt.close()
f_loan.savefig("./pictures/Loan amount and funded amount distribution.jpg", dpi=200, bbox_inches="tight")
```
## 每年贷款笔数直方图与每年贷款总金额直方图
issue_d是贷款发放的时间。这是一个年+月形式的时间特征。我们可以使用正则表达式，将其分解为年和月两个特征。然后我们按照年份统计每年放贷笔数和每年放贷的总金额，画成直方图。可以看到贷款笔数和贷款总金额在2012-2015年逐年攀升，而2015年-2018年上升幅度不大，这可能与公司放贷的经营策略有关。我们注意到两个图的分布十分相似，这说明每笔贷款的平均贷款金额波动不大。
```python
# histogram of annual loan figures and histogram of total amount of annual loan lending
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
loan_year_num = loan_data["year"].value_counts().to_dict()
loan_year_num_pd = pd.DataFrame(list(loan_year_num.items()), columns=["year", "loan times"])
loan_year_num_pd.sort_values("year", inplace=True)
# print(loan_year_num_pd)
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
loan_money_count_per_year = loan_data.groupby("year")["loan_amnt"].sum().to_dict()
loan_money_count_per_year_pd = pd.DataFrame(list(loan_money_count_per_year.items()), columns=["year", "loan_amnt"])
loan_money_count_per_year_pd.sort_values("year", inplace=True)
# print(loan_money_count_per_year_pd)
sns.set_style("whitegrid")
f_loan_per_year, ax_loan_per_year = plt.subplots(1, 2, figsize=(15, 10))
sns.barplot(loan_year_num_pd["year"], loan_year_num_pd["loan times"], ax=ax_loan_per_year[0],
            palette="tab10")
sns.barplot(loan_money_count_per_year_pd["year"], loan_money_count_per_year_pd["loan_amnt"], ax=ax_loan_per_year[1],
            palette="tab10")
ax_loan_per_year[0].set_title("loan times per year", fontsize=16)
ax_loan_per_year[1].set_title("Loan amount per year", fontsize=16)
ax_loan_per_year[0].set_xlabel("year", fontsize=16)
ax_loan_per_year[0].set_ylabel("loan times", fontsize=16)
ax_loan_per_year[1].set_xlabel("year", fontsize=16)
ax_loan_per_year[1].set_ylabel("loan amount", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["year"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 87)
f_loan_per_year.savefig("./pictures/loan times and loan amount per year.jpg", dpi=200, bbox_inches="tight")
```
## 各年各月每笔贷款平均贷款金额3D柱状图和3D折线图
上面我们猜测每笔贷款平均金额的变化不大。现在我们来验证一下。根据上面我们分解得到的年和月两个特征，我们将数据集先按年分组，再按月分组，然后计算每个月中每笔贷款的平均贷款金额，画出每年每月每笔贷款的平均贷款金额的3D柱状图和3D折线图，我们可以发现每笔贷款平均金额确实变化不大。对于2012年的那几个月每笔贷款平均金额偏低大概是由于2012年的前面几个月份没有数据导致的。
通过3D折线图我们可以清楚地看出每笔贷款的平均贷款金额一直变化不大。
```python
# # 各年各月每笔贷款平均贷款金额3D柱状图和3D折线图
# loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
# months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# years_list = ["2012", "2013", "2014", "2015", "2016", "2017", "2018"]
# mean_loan_amnt_per_year_per_month_dict = loan_data.groupby(["month", "year"])["loan_amnt"].mean().to_dict()
# # print(loan_amnt_per_year_per_month_dict)
# max_value = max(mean_loan_amnt_per_year_per_month_dict.values())
# mean_loan_amnt_per_year_per_month_list = []
# for key, value in mean_loan_amnt_per_year_per_month_dict.items():
# 	temp = [key[0], key[1], value]
# 	mean_loan_amnt_per_year_per_month_list.append(temp)
# # print(loan_amnt_per_year_per_month_list)
# mean_loan_amnt_per_year_per_month_bar3d = Bar3D("每月贷款金额3D柱状图", width=1500, height=1000)
# range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43",
#                "#d73027", "#a50026"]
# mean_loan_amnt_per_year_per_month_bar3d.add("mean loan amnt per year per month bar3D", x_axis=months_list, y_axis=years_list,
#                                             data=mean_loan_amnt_per_year_per_month_list,
#                                             is_visualmap=True, visual_range=[0, max_value], visual_range_color=range_color,
#                                             grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# mean_loan_amnt_per_year_per_month_bar3d.render(path="./pictures/mean loan amnt per year per month bar3D.html")
# months_to_num_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
#                       "Oct": 10, "Nov": 11, "Dec": 12}
# for item in mean_loan_amnt_per_year_per_month_list:
# 	item[0], item[1] = months_to_num_dict[item[0]], int(item[1])
# # 画折线图时按照给定数据的输入顺序连线,所以我们要对列表先按月再按年从小到大排序
# mean_loan_amnt_per_year_per_month_list.sort(key=lambda x: x[0])
# mean_loan_amnt_per_year_per_month_list.sort(key=lambda x: x[1])
# colorscale = ["#9ecae1", "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9", "#08519c",
#               "#0b4083", "#08306b"]
# mean_loan_amnt_per_year_per_month_line3d = Line3D("每月贷款金额变化3D折线图", width=1500, height=1000)
# mean_loan_amnt_per_year_per_month_line3d.add("mean loan amnt per year per month line3D",
#                                              data=mean_loan_amnt_per_year_per_month_list,
#                                              yaxis3d_min=2012, yaxis3d_max=2018,
#                                              is_visualmap=True, visual_range=[0, max_value], visual_range_color=colorscale,
#                                              grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# mean_loan_amnt_per_year_per_month_line3d.render(path="./pictures/mean loan amnt per year per month line3D.html")
# loan_data.drop(["month","year"], axis=1, inplace=True)
# print(loan_data.shape)
```
## 各年各月贷款笔数3D柱状图和3D折线图
我们继续数据集先按年分组，再按月分组，然后统计2012-2018年各月的贷款笔数。可以发现自2015年起每月的贷款笔数比起2014年及以前猛然增加许多，另外2016年3月的贷款笔数远远高于其他月，推测是公司在该月有什么重大的举措导致。
我们再将上面的数据用3D折线图的形式画出。我们可以发现每月的放贷数量和月份有明显的相关性，比如每年的7月或8月和10月的放贷笔数较多。
```python
# # 各年各月贷款笔数3D柱状图和3D折线图
# loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
# # print(loan_data["month"].value_counts())
# # print(loan_data["year"].value_counts())
# months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# years_list = ["2012", "2013", "2014", "2015", "2016", "2017", "2018"]
# loan_times_per_year_num_dict = loan_data.groupby(["month"])["year"].value_counts().to_dict()
# max_value = max(loan_times_per_year_num_dict.values())
# loan_times_per_month_per_year_num_list = []
# for key, value in loan_times_per_year_num_dict.items():
# 	temp = [key[0], key[1], value]
# 	loan_times_per_month_per_year_num_list.append(temp)
# # print(loan_per_month_per_year_num_list)
# loan_times_per_month_per_year_bar3d = Bar3D("每月贷款笔数3D柱状图", width=1500, height=1000)
# range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43",
#                "#d73027", "#a50026"]
# loan_times_per_month_per_year_bar3d.add("loan times per month per year bar3D", x_axis=months_list, y_axis=years_list,
#                                         data=loan_times_per_month_per_year_num_list,
#                                         is_visualmap=True, visual_range=[0, max_value], visual_range_color=range_color,
#                                         grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# loan_times_per_month_per_year_bar3d.render(path="./pictures/loan times per month per year bar3D.html")
# months_to_num_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
#                       "Oct": 10, "Nov": 11, "Dec": 12}
# for item in loan_times_per_month_per_year_num_list:
# 	item[0], item[1] = months_to_num_dict[item[0]], int(item[1])
# # 画折线图时按照给定数据的输入顺序连线,所以我们要对列表先按月再按年从小到大排序
# loan_times_per_month_per_year_num_list.sort(key=lambda x: x[0])
# loan_times_per_month_per_year_num_list.sort(key=lambda x: x[1])
# # print(loan_times_per_month_per_year_num_list)
# # loan_times_per_month_per_year_num_list=[[8, 2012, 894], [9, 2012, 5924], [10, 2012, 6192], [11, 2012, 6312], [12, 2012, 6006], [1, 2013, 6814], [2, 2013, 7506], [3, 2013, 8199], [4, 2013, 9354], [5, 2013, 10285], [6, 2013, 10815], [7, 2013, 11816], [8, 2013, 12562], [9, 2013, 12866], [10, 2013, 13858], [11, 2013, 14561], [12, 2013, 14854], [1, 2014, 15470], [2, 2014, 15111], [3, 2014, 16296], [4, 2014, 18829], [5, 2014, 18870], [6, 2014, 16996], [7, 2014, 28948], [8, 2014, 18632], [9, 2014, 10498], [10, 2014, 38244], [11, 2014, 24679], [12, 2014, 10173], [1, 2015, 34691], [2, 2015, 23474], [3, 2015, 25123], [4, 2015, 35052], [5, 2015, 31547], [6, 2015, 28170], [7, 2015, 45446], [8, 2015, 35469], [9, 2015, 28343], [10, 2015, 48064], [11, 2015, 37084], [12, 2015, 43702], [1, 2016, 29548], [2, 2016, 35778], [3, 2016, 56707], [4, 2016, 33093], [5, 2016, 25975], [6, 2016, 30512], [7, 2016, 32575], [8, 2016, 33488], [9, 2016, 26432], [10, 2016, 32318], [11, 2016, 34068], [12, 2016, 35618], [1, 2017, 31435], [2, 2017, 27418], [3, 2017, 36754], [4, 2017, 29270], [5, 2017, 37245], [6, 2017, 37548], [7, 2017, 38784], [8, 2017, 42765], [9, 2017, 38988], [10, 2017, 37434], [11, 2017, 41513], [12, 2017, 37376], [1, 2018, 35718], [2, 2018, 32126], [3, 2018, 38054], [4, 2018, 42177], [5, 2018, 45489], [6, 2018, 40821], [7, 2018, 42372], [8, 2018, 45298], [9, 2018, 38380], [10, 2018, 45540], [11, 2018, 41247], [12, 2018, 39480]]
# colorscale = ["#9ecae1", "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9", "#08519c",
#               "#0b4083", "#08306b"]
# loan_times_per_month_per_year_line3d = Line3D("每月贷款笔数变化3D折线图", width=1500, height=1000)
# loan_times_per_month_per_year_line3d.add("loan times per month per year line3D",
#                                          data=loan_times_per_month_per_year_num_list,
#                                          yaxis3d_min=2012, yaxis3d_max=2018,
#                                          is_visualmap=True, visual_range=[0, max_value], visual_range_color=colorscale,
#                                          grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# loan_times_per_month_per_year_line3d.render(path="./pictures/loan times per month per year line3D.html")
# loan_data.drop(["month","year"], axis=1, inplace=True)
# print(loan_data.shape)
```
## 各州贷款笔数地理坐标系图和直方图
该数据集中的贷款信息来自美国的各个州。特征addr_state表明了每笔贷款申请发生的地点(州)。我们以州为单位将数据集分组并统计每个州的贷款笔数，然后绘制各州贷款笔数的地理坐标系图。我们可以看到加州的贷款笔数是最多的（不愧是美国GDP最高的州）。
```python
# the map of geographical coordinates of each state"s loan figures
# addr_state即申请贷款的人的所属州,是两位代码,可以被plotly识别
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
loan_times_per_state = loan_data["addr_state"].value_counts().to_dict()
loan_times_per_state_pd = pd.DataFrame(list(loan_times_per_state.items()), columns=["state_code", "loan_times"])
loan_times_per_state_pd["state_name"] = None
# print(loan_times_per_state_pd)
for i in range(loan_times_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_times_per_state_pd.ix[i, "state_code"]]
	loan_times_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_times_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Blues"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=True,
             locations=loan_times_per_state_pd["state_code"], z=loan_times_per_state_pd["loan_times"].astype(float),
             locationmode="USA-states", text=loan_times_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="loan times", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="loan times per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="loan times per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/loan times per state map.png", width=2500, height=1500)
```
我们再画出贷款笔数前30名的州。前四名依次为California、Texas、NewYork、Florida。其中California的贷款笔数远远高于其他州。
```python
# Histogram of each state"s loan figures (the top 30 states with the largest number of loans)
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
loan_times = loan_data["addr_state"].value_counts().to_dict()
loan_times_pd = pd.DataFrame(list(loan_times.items()), columns=["state_code", "loan_times"])
loan_times_pd["state_name"] = None
# print(loan_times_pd)
for i in range(loan_times_pd.shape[0]):
	state_name = code_and_name_dict[loan_times_pd.ix[i, "state_code"]]
	loan_times_pd.ix[i, "state_name"] = state_name
# print(loan_times_pd)
loan_times_pd_30 = loan_times_pd[0:30]
loan_times_pd_30.drop(["state_code"], axis=1)
sns.set_style("whitegrid")
f_loan_times_per_state, ax_loan_times_per_state = plt.subplots(figsize=(15, 10))
# # palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_pd_30["loan_times"], loan_times_pd_30["state_name"], ax=ax_loan_times_per_state,
            palette="tab10")
ax_loan_times_per_state.set_title("loan times per state", fontsize=16)
ax_loan_times_per_state.set_xlabel("loan times", fontsize=16)
ax_loan_times_per_state.set_ylabel("state name", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_state.savefig("./pictures/loan times per state bar.jpg", dpi=200, bbox_inches="tight")
```
## 贷款次数最多的前30种职业直方图
特征emp_title为申请贷款时贷款人的职业，我们按该特征分组数据集，计算每个职业的贷款笔数，看看申请贷款最多的职业是哪几个。我们发现教师和经理人的申请贷款次数是最多的，远远超出其他职业。
我们注意到这个特征是个类别型特征，但特征的取值非常多，且每个取值的出现次数相对于总样本量很小，因此这个特征如果变为one_hot编码后其每一列one_hot编码特征的方差都会很小，这对于模型的预测不利。因此在后面模型预测前的预处理时我们删除了该特征。
```python
# histogram of the top 30 profession of loan figures
loan_times_title = loan_data["emp_title"].value_counts().to_dict()
loan_times_title_pd = pd.DataFrame(list(loan_times_title.items()), columns=["title", "loan_times"])
loan_times_title_pd_30 = loan_times_title_pd[0:30]
sns.set_style("whitegrid")
f_loan_times_per_title, ax_loan_times_per_title = plt.subplots(figsize=(15, 10))
# # palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_title_pd_30["loan_times"], loan_times_title_pd_30["title"], ax=ax_loan_times_per_title,
            palette="tab10")
ax_loan_times_per_title.set_title("loan times per title", fontsize=16)
ax_loan_times_per_title.set_xlabel("loan times", fontsize=16)
ax_loan_times_per_title.set_ylabel("title", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_title.savefig("./pictures/loan times per title bar.jpg", dpi=200, bbox_inches="tight")
```
## 工作年限与贷款笔数的直方图
特征emp_length为贷款申请人的工作年限。我们将数据集按工作年限进行分组，统计每个工作年限的贷款笔数。我们可以发现工作10年以上的人申请贷款数目远远超过其他年限。
​	要注意的一点是，该特征虽然是类别型特征的形式，但是其内容却是典型的数值，因此后面模型预测前我们要将其转化为数值型特征。
```python
# histogram of the year of participation in working with loan figures
loan_times_length = loan_data["emp_length"].value_counts().to_dict()
# print(loan_times_length)
# {"10+ years": 713245, "2 years": 192330, "< 1 year": 179177, "3 years": 170699, "1 year": 139017, "5 years": 130985,
# "4 years": 128027, "6 years": 96294, "7 years": 87537, "8 years": 87182, "9 years": 75405}
loan_times_length_pd = pd.DataFrame(list(loan_times_length.items()), columns=["length", "loan_times"])
sns.set_style("whitegrid")
f_loan_times_per_length, ax_loan_times_per_length = plt.subplots(figsize=(15, 10))
# palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_length_pd["length"], loan_times_length_pd["loan_times"], ax=ax_loan_times_per_length,
            palette="Blues_d")
ax_loan_times_per_length.set_title("loan times per length", fontsize=16)
ax_loan_times_per_length.set_xlabel("worked length", fontsize=16)
ax_loan_times_per_length.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_length.savefig("./pictures/loan times per length bar.jpg", dpi=200, bbox_inches="tight")
```
## 用户年收入与贷款笔数直方图
特征annual_inc为申请人的年收入。这是一个数值型特征。我们将收入分成三档:20000以下为low，20000-60000为mid，>60000为high。按照这三档将该数值型特征变为类别型特征，然后将数据集按该特征分组，计算每组的贷款笔数。我们发现年收入为low的人贷款笔数远远小于另外两档，推测是因为银行的审核贷款机制判定大部分年收入为low的人没有能力偿还贷款，因此不批贷款导致。
```python
# histogram of the customer"s annual income with loan figures
# 我们将年收入分为三档:20000以下为low，20000-60000为mid，>60000为high
max_value = loan_data["annual_inc"].max() + 1.0
set_bins = [0.0, 20000.0, 60000.0, max_value]
set_label = ["low", "mid", "high"]
loan_data["income"] = pd.cut(loan_data["annual_inc"], bins=set_bins, labels=set_label)
loan_times_income = loan_data["income"].value_counts().to_dict()
# print(loan_times_income)
# {"high": 1187055, "mid": 912572, "low": 37443}
loan_times_income_pd = pd.DataFrame(list(loan_times_income.items()), columns=["income", "loan_times"])
sns.set_style("whitegrid")
f_loan_times_per_income, ax_loan_times_per_income = plt.subplots(figsize=(15, 10))
# palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_income_pd["income"], loan_times_income_pd["loan_times"], ax=ax_loan_times_per_income,
            palette="muted")
ax_loan_times_per_income.set_title("loan times per income", fontsize=16)
ax_loan_times_per_income.set_xlabel("income", fontsize=16)
ax_loan_times_per_income.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["income"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_times_per_income.savefig("./pictures/loan times per income bar.jpg", dpi=200, bbox_inches="tight")
```
## 优质贷款与不良贷款的比例
特征loan_status为贷款状态，我们用1表示贷款状况良好，用0表示不良贷款，则各种情况如下:
loan_status_dict = {'Fully Paid': 1, 'Current': 1, 'Charged Off': 0, 'Late (31-120 days)': 0,'In Grace Period': 0, 'Late (16-30 days)': 0, 'Default': 0}
按照上面的映射规则处理特征loan_status，这样我们就可以确定一个贷款是优质贷款还是不良贷款。画出总的优质贷款和不良贷款的饼状比例图。然后再按年分组，画出每年优质贷款和不良贷款占总贷款笔数的比例。
从右侧每年优质贷款和不良贷款比例的直方图中我们可以看出自2016年以后放贷笔数持续增长，但不良贷款的数量却明显下降了，这说明公司的风控部门的工作卓有成效。
另外，我们可以发现这个数据集的正负样本不平衡情况很严重，在进行模型拟合和预测时要进行特殊处理。
```python
# The ratio of good loans and bad loans for each year
# print(loan_data["loan_status"].value_counts().to_dict())
# {"Fully Paid": 962556, "Current": 899615, "Charged Off": 241514, "Late (31-120 days)": 21051, "In Grace Period": 8701,
# "Late (16-30 days)": 3607, "Default": 29}
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count"] = loan_data["loan_status_count"].astype("float")
# print(loan_data["loan_status"].value_counts().to_dict())
# {1.0: 1862171, 0.0: 274902}可以看到正负样本不均衡，在后面我们训练模型预测loan_status时需要注意正负样本不平衡的问题
loan_status_count = loan_data["loan_status_count"].value_counts().to_dict()
if 0 not in loan_status_count.keys():
	loan_status_count["0"] = 0.0
count_sum = 0
for key, value in loan_status_count.items():
	count_sum += value
for key, value in loan_status_count.items():
	value = value / count_sum
	loan_status_count[key] = value
loan_status_count_pd = pd.DataFrame(list(loan_status_count.items()), columns=["loan status", "count_percent"])
# print(loan_status_count_pd)
#    loan status  count_percent
# 0          1.0       0.871365
# 1          0.0       0.128635
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
f_loan_status, ax_loan_status = plt.subplots(1, 2, figsize=(15, 10))
labels = "Good loans", "Bad loans"
ax_loan_status[0].pie(loan_status_count_pd["count_percent"], autopct="%1.2f%%", shadow=True,
                      labels=labels, startangle=70)
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count"] = loan_data["loan_status_count"].map(loan_status_dict)
sns.barplot(x=loan_data["year"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count"], hue_order=labels,
            ax=ax_loan_status[1], estimator=lambda x: len(x) / len(loan_data["loan_status_count"]) * 100)
ax_loan_status[0].set_title("good loans and bad loans percent", fontsize=16)
ax_loan_status[0].set_ylabel("Loans percent", fontsize=16)
ax_loan_status[1].set_title("good loans and bad loans percent per year", fontsize=16)
ax_loan_status[1].set_ylabel("Loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count", "year"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status.savefig("./pictures/good loans and bad loans percent per year.jpg", dpi=200,
                      bbox_inches="tight")
```
## 各州不良贷款笔数/比例的地理坐标系图
我们按州为单位统计每个州的不良贷款笔数，并用颜色的深浅来区别。可以看到加州的不良贷款数远远高于其他州，但结合之前的各州总贷款笔数直方图，这可能是由于加州的放贷笔数最多的原因。
```python
# the map of geographical coordinates of each state"s good loan number and bad loan number
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# 为了便于计算坏账数,我们令坏账为1,好帐为0
loan_status_dict = {"Fully Paid": 0, "Current": 0, "Charged Off": 1, "Late (31-120 days)": 1,
                    "In Grace Period": 1, "Late (16-30 days)": 1, "Default": 1}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_2"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_2"] = loan_data["loan_status_count_2"].astype("float")
# print(loan_data["loan_status_count"].value_counts().to_dict())
# {0.0: 1862171, 1.0: 274902}
loan_status_per_state = loan_data.groupby("addr_state")["loan_status_count_2"].sum().to_dict()
# print(loan_status_per_state)
loan_status_per_state_pd = pd.DataFrame(list(loan_status_per_state.items()),
                                        columns=["state_code", "bad_loan_num"])
loan_status_per_state_pd["state_name"] = None
# print(loan_status_per_state_pd)
for i in range(loan_status_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_status_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Hot"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=True,
             locations=loan_status_per_state_pd["state_code"], z=loan_status_per_state_pd["bad_loan_num"],
             locationmode="USA-states", text=loan_status_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="bad loans num", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="bad loans num per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
loan_data.drop(["loan_status_count_2"], axis=1, inplace=True)
print(loan_data.shape)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="bad loans num per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/bad loans num per state map.png", width=2500, height=1500)
```
继续计算每个州的不良贷款占各州总贷款笔数的比例。可以发现前面的推断是正确的，各州之间不良贷款比例的差距并不大。注意有一个州大概是数据特别少或者数据缺失，不良贷款比例接近0。
```python
# the map of geographical coordinates of each state"s good loan ratio and bad loan ratio
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# 为了便于计算坏账数,我们令坏账为1,好帐为0
loan_status_dict = {"Fully Paid": 0, "Current": 0, "Charged Off": 1, "Late (31-120 days)": 1,
                    "In Grace Period": 1, "Late (16-30 days)": 1, "Default": 1}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_3"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_3"] = loan_data["loan_status_count_3"].astype("float")
# print(loan_data["loan_status_count"].value_counts().to_dict())
# {0.0: 1862171, 1.0: 274902}
loan_status_per_state = loan_data.groupby("addr_state")["loan_status_count_3"].sum().to_dict()
loan_status_per_state_pd = pd.DataFrame(list(loan_status_per_state.items()),
                                        columns=["state_code", "bad_loan_percent"])
loan_times_per_state_sum_dict = loan_data["addr_state"].value_counts().to_dict()
loan_status_per_state_pd["state_name"] = None
# print(loan_status_per_state_pd)
for i in range(loan_status_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_status_per_state_pd)
# print(loan_times_per_state_sum_dict)
for i in range(loan_status_per_state_pd.shape[0]):
	per_state_sum = loan_times_per_state_sum_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "bad_loan_percent"] = float(
		loan_status_per_state_pd.ix[i, "bad_loan_percent"]) / per_state_sum
# print(loan_status_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Reds"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=False,
             locations=loan_status_per_state_pd["state_code"], z=loan_status_per_state_pd["bad_loan_percent"],
             locationmode="USA-states", text=loan_status_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="bad loans percent", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="bad loans percent per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
loan_data.drop(["loan_status_count_3"], axis=1, inplace=True)
print(loan_data.shape)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="bad loans percent per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/bad loans percent per state map.png", width=2500, height=1500)
```
## 贷款目的词云图
特征purpose为每笔贷款申请时填写的目的。我们将数据集按该特征分组，然后计算每种purpose的贷款笔数，画成词云图。词云图中文字尺寸越大的目的，代表该目的的贷款笔数越多。我们可以发现贷款目的中debt consolidation(债务合并)这一目的的比例远远超过其他目的。
```python
# loan purpose word cloud map
loan_times_per_purpose_sum_dict = loan_data["purpose"].value_counts().to_dict()
# print(loan_times_per_purpose_sum_dict)
loan_times_per_purpose_sum_pd = pd.DataFrame(list(loan_times_per_purpose_sum_dict.items()),
                                             columns=["purpose", "loan_times"])
# print(loan_times_per_purpose_sum_pd)
wordcloud = WordCloud(width=1500, height=1000)
wordcloud.add("loan purpose word cloud", loan_times_per_purpose_sum_pd["purpose"],
              loan_times_per_purpose_sum_pd["loan_times"], shape="diamond",
              rotate_step=60, word_size_range=[10, 100])
wordcloud.render(path="./pictures/loan purpose word cloud.html")
wordcloud.render(path="./pictures/loan purpose word cloud.pdf")
```
## 各种贷款目的的优质/不良贷款比例
我们将数据集按贷款目的分组，然后计算每种目的中的优质贷款笔数和不良贷款笔数占总贷款笔数的比例。可以发现贷款目的中debt consolidation(债务合并)这一目的的比例远远超过其他目的，同时其不良贷款比例也是最高的。
```python
# the ratio of good loans and bad loans for each loan purpose
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_4"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_4"] = loan_data["loan_status_count_4"].astype("float")
f_loan_status_purpose, ax_loan_status_purpose = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_4"] = loan_data["loan_status_count_4"].map(loan_status_dict)
sns.barplot(x=loan_data["purpose"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_4"], hue_order=labels,
            ax=ax_loan_status_purpose, estimator=lambda x: len(x) / len(loan_data["loan_status_count_4"]) * 100)
ax_loan_status_purpose.set_title("Loan status per purpose percent", fontsize=16)
ax_loan_status_purpose.set_xticklabels(ax_loan_status_purpose.get_xticklabels(), rotation=45)
ax_loan_status_purpose.set_xlabel("purpose", fontsize=16)
ax_loan_status_purpose.set_ylabel("per purpose loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_4"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_purpose.savefig("./pictures/Loan status per purpose percent bar.jpg", dpi=200,
                              bbox_inches="tight")
```
## 优质/不良贷款客户的住房情况比例
将数据集先按优质/不良贷款分组，然后每个分组再按住房情况进行分组。我们可以发现在MORTGAGE（即按揭）这种情况中，优质贷款客户中属于按揭购买住房的比例明显更高，这部分人应当属于优质客户，可以作为放贷的重点发展对象。
```python
# ratio of housing for good loans customer and bad loans customer
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_5"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_5"] = loan_data["loan_status_count_5"].astype("float")
f_loan_status_home, ax_loan_status_home = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_5"] = loan_data["loan_status_count_5"].map(loan_status_dict)
sns.barplot(x=loan_data["home_ownership"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_5"],
            hue_order=labels,
            ax=ax_loan_status_home, estimator=lambda x: len(x) / len(loan_data["loan_status_count_5"]) * 100)
ax_loan_status_home.set_title("Loan status per home percent", fontsize=16)
ax_loan_status_home.set_xlabel("home ownership", fontsize=16)
ax_loan_status_home.set_ylabel("per home ownership loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_5"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_home.savefig("./pictures/Loan status per home ownership percent bar.jpg", dpi=200,
                           bbox_inches="tight")
```
## 按收入分组的不良贷款比例
我们仍按收入将数据集分成三组:20000以下为low，20000-60000为mid，>60000为high。计算每个分组的不良贷款比例。可以发现高收入客户中不良贷款笔数占的高收入客户的总贷款笔数的比例相对更小。
```python
# ratio of good loans and bad loans for each income level
# 我们将年收入分为三档:20000以下为low，20000-60000为mid，>60000为high
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_6"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_6"] = loan_data["loan_status_count_6"].astype("float")
max_value = loan_data["annual_inc"].max() + 1.0
set_bins = [0.0, 20000.0, 60000.0, max_value]
set_label = ["low", "mid", "high"]
loan_data["income"] = pd.cut(loan_data["annual_inc"], bins=set_bins, labels=set_label)
f_loan_status_income, ax_loan_status_income = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_6"] = loan_data["loan_status_count_6"].map(loan_status_dict)
sns.barplot(x=loan_data["income"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_6"], hue_order=labels,
            ax=ax_loan_status_income, estimator=lambda x: len(x) / len(loan_data["loan_status_count_6"]) * 100)
ax_loan_status_income.set_title("Loan status per income percent", fontsize=16)
ax_loan_status_income.set_xlabel("income", fontsize=16)
ax_loan_status_income.set_ylabel("per income loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_6", "income"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_income.savefig("./pictures/Loan status per income percent bar.jpg", dpi=200,
                             bbox_inches="tight")
```
## 各个信用等级的优质/不良贷款比例
特征grade代表每笔贷款的申请人的信用等级。信用等级从高到低为A-G，信用等级较高的客户贷款次数也较多，因为这类人群有能力还款，所以愿意贷款。另外，信用等级较高的客户不良贷款的比例也更小（相对于本信用等级总贷款笔数）。
```python
# the ratio of good loans and bad loans for each credit rating
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status"].map(loan_status_dict)
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status_count_7"].astype("float")
f_loan_status_grade, ax_loan_status_grade = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status_count_7"].map(loan_status_dict)
sns.barplot(x=loan_data_sorted["grade"], y=loan_data_sorted["loan_amnt"], hue=loan_data_sorted["loan_status_count_7"],
            hue_order=labels,
            ax=ax_loan_status_grade, estimator=lambda x: len(x) / len(loan_data_sorted["loan_status_count_7"]) * 100)
ax_loan_status_grade.set_title("Loan status per grade percent", fontsize=16)
ax_loan_status_grade.set_xlabel("grade", fontsize=16)
ax_loan_status_grade.set_ylabel("per grade loans percent", fontsize=16)
plt.show()
plt.close()
print(loan_data.shape)
f_loan_status_grade.savefig("./pictures/Loan status per grade percent bar.jpg", dpi=200,
                            bbox_inches="tight")
```
## 贷款利率的数据分布
特征int_rate代表每笔贷款的利率。我们单独提取出这个特征的所有值，画其数据分布图。我们可以发现这个特征的取值比较符合高斯分布，但中间有些值的离散情况比较严重。
```python
# data distribution of loan interest rates
sns.set_style("whitegrid")
f_int_rate, ax_int_rate = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(loan_data["int_rate"], ax=ax_int_rate[0], color="#2F8FF7")
sns.violinplot(y=loan_data["int_rate"], ax=ax_int_rate[1], inner="quartile", palette="Blues")
ax_int_rate[0].set_title("Int rate distribution", fontsize=16)
ax_int_rate[1].set_title("Int rate distribution", fontsize=16)
ax_int_rate[0].set_xlabel("Int rate", fontsize=16)
ax_int_rate[0].set_ylabel("Int rate", fontsize=16)
plt.show()
plt.close()
f_int_rate.savefig("./pictures/Int rate distribution.jpg", dpi=200, bbox_inches="tight")
```
## 优质/不良贷款在各个信用等级的平均利率
我们将数据集按信用等级分组，每个信用等级再按优质/不良贷款分组，然后计算每个分组的平均利率。我们发现越是高信用客户，其越倾向于利率较低的贷款。另外每个信用等级中不良贷款和优质贷款中的平均利率基本相同。
```python
# average interest rate of good loans and bad loans for each credit ratings
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_8"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_8"] = loan_data["loan_status_count_8"].astype("float")
f_inc_rate_grade, ax_inc_rate_grade = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_8"] = loan_data["loan_status_count_8"].map(loan_status_dict)
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
sns.barplot(x=loan_data_sorted["grade"], y=loan_data_sorted["int_rate"], hue=loan_data_sorted["loan_status_count_8"],
            hue_order=labels, ax=ax_inc_rate_grade, ci=None)
ax_inc_rate_grade.set_title("mean int rate per grade", fontsize=16)
ax_inc_rate_grade.set_xlabel("grade", fontsize=16)
ax_inc_rate_grade.set_ylabel("mean int rate", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_8"], axis=1, inplace=True)
print(loan_data.shape)
f_inc_rate_grade.savefig("./pictures/mean int rate per grade bar.jpg", dpi=200,
                         bbox_inches="tight")

```
## 贷款人每月还款占其总收入比例的数据分布
特征dti为每笔贷款的贷款人每月还款占其总收入比例的数据分布。该数据中有异常值-1.0，且有很大的离散值，我们需要将其先过滤掉然后再画数据分布图。该特征是一个比较完美的高斯分布，这对于我们后面模型拟合前的预处理有很好的指示效果。
需要注意的是，实际上不止dti这一个特征有异常值，在后面模型拟合前对数据集进行预处理时必须考虑异常值对数据集的影响。
```python
# data distribution of the percentage of the lender"s month-repayments divide the lender"s income
# max_value=loan_data["dti"].max()
# min_value=loan_data["dti"].min()
# print(max_value,min_value)
# 999.0 -1.0 这里的数值应当是百分比
# 该数据中有异常值-1.0,且有很大的离散值,我们需要将其先过滤掉,否则图像效果不好
loan_data_dti = loan_data[loan_data["dti"] <= 100.0]
loan_data_dti = loan_data_dti[loan_data_dti["dti"] > 0.0]
sns.set_style("whitegrid")
f_dti, ax_dti = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(loan_data_dti["dti"], ax=ax_dti[0], color="#F7522F")
sns.violinplot(y=loan_data_dti["dti"], ax=ax_dti[1], inner="quartile", palette="Reds")
ax_dti[0].set_title("dti distribution", fontsize=16)
ax_dti[1].set_title("dti distribution", fontsize=16)
ax_dti[0].set_xlabel("dti", fontsize=16)
ax_dti[1].set_ylabel("dti", fontsize=16)
plt.show()
plt.close()
f_dti.savefig("./pictures/dti distribution.jpg", dpi=200, bbox_inches="tight")
```
## 优质/不良贷款的贷款人每月还款占其总收入比例的数据分布
我们再将数据集按优质/不良贷款分组，查看每个分组的贷款人每月还款占其总收入比例的数据分布。看起来两个分组中的高斯分布十分接近，说明优质/不良贷款分组对特征dti的分布没有什么影响。
```python
# data distribution of the percentage of the good lender"s month-repayments divide the lender"s income and the percentage of the bad lender"s month-repayments divide the lender"s income
# 请先过滤异常值和极大离散点,只保留0-200之间的数据
loan_data_dti = loan_data[loan_data["dti"] <= 100.0]
loan_data_dti = loan_data_dti[loan_data_dti["dti"] >= 0.0]
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data_dti["loan_status_count_9"] = loan_data_dti["loan_status"].map(loan_status_dict)
loan_data_dti["loan_status_count_9"] = loan_data_dti["loan_status_count_9"].astype("float")
labels = "Bad loans", "Good loans"
# 取出groupby后的分组结果
loans_dti_per_status = dict(list(loan_data_dti.groupby("loan_status_count_9")["dti"]))
good_loan_dti = pd.DataFrame(loans_dti_per_status[1.0], index=None).reset_index(drop=True)
bad_loan_dti = pd.DataFrame(loans_dti_per_status[0.0], index=None).reset_index(drop=True)
# print(good_loan_dti, bad_loan_dti)
# print(good_loan_dti.shape, bad_loan_dti.shape)
sns.set_style("whitegrid")
f_dti_per_loan_status, ax_dti_per_loan_status = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(good_loan_dti["dti"], ax=ax_dti_per_loan_status[0], color="#2F8FF7")
sns.distplot(bad_loan_dti["dti"], ax=ax_dti_per_loan_status[1], color="#F7522F")
ax_dti_per_loan_status[0].set_title("good loans dti distribution", fontsize=16)
ax_dti_per_loan_status[1].set_title("bad loans dti distribution", fontsize=16)
ax_dti_per_loan_status[0].set_xlabel("dti", fontsize=16)
ax_dti_per_loan_status[1].set_ylabel("dti", fontsize=16)
plt.show()
plt.close()
print(loan_data.shape)
f_dti_per_loan_status.savefig("./pictures/dti distribution per loan status.jpg", dpi=200, bbox_inches="tight")
```
## 优质/不良贷款中短期和长期贷款的比例
贷款期限分为36个月和60个月两种，36个月的短期贷款笔数更多，但短期贷款中不良贷款笔数占短期贷款总笔数的比例更小，说明短期贷款的风险更小。
```python
# ratio of short-term and long-term loans for good loans and bad loans
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_10"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_10"] = loan_data["loan_status_count_10"].astype("float")
f_loan_status_term, ax_loan_status_term = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_10"] = loan_data["loan_status_count_10"].map(loan_status_dict)
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
sns.barplot(x=loan_data_sorted["term"], y=loan_data_sorted["loan_amnt"], hue=loan_data_sorted["loan_status_count_10"],
            hue_order=labels,
            ax=ax_loan_status_term, estimator=lambda x: len(x) / len(loan_data_sorted["loan_status_count_10"]) * 100)
ax_loan_status_term.set_title("loan times per term", fontsize=16)
ax_loan_status_term.set_xlabel("term", fontsize=16)
ax_loan_status_term.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_10"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_term.savefig("./pictures/loan times per term bar.jpg", dpi=200,
                           bbox_inches="tight")
```
# 模型训练和预测前的数据预处理
如果我们想直接进行模型训练和预测，跳过数据可视化分析部分，那么我们只需运行数据可视化分析前的数据预处理中所有部分的代码，然后接着运行模型训练和预测前的数据集处理中所有代码，再运行模型训练和预测代码即可。
## 引入包和载入数据集
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

loan_data = pd.read_csv("loan_clean_data.csv", low_memory=False)
print(loan_data.shape)
```
## 样本类别不平衡状态的统计
该数据集中存在严重的样本不平衡，训练集中正样本：负样本=6.775：1。在后面进行模型训练和预测时我们需要注意这个问题。
```python
# There is a serious problem of sample imbalance in this dataset
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_11"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_11"] = loan_data["loan_status_count_11"].astype("float")
loan_data_status_count = loan_data["loan_status_count_11"].value_counts().to_dict()
sum_value = 0.0
for key, value in loan_data_status_count.items():
	sum_value += value
for key, value in loan_data_status_count.items():
	loan_data_status_count[key] = value / sum_value
print(loan_data_status_count)
# {1.0: 0.8713651803190625, 0.0: 0.12863481968093743}
loan_data.drop(["loan_status_count_11"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 87)
```
## 特征的抛弃、转换、填充
特征loan_status代表该贷款的状态，我们可按下面的映射转为数值型特征:
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
​                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
即变为二分类问题。
特征emp_length代表贷款人的工作年限，可按下面的映射转为数值型特征：
emp_length_dict = {'10+ years': 10, '2 years': 2, '< 1 year': 0.5, '3 years': 3, '1 year': 1, '5 years': 5,'4 years': 4, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9}
注意小于1年的映射为0.5，另外该特征的缺失值填充为0。
删除无用特征：emp_title', 'title', 'zip_code', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d。
emp_title的特征取值过多且每个取值出现次数占总样本数的比例很小，对模型学习和预测没有帮助，故删除。
'zip_code', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d’这几个特征对于模型学习没有什么实际的意义，故也删除。
```python
# feature:loan_status‘s change to 0 or 1
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 1 is good loan,0 is bad loan
loan_data["loan_status"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status"] = loan_data["loan_status"].astype("float")
# print(loan_data["emp_length"].value_counts().to_dict)
emp_length_dict = {"10+ years": 10, "2 years": 2, "< 1 year": 0.5, "3 years": 3, "1 year": 1, "5 years": 5,
                   "4 years": 4, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9}
loan_data["emp_length"] = loan_data["emp_length"].map(emp_length_dict)
loan_data["emp_length"] = loan_data["emp_length"].astype("float")
# fill missing value of emp_length to 0
loan_data["emp_length"].fillna(value=0, inplace=True)
# drop some features
loan_data.drop(["emp_title", "title", "zip_code", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"], axis=1,
               inplace=True)
loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
loan_data.drop(["issue_d"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 82)
```
## 数值型特征的皮尔森相关系数矩阵
皮尔森相关系数矩阵可以表示两两特征之间的线性相关性。 若>0，表明两个变量是正相关，即一个变量的值越大，另一个变量的值也会越大；若<0，表明两个变量是负相关，即一个变量的值越大，另一个变量的值反而会越小；若r=0，表明两个变量间不是线性相关，但有可能是非线性相关。
注意皮尔森相关系数只需要对数值型特征进行计算。我们可以看到一开始可视化的loan_amnt和funded_amnt的两个特征（下图第一个和第二个特征）相关系数接近1，这验证了我们之前对这两个特征的相关性猜想。
```python
numerical_feature_name = loan_data.columns[(loan_data.dtypes == "float64") | (loan_data.dtypes == "int64")].tolist()
# category_feature_name = loan_data.columns[loan_data.dtypes == "object"].tolist()
# draw pearson correlation coefficient matrix
# 若>0，表明两个变量是正相关,即一个变量的值越大，另一个变量的值也会越大
# 若<0，表明两个变量是负相关，即一个变量的值越大另一个变量的值反而会越小
# 若r=0，表明两个变量间不是线性相关，但有可能是非线性相关
corrmat = loan_data[numerical_feature_name].corr()
f, ax = plt.subplots(figsize=(30, 20))
# vmax、vmin即热力图颜色取值的最大值和最小值,默认会从data中推导
# square=True会将单元格设为正方形
sns.heatmap(corrmat, square=True, ax=ax, cmap="Blues", linewidths=0.5)
ax.set_title("Correlation coefficient matrix", fontsize=16)
ax.set_xlabel("feature names", fontsize=16)
ax.set_ylabel("feature names", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/Correlation coefficient matrix.jpg", dpi=200, bbox_inches="tight")
```
## 将特征和标签分开
```python
# split features and labels
X = loan_data.ix[:, loan_data.columns != "loan_status"]
Y = loan_data["loan_status"]
# Category features convert to one_hot code
X = pd.get_dummies(X, drop_first=True)
print(X.shape)
# (2137073, 200)
```
## 划分训练集和测试集
我们按照8:2划分训练集和测试集，训练集用来进行模型学习，测试集用来测试模型性能。
对于类别型特征，我们将其全部变为one_hot编码，注意有k个取值的类别型特征转为one_hot编码只需要k列one_hot编码特征即可。
```python
# divide training sets and testing sets
numerical_feature_name_2 = X.columns[(X.dtypes == "float64") | (X.dtypes == "int64")].tolist()
category_feature_name_2 = X.columns[X.dtypes == "uint8"].tolist()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1709658, 200) (427415, 200) (1709658,) (427415,)
print(len(numerical_feature_name_2), len(category_feature_name_2))
# 67 133
# 此时类别型特征已经全部变为one_hot编码(k-1列),数值型特征还需要归一化,由于特征值中有异常值,我们使用RobustScaler方法归一化
x_train_num, x_train_cat = x_train[numerical_feature_name_2], x_train[category_feature_name_2]
x_test_num, x_test_cat = x_test[numerical_feature_name_2], x_test[category_feature_name_2]
# get feature names
feature_names = list(x_train_num.columns)
feature_names.extend(list(x_train_cat.columns))
feature_names=np.array(feature_names)
print(feature_names.shape)
# 200
```
## 数值型特征归一化
该数据集中存在异常点(如前面的dti特征中就有异常点),我们也无法确保其他特征中没有异常点，因此我们对数值型特征进行RobustScaler归一化，这种算法取第一分位数到第四分位数之间的数据生成均值和标准差，然后对特征进行z score标准化。
```python
# robust scalar,默认为第一分位数到第四分位数之间的范围计算均值和方差,归一化还是z_score标准化
rob_scaler = RobustScaler()
x_train_num_rob = rob_scaler.fit_transform(x_train_num)
x_test_num_rob = rob_scaler.transform(x_test_num)
x_train_nom_pd = pd.DataFrame(np.hstack((x_train_num_rob, x_train_cat)))
x_test_nom_pd = pd.DataFrame(np.hstack((x_test_num_rob, x_test_cat)))
y_test_pd = pd.DataFrame(y_test)
x_train_sm_np, y_train_sm_np = x_train_nom_pd, y_train
print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# (1709658, 200) (1709658,) (427415, 200) (427415,)
```
## 样本不平衡问题的处理
由于我们后面分别使用lr、rf、xgb模型进行模型训练和预测。使用lr模型时，使用参数class_weight="balanced"即根据样本比例确定样本的权重来进行训练即可解决样本不平衡问题。使用rf和xgb模型时，由于模型本身的特点，不需要进行样本不平衡的处理。
我们还可以使用SMOTE算法生成少数类样本，使得不同类别的样本数量大致平衡。由于该算法占用内存较多，对于该数据集，你需要至少32GB内存才可以使用SMOTE算法。
**SMOTE算法流程：**
对于少数类中每一个样本x，以欧氏距离为标准计算它到少数类样本集中所有样本的距离，得到其k个近邻;
根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本x，重复N次下列操作：从其k个近邻中随机选择某个样本，假设选择的近邻为xn。对于每一个随机选出的近邻xn，生成一个0-1之间的随机权重t，按公式x+t*(x-xn)构建一个新的少数类样本。
```python
# # we can choose class_weight="balanced" to deal with sample imbalance problem when we use logistic model
# # if we use random foreast model or xgboost model,we don’t need to deal with sample imbalance problem
# # besides,we can also use SMOTE to generate some sample of the category with low number of samples,but it needs over 16GB memory
# # so,if you want to use SMOTE,you can run all these code on a local computer with at least 32GB memory
# # SMOTE算法即对于少数类中的每一个样本a,执行N次下列操作:
# # 从k个最近邻样本中随机选择一个样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
# # n_jobs=-1表示使用所有CPU
# sm = SMOTE(k_neighbors=10, random_state=0, n_jobs=-1)
# x_train_sm_np, y_train_sm_np = sm.fit_sample(x_train_nom_pd, y_train)
# print(x_train_sm_np.shape, y_train_sm_np.shape)


# x_train_sm_pd = pd.DataFrame(x_train_sm_np)
# y_train_sm_pd = pd.DataFrame(y_train_sm_np)
# x_train_sm_pd.to_csv("x_train_sm_np.csv", index=None)
# y_train_sm_pd.to_csv("y_train_sm_np.csv", index=None)
# x_test_nom_pd.to_csv("x_test_nom_np.csv", index=None)
# y_test_pd.to_csv("y_test.csv", index=None)
# x_train_sm_np = np.array(pd.read_csv("x_train_sm_np.csv", low_memory=False))
# y_train_sm_np = np.array(pd.read_csv("y_train_sm_np.csv", low_memory=False))
# x_test_nom_pd = np.array(pd.read_csv("x_test_nom_np.csv", low_memory=False))
# y_test = np.array(pd.read_csv("y_test.csv", low_memory=False))
# print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# # (2980148, 200) (2980148, 1) (427415, 200) (427415, 1)
# # 标签需要降维成一维数组
# y_train_sm_np = y_train_sm_np.ravel()
# y_test = y_test.ravel()
# print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# # (2980148, 200) (2980148,) (427415, 200) (427415,)


# if your computer's memory is not more than 16GB,it is not enough to run three models at the same time,you can choose one model to run.
```
# 模型训练和预测
## 使用lr模型进行训练和预测
sag即随机平均梯度下降，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候；class_weight="balanced"根据用来训练的样本的各个类别的比例确定权重；n_jobs=-1表示使用所有CPU一起进行模型拟合和预测。
```python
# use logistic regression model to train and predict
# jobs=-1使用所有CPU进行运算
# sag即随机平均梯度下降，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候
# class_weight="balanced"根据用来训练的样本的各个类别的比例确定权重
print("use logistic model to train and predict")
lr = LogisticRegression(solver="sag", class_weight="balanced", n_jobs=-1)
lr.fit(x_train_sm_np, y_train_sm_np)
lr_y_pred = lr.predict(x_test_nom_pd)
lr_test_acc = accuracy_score(y_test, lr_y_pred)
lr_classification_score = classification_report(y_test, lr_y_pred)
print("Lr model test accuracy:{:.2f}".format(lr_test_acc))
print("Lr model classification_score:\n", lr_classification_score)
lr_confusion_score = confusion_matrix(y_test, lr_y_pred)
f_lr, ax_lr = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
lr_cm_pred_label_sum = lr_confusion_score.sum(axis=0)
lr_cm_true_label_sum = lr_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
lr_model_precision, lr_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
lr_model_precision[0][0], lr_model_precision[1][0] = lr_confusion_score[0][0] / lr_cm_pred_label_sum[0], \
                                                     lr_confusion_score[1][0] / lr_cm_pred_label_sum[0]
lr_model_precision[0][1], lr_model_precision[1][1] = lr_confusion_score[0][1] / lr_cm_pred_label_sum[1], \
                                                     lr_confusion_score[1][1] / lr_cm_pred_label_sum[1]
lr_model_recall[0][0], lr_model_recall[0][1] = lr_confusion_score[0][0] / lr_cm_true_label_sum[0], \
                                               lr_confusion_score[0][1] / lr_cm_true_label_sum[0]
lr_model_recall[1][0], lr_model_recall[1][1] = lr_confusion_score[1][0] / lr_cm_true_label_sum[1], \
                                               lr_confusion_score[1][1] / lr_cm_true_label_sum[1]
sns.heatmap(lr_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_lr[0], square=True, linewidths=0.5)
sns.heatmap(lr_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_lr[1], square=True, linewidths=0.5)
sns.heatmap(lr_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_lr[2], square=True, linewidths=0.5)
ax_lr[0].set_title("lr confusion matrix", fontsize=16)
ax_lr[1].set_title("lr model precision", fontsize=16)
ax_lr[2].set_title("lr model recall", fontsize=16)
ax_lr[0].set_xlabel("Predicted label", fontsize=16)
ax_lr[0].set_ylabel("True label", fontsize=16)
ax_lr[1].set_xlabel("Predicted label", fontsize=16)
ax_lr[1].set_ylabel("True label", fontsize=16)
ax_lr[2].set_xlabel("Predicted label", fontsize=16)
ax_lr[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_lr.savefig("./pictures/lr model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# result
# Lr model test accuracy:0.88
# Lr model classification_score:
#                precision    recall  f1-score   support
#          0.0       0.54      0.73      0.62     55318
#          1.0       0.96      0.91      0.93    372097
#    micro avg       0.88      0.88      0.88    427415
#    macro avg       0.75      0.82      0.78    427415
# weighted avg       0.90      0.88      0.89    427415
```
混淆矩阵从左到右三个子图为lr模型预测标签/实际标签的样本的数量，lr模型预测负/正样本的精确率（第二张图左上角和右下角数值），lr模型预测负/正样本的召回率（第三张图左上角和右下角数值）。
Lr模型预测总体准确率0.88，其中正样本预测精确率0.96，但负样本预测精确率很低，只有0.54。同时正样本预测召回率0.91，负样本召回率0.73也很低。由于我们更想用模型预测出不良贷款，而这个模型的负样本准确率和召回率太低了，模型的预测性能不太好。
## 使用rf模型进行训练和预测
n_estimators=200表示模型有200课树，n_jobs=-1使用所有CPU进行模型拟合和预测。
```python
# use randomforest model to train and predict
print("use randomforest model to train and predict")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(x_train_sm_np, y_train_sm_np)
rf_y_pred = rf.predict(x_test_nom_pd)
rf_test_acc = accuracy_score(y_test, rf_y_pred)
rf_classification_score = classification_report(y_test, rf_y_pred)
print("Rf model test accuracy:{:.4f}".format(rf_test_acc))
print("rf model classification_score:\n", rf_classification_score)
rf_confusion_score = confusion_matrix(y_test, rf_y_pred)
# print(rf_confusion_score)
f_rf, ax_rf = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
rf_cm_pred_label_sum = rf_confusion_score.sum(axis=0)
rf_cm_true_label_sum = rf_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
rf_model_precision, rf_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
rf_model_precision[0][0], rf_model_precision[1][0] = rf_confusion_score[0][0] / rf_cm_pred_label_sum[0], \
                                                     rf_confusion_score[1][0] / rf_cm_pred_label_sum[0]
rf_model_precision[0][1], rf_model_precision[1][1] = rf_confusion_score[0][1] / rf_cm_pred_label_sum[1], \
                                                     rf_confusion_score[1][1] / rf_cm_pred_label_sum[1]
rf_model_recall[0][0], rf_model_recall[0][1] = rf_confusion_score[0][0] / rf_cm_true_label_sum[0], \
                                               rf_confusion_score[0][1] / rf_cm_true_label_sum[0]
rf_model_recall[1][0], rf_model_recall[1][1] = rf_confusion_score[1][0] / rf_cm_true_label_sum[1], \
                                               rf_confusion_score[1][1] / rf_cm_true_label_sum[1]
sns.heatmap(rf_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_rf[0], square=True, linewidths=0.5)
sns.heatmap(rf_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[1], square=True, linewidths=0.5)
sns.heatmap(rf_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[2], square=True, linewidths=0.5)
ax_rf[0].set_title("rf confusion matrix", fontsize=16)
ax_rf[1].set_title("rf model precision", fontsize=16)
ax_rf[2].set_title("rf model recall", fontsize=16)
ax_rf[0].set_xlabel("Predicted label", fontsize=16)
ax_rf[0].set_ylabel("True label", fontsize=16)
ax_rf[1].set_xlabel("Predicted label", fontsize=16)
ax_rf[1].set_ylabel("True label", fontsize=16)
ax_rf[2].set_xlabel("Predicted label", fontsize=16)
ax_rf[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_rf.savefig("./pictures/rf model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# Rf model test accuracy:0.9828
# rf model classification_score:
#                precision    recall  f1-score   support
#
#          0.0       1.00      0.87      0.93     55318
#          1.0       0.98      1.00      0.99    372097
#
#    micro avg       0.98      0.98      0.98    427415
#    macro avg       0.99      0.93      0.96    427415
# weighted avg       0.98      0.98      0.98    427415
```
混淆矩阵从左到右三个子图为rf模型预测标签/实际标签的样本的数量，rf模型预测负/正样本的精确率（第二张图左上角和右下角数值），rf模型预测负/正样本的召回率（第三张图左上角和右下角数值）。
rf模型预测总体准确率0.98，其中正样本预测精确率0.981，负样本预测精确率0.999。同时正样本预测召回率0.999，负样本召回率0.868。该模型对于负样本（不良贷款）预测的精确率很高，召回率也相对较好，故该模型比较适合用来预测。
**我们还可以画出该模型训练时各特征的贡献度:**
```python
# random forest model feature contribution visualization
feature_importances = rf.feature_importances_
# print(feature_importances)
# y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
indices = np.argsort(feature_importances)[::-1]
# 只取贡献度最高的30个特征来作图
show_indices = indices[0:30]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=feature_importances[show_indices], y=feature_names[show_indices], ax=ax)
ax.set_title("rf model feature importance top30", fontsize=16)
ax.set_xlabel("feature importance score", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/rf model feature importance top30.jpg", dpi=200, bbox_inches="tight")
```
## 使用xgb模型进行训练和预测
n_estimators=200表示使用200课回归树，nthread=-1表示使用CPU所有线程进行模型拟合和预测。
```python
# use XGBoost model to train and predict
xgb = XGBClassifier(n_estimators=200, nthread=-1)
xgb.fit(x_train_sm_np, y_train_sm_np)
xgb_y_pred = xgb.predict(x_test_nom_pd)
xgb_test_acc = accuracy_score(y_test, xgb_y_pred)
xgb_classification_score = classification_report(y_test, xgb_y_pred)
print("Xgb model test accuracy:{:.4f}".format(xgb_test_acc))
print("Xgb model classification_score:\n", xgb_classification_score)
xgb_confusion_score = confusion_matrix(y_test, xgb_y_pred)
# print(xgb_confusion_score)
f_xgb, ax_xgb = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
xgb_cm_pred_label_sum = xgb_confusion_score.sum(axis=0)
xgb_cm_true_label_sum = xgb_confusion_score.sum(axis=1)
# print(xgb_cm_pred_label_sum,xgb_cm_true_label_sum)
# 计算正样本和负样本的精确率和召回率
xgb_model_precision, xgb_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
xgb_model_precision[0][0], xgb_model_precision[1][0] = xgb_confusion_score[0][0] / xgb_cm_pred_label_sum[0], \
                                                       xgb_confusion_score[1][0] / xgb_cm_pred_label_sum[0]
xgb_model_precision[0][1], xgb_model_precision[1][1] = xgb_confusion_score[0][1] / xgb_cm_pred_label_sum[1], \
                                                       xgb_confusion_score[1][1] / xgb_cm_pred_label_sum[1]
xgb_model_recall[0][0], xgb_model_recall[0][1] = xgb_confusion_score[0][0] / xgb_cm_true_label_sum[0], \
                                                 xgb_confusion_score[0][1] / xgb_cm_true_label_sum[0]
xgb_model_recall[1][0], xgb_model_recall[1][1] = xgb_confusion_score[1][0] / xgb_cm_true_label_sum[1], \
                                                 xgb_confusion_score[1][1] / xgb_cm_true_label_sum[1]
sns.heatmap(xgb_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_xgb[0], square=True, linewidths=0.5)
sns.heatmap(xgb_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_xgb[1], square=True, linewidths=0.5)
sns.heatmap(xgb_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_xgb[2], square=True, linewidths=0.5)
ax_xgb[0].set_title("xgb confusion matrix", fontsize=16)
ax_xgb[1].set_title("xgb model precision", fontsize=16)
ax_xgb[2].set_title("xgb model recall", fontsize=16)
ax_xgb[0].set_xlabel("Predicted label", fontsize=16)
ax_xgb[0].set_ylabel("True label", fontsize=16)
ax_xgb[1].set_xlabel("Predicted label", fontsize=16)
ax_xgb[1].set_ylabel("True label", fontsize=16)
ax_xgb[2].set_xlabel("Predicted label", fontsize=16)
ax_xgb[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_xgb.savefig("./pictures/xgb model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# Xgb model test accuracy:0.9809
# Xgb model classification_score:
#                precision    recall  f1-score   support
#
#          0.0       1.00      0.85      0.92     55318
#          1.0       0.98      1.00      0.99    372097
#
#    micro avg       0.98      0.98      0.98    427415
#    macro avg       0.99      0.93      0.95    427415
# weighted avg       0.98      0.98      0.98    427415
```
混淆矩阵从左到右三个子图为xgb模型预测标签/实际标签的样本的数量，xgb模型预测负/正样本的精确率（第二张图左上角和右下角数值），xgb模型预测负/正样本的召回率（第三张图左上角和右下角数值）。
xgb模型预测总体准确率0.98，其中正样本预测精确率0.978，但负样本预测精确率0.999。同时正样本预测召回率0.999，负样本召回率0.854。该模型对于负样本（不良贷款）预测的精确率也很高，召回率也相对较好，模型性能略逊于rf模型，但相差很小，该模型也比较适合用来预测。
**我们还可以画出该模型训练时各特征的贡献度:**
```python
# xgboost model feature contribution visualization
feature_importances = xgb.feature_importances_
# print(feature_importances)
# y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
indices = np.argsort(feature_importances)[::-1]
# 只取贡献度最高的30个特征来作图
show_indices = indices[0:30]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=feature_importances[show_indices], y=feature_names[show_indices], ax=ax)
ax.set_title("xgb model feature importance top30", fontsize=16)
ax.set_xlabel("feature importance score", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/xgb model feature importance top30.jpg", dpi=200, bbox_inches="tight")
```
# 完整本地运行代码
包含data_visualization.py和model_training_and_testing.py两个文件。
data_visualization.py
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from pyecharts import Bar3D, Line3D
from pyecharts import WordCloud
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# Data preprocessing before Data analysis visualization
loan_data = pd.read_csv("loan.csv", low_memory=False)
print(loan_data.shape)


# (2260668, 145)


# calculate the missing value percent of features
def draw_missing_data_table(data):
	total = data.isnull().sum().sort_values(ascending=False)
	percent = (data.isnull().sum() / data.shape[0]).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
	missing_data.reset_index(inplace=True)
	missing_data.rename(columns={"index": "feature_name"}, inplace=True)

	return missing_data


# save missing value percent of features
missing_data_count = draw_missing_data_table(loan_data)
missing_data_count.to_csv("missing_data_count.csv")
missing_data_count = pd.read_csv("missing_data_count.csv", header=0, index_col=0)
missing_data_count = missing_data_count[missing_data_count["Percent"] > 0.0]
print(missing_data_count.head())
#                                  feature_name    Total   Percent
# 0                                          id  2260668  1.000000
# 1                                   member_id  2260668  1.000000
# 2                                         url  2260668  1.000000
# 3  orig_projected_additional_accrued_interest  2252242  0.996273
# 4                         hardship_start_date  2250055  0.995305


# draw a graph of missing value percent of features(percent>0.03)
missing_data_count_show = missing_data_count[missing_data_count["Percent"] > 0.03]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=missing_data_count_show["Percent"], y=missing_data_count_show["feature_name"], ax=ax)
ax.set_title("Missing value percent for each feature", fontsize=16)
ax.set_xlabel("missing percent", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/Missing value percent for each feature.jpg", dpi=200, bbox_inches="tight")

# delete features that missing value percent more than 0.15
for index, feature_count_null in missing_data_count.iterrows():
	if feature_count_null["Percent"] > 0.15:
		drop_feature_name = feature_count_null["feature_name"]
		loan_data.drop([drop_feature_name], axis=1, inplace=True)

missing_data_count = missing_data_count[missing_data_count["Percent"] <= 0.15]
print(missing_data_count.head())
#              feature_name   Total   Percent
# 58  mths_since_recent_inq  295435  0.130685
# 59              emp_title  166969  0.073858
# 60       num_tl_120dpd_2m  153657  0.067970
# 61             emp_length  146907  0.064984
# 62     mo_sin_old_il_acct  139071  0.061518


# delete rows which contain missing value for features that  missing value precent less than 0.04
for index, feature_count_null in missing_data_count.iterrows():
	if feature_count_null["Percent"] < 0.04:
		drop_feature_name = feature_count_null["feature_name"]
		drop_index = loan_data[loan_data[drop_feature_name].isnull().values == True].index
		loan_data.drop(index=drop_index, axis=0, inplace=True)

print(loan_data.shape)
# (2137073, 87)


# calculate the missing value percent of features again,save missing value percent of features
missing_data_count_2 = draw_missing_data_table(loan_data)
missing_data_count_2.to_csv("missing_data_count_2.csv")
missing_data_count_2 = missing_data_count_2[missing_data_count_2["Percent"] > 0.0]
print(missing_data_count_2)
#             feature_name   Total   Percent
# 0  mths_since_recent_inq  235741  0.110310
# 1              emp_title  154722  0.072399
# 2             emp_length  137175  0.064188
# 3       num_tl_120dpd_2m   81243  0.038016
# 4     mo_sin_old_il_acct   66915  0.031312


# fill missing value of mths_since_recent_inq/num_tl_120dpd_2m/mo_sin_old_il_acct by mean value of each feature
# don"t fill emp_title and emp_length
loan_data["mths_since_recent_inq"].fillna(loan_data["mths_since_recent_inq"].mean(), inplace=True)
loan_data["num_tl_120dpd_2m"].fillna(loan_data["num_tl_120dpd_2m"].mean(), inplace=True)
loan_data["mo_sin_old_il_acct"].fillna(loan_data["mo_sin_old_il_acct"].mean(), inplace=True)
# Convert the value of feature:"term" from category to numeric
term_dict = {" 36 months": 36, " 60 months": 60}
loan_data["term"] = loan_data["term"].map(term_dict)
loan_data["term"] = loan_data["term"].astype("float")

# calculate the missing value percent of features the three times,save missing value percent of features
missing_data_count_3 = draw_missing_data_table(loan_data)
missing_data_count_3.to_csv("missing_data_count_3.csv")
missing_data_count_3 = missing_data_count_3[missing_data_count_3["Percent"] > 0.0]
print(missing_data_count_3)
print(loan_data.shape)
#   feature_name   Total   Percent
# 0    emp_title  154722  0.072399
# 1   emp_length  137175  0.064188
# (2137073, 87)


# save the dataset after all missing value operation
loan_data.to_csv("loan_clean_data.csv", index=None)
loan_data = pd.read_csv("loan_clean_data.csv", low_memory=False)
print(loan_data.shape)
# (2137073, 87)


# Data analysis visualization


# seaborn has five themes:darkgrid(灰色网格)\whitegrid(白色网格)\dark(黑色)\white(白色)\ticks(十字叉)
# Palette has those options:"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
# plotly only supports world maps,the United States maps
# pyechart only supports world maps and China maps


# Data distribution of the loan amount and actual loan amount
sns.set_style("whitegrid")
f_loan, ax_loan = plt.subplots(2, 2, figsize=(15, 10))
sns.distplot(loan_data["loan_amnt"], ax=ax_loan[0, 0], color="#F7522F")
sns.violinplot(y=loan_data["loan_amnt"], ax=ax_loan[0, 1], inner="quartile", palette="Reds")
sns.distplot(loan_data["funded_amnt"], ax=ax_loan[1, 0], color="#2F8FF7")
sns.violinplot(y=loan_data["funded_amnt"], ax=ax_loan[1, 1], inner="quartile", palette="Blues")
ax_loan[0, 0].set_title("Loan amount distribution", fontsize=16)
ax_loan[0, 1].set_title("Loan amount distribution", fontsize=16)
ax_loan[1, 0].set_title("Funded amount distribution", fontsize=16)
ax_loan[1, 1].set_title("Funded amount distribution", fontsize=16)
ax_loan[0, 0].set_xlabel("loan amount", fontsize=16)
ax_loan[1, 0].set_xlabel("loan amount", fontsize=16)
ax_loan[0, 1].set_ylabel("loan amount", fontsize=16)
ax_loan[1, 1].set_ylabel("loan amount", fontsize=16)
plt.show()
plt.close()
f_loan.savefig("./pictures/Loan amount and funded amount distribution.jpg", dpi=200, bbox_inches="tight")

# histogram of annual loan figures and histogram of total amount of annual loan lending
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
loan_year_num = loan_data["year"].value_counts().to_dict()
loan_year_num_pd = pd.DataFrame(list(loan_year_num.items()), columns=["year", "loan times"])
loan_year_num_pd.sort_values("year", inplace=True)
# print(loan_year_num_pd)
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
loan_money_count_per_year = loan_data.groupby("year")["loan_amnt"].sum().to_dict()
loan_money_count_per_year_pd = pd.DataFrame(list(loan_money_count_per_year.items()), columns=["year", "loan_amnt"])
loan_money_count_per_year_pd.sort_values("year", inplace=True)
# print(loan_money_count_per_year_pd)
sns.set_style("whitegrid")
f_loan_per_year, ax_loan_per_year = plt.subplots(1, 2, figsize=(15, 10))
sns.barplot(loan_year_num_pd["year"], loan_year_num_pd["loan times"], ax=ax_loan_per_year[0],
            palette="tab10")
sns.barplot(loan_money_count_per_year_pd["year"], loan_money_count_per_year_pd["loan_amnt"], ax=ax_loan_per_year[1],
            palette="tab10")
ax_loan_per_year[0].set_title("loan times per year", fontsize=16)
ax_loan_per_year[1].set_title("Loan amount per year", fontsize=16)
ax_loan_per_year[0].set_xlabel("year", fontsize=16)
ax_loan_per_year[0].set_ylabel("loan times", fontsize=16)
ax_loan_per_year[1].set_xlabel("year", fontsize=16)
ax_loan_per_year[1].set_ylabel("loan amount", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["year"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 87)
f_loan_per_year.savefig("./pictures/loan times and loan amount per year.jpg", dpi=200, bbox_inches="tight")

# # 各年各月每笔贷款平均贷款金额3D柱状图和3D折线图
# loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
# months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# years_list = ["2012", "2013", "2014", "2015", "2016", "2017", "2018"]
# mean_loan_amnt_per_year_per_month_dict = loan_data.groupby(["month", "year"])["loan_amnt"].mean().to_dict()
# # print(loan_amnt_per_year_per_month_dict)
# max_value = max(mean_loan_amnt_per_year_per_month_dict.values())
# mean_loan_amnt_per_year_per_month_list = []
# for key, value in mean_loan_amnt_per_year_per_month_dict.items():
# 	temp = [key[0], key[1], value]
# 	mean_loan_amnt_per_year_per_month_list.append(temp)
# # print(loan_amnt_per_year_per_month_list)
# mean_loan_amnt_per_year_per_month_bar3d = Bar3D("每月贷款金额3D柱状图", width=1500, height=1000)
# range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43",
#                "#d73027", "#a50026"]
# mean_loan_amnt_per_year_per_month_bar3d.add("mean loan amnt per year per month bar3D", x_axis=months_list, y_axis=years_list,
#                                             data=mean_loan_amnt_per_year_per_month_list,
#                                             is_visualmap=True, visual_range=[0, max_value], visual_range_color=range_color,
#                                             grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# mean_loan_amnt_per_year_per_month_bar3d.render(path="./pictures/mean loan amnt per year per month bar3D.html")
# months_to_num_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
#                       "Oct": 10, "Nov": 11, "Dec": 12}
# for item in mean_loan_amnt_per_year_per_month_list:
# 	item[0], item[1] = months_to_num_dict[item[0]], int(item[1])
# # 画折线图时按照给定数据的输入顺序连线,所以我们要对列表先按月再按年从小到大排序
# mean_loan_amnt_per_year_per_month_list.sort(key=lambda x: x[0])
# mean_loan_amnt_per_year_per_month_list.sort(key=lambda x: x[1])
# colorscale = ["#9ecae1", "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9", "#08519c",
#               "#0b4083", "#08306b"]
# mean_loan_amnt_per_year_per_month_line3d = Line3D("每月贷款金额变化3D折线图", width=1500, height=1000)
# mean_loan_amnt_per_year_per_month_line3d.add("mean loan amnt per year per month line3D",
#                                              data=mean_loan_amnt_per_year_per_month_list,
#                                              yaxis3d_min=2012, yaxis3d_max=2018,
#                                              is_visualmap=True, visual_range=[0, max_value], visual_range_color=colorscale,
#                                              grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# mean_loan_amnt_per_year_per_month_line3d.render(path="./pictures/mean loan amnt per year per month line3D.html")
# loan_data.drop(["month","year"], axis=1, inplace=True)
# print(loan_data.shape)


# # 各年各月贷款笔数3D柱状图和3D折线图
# loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
# # print(loan_data["month"].value_counts())
# # print(loan_data["year"].value_counts())
# months_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# years_list = ["2012", "2013", "2014", "2015", "2016", "2017", "2018"]
# loan_times_per_year_num_dict = loan_data.groupby(["month"])["year"].value_counts().to_dict()
# max_value = max(loan_times_per_year_num_dict.values())
# loan_times_per_month_per_year_num_list = []
# for key, value in loan_times_per_year_num_dict.items():
# 	temp = [key[0], key[1], value]
# 	loan_times_per_month_per_year_num_list.append(temp)
# # print(loan_per_month_per_year_num_list)
# loan_times_per_month_per_year_bar3d = Bar3D("每月贷款笔数3D柱状图", width=1500, height=1000)
# range_color = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43",
#                "#d73027", "#a50026"]
# loan_times_per_month_per_year_bar3d.add("loan times per month per year bar3D", x_axis=months_list, y_axis=years_list,
#                                         data=loan_times_per_month_per_year_num_list,
#                                         is_visualmap=True, visual_range=[0, max_value], visual_range_color=range_color,
#                                         grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# loan_times_per_month_per_year_bar3d.render(path="./pictures/loan times per month per year bar3D.html")
# months_to_num_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
#                       "Oct": 10, "Nov": 11, "Dec": 12}
# for item in loan_times_per_month_per_year_num_list:
# 	item[0], item[1] = months_to_num_dict[item[0]], int(item[1])
# # 画折线图时按照给定数据的输入顺序连线,所以我们要对列表先按月再按年从小到大排序
# loan_times_per_month_per_year_num_list.sort(key=lambda x: x[0])
# loan_times_per_month_per_year_num_list.sort(key=lambda x: x[1])
# # print(loan_times_per_month_per_year_num_list)
# # loan_times_per_month_per_year_num_list=[[8, 2012, 894], [9, 2012, 5924], [10, 2012, 6192], [11, 2012, 6312], [12, 2012, 6006], [1, 2013, 6814], [2, 2013, 7506], [3, 2013, 8199], [4, 2013, 9354], [5, 2013, 10285], [6, 2013, 10815], [7, 2013, 11816], [8, 2013, 12562], [9, 2013, 12866], [10, 2013, 13858], [11, 2013, 14561], [12, 2013, 14854], [1, 2014, 15470], [2, 2014, 15111], [3, 2014, 16296], [4, 2014, 18829], [5, 2014, 18870], [6, 2014, 16996], [7, 2014, 28948], [8, 2014, 18632], [9, 2014, 10498], [10, 2014, 38244], [11, 2014, 24679], [12, 2014, 10173], [1, 2015, 34691], [2, 2015, 23474], [3, 2015, 25123], [4, 2015, 35052], [5, 2015, 31547], [6, 2015, 28170], [7, 2015, 45446], [8, 2015, 35469], [9, 2015, 28343], [10, 2015, 48064], [11, 2015, 37084], [12, 2015, 43702], [1, 2016, 29548], [2, 2016, 35778], [3, 2016, 56707], [4, 2016, 33093], [5, 2016, 25975], [6, 2016, 30512], [7, 2016, 32575], [8, 2016, 33488], [9, 2016, 26432], [10, 2016, 32318], [11, 2016, 34068], [12, 2016, 35618], [1, 2017, 31435], [2, 2017, 27418], [3, 2017, 36754], [4, 2017, 29270], [5, 2017, 37245], [6, 2017, 37548], [7, 2017, 38784], [8, 2017, 42765], [9, 2017, 38988], [10, 2017, 37434], [11, 2017, 41513], [12, 2017, 37376], [1, 2018, 35718], [2, 2018, 32126], [3, 2018, 38054], [4, 2018, 42177], [5, 2018, 45489], [6, 2018, 40821], [7, 2018, 42372], [8, 2018, 45298], [9, 2018, 38380], [10, 2018, 45540], [11, 2018, 41247], [12, 2018, 39480]]
# colorscale = ["#9ecae1", "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9", "#08519c",
#               "#0b4083", "#08306b"]
# loan_times_per_month_per_year_line3d = Line3D("每月贷款笔数变化3D折线图", width=1500, height=1000)
# loan_times_per_month_per_year_line3d.add("loan times per month per year line3D",
#                                          data=loan_times_per_month_per_year_num_list,
#                                          yaxis3d_min=2012, yaxis3d_max=2018,
#                                          is_visualmap=True, visual_range=[0, max_value], visual_range_color=colorscale,
#                                          grid3d_width=200, grid3d_height=100, grid3d_depth=80)
# # 3D图不能保存为png格式
# loan_times_per_month_per_year_line3d.render(path="./pictures/loan times per month per year line3D.html")
# loan_data.drop(["month","year"], axis=1, inplace=True)
# print(loan_data.shape)

# This is my free plotly account,this account allows up to 100 images to be generated every 24 hours.Please use your own plotly account.
plotly.tools.set_credentials_file(username="zgcr", api_key="GQW92qmUOFbZmTQwQtJ1")
plotly.tools.set_config_file(world_readable=True)

# the map of geographical coordinates of each state"s loan figures
# addr_state即申请贷款的人的所属州,是两位代码,可以被plotly识别
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
loan_times_per_state = loan_data["addr_state"].value_counts().to_dict()
loan_times_per_state_pd = pd.DataFrame(list(loan_times_per_state.items()), columns=["state_code", "loan_times"])
loan_times_per_state_pd["state_name"] = None
# print(loan_times_per_state_pd)
for i in range(loan_times_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_times_per_state_pd.ix[i, "state_code"]]
	loan_times_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_times_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Blues"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=True,
             locations=loan_times_per_state_pd["state_code"], z=loan_times_per_state_pd["loan_times"].astype(float),
             locationmode="USA-states", text=loan_times_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="loan times", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="loan times per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="loan times per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/loan times per state map.png", width=2500, height=1500)

# Histogram of each state"s loan figures (the top 30 states with the largest number of loans)
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
loan_times = loan_data["addr_state"].value_counts().to_dict()
loan_times_pd = pd.DataFrame(list(loan_times.items()), columns=["state_code", "loan_times"])
loan_times_pd["state_name"] = None
# print(loan_times_pd)
for i in range(loan_times_pd.shape[0]):
	state_name = code_and_name_dict[loan_times_pd.ix[i, "state_code"]]
	loan_times_pd.ix[i, "state_name"] = state_name
# print(loan_times_pd)
loan_times_pd_30 = loan_times_pd[0:30]
loan_times_pd_30.drop(["state_code"], axis=1)
sns.set_style("whitegrid")
f_loan_times_per_state, ax_loan_times_per_state = plt.subplots(figsize=(15, 10))
# # palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_pd_30["loan_times"], loan_times_pd_30["state_name"], ax=ax_loan_times_per_state,
            palette="tab10")
ax_loan_times_per_state.set_title("loan times per state", fontsize=16)
ax_loan_times_per_state.set_xlabel("loan times", fontsize=16)
ax_loan_times_per_state.set_ylabel("state name", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_state.savefig("./pictures/loan times per state bar.jpg", dpi=200, bbox_inches="tight")

# histogram of the top 30 profession of loan figures
loan_times_title = loan_data["emp_title"].value_counts().to_dict()
loan_times_title_pd = pd.DataFrame(list(loan_times_title.items()), columns=["title", "loan_times"])
loan_times_title_pd_30 = loan_times_title_pd[0:30]
sns.set_style("whitegrid")
f_loan_times_per_title, ax_loan_times_per_title = plt.subplots(figsize=(15, 10))
# # palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_title_pd_30["loan_times"], loan_times_title_pd_30["title"], ax=ax_loan_times_per_title,
            palette="tab10")
ax_loan_times_per_title.set_title("loan times per title", fontsize=16)
ax_loan_times_per_title.set_xlabel("loan times", fontsize=16)
ax_loan_times_per_title.set_ylabel("title", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_title.savefig("./pictures/loan times per title bar.jpg", dpi=200, bbox_inches="tight")

# histogram of the year of participation in working with loan figures
loan_times_length = loan_data["emp_length"].value_counts().to_dict()
# print(loan_times_length)
# {"10+ years": 713245, "2 years": 192330, "< 1 year": 179177, "3 years": 170699, "1 year": 139017, "5 years": 130985,
# "4 years": 128027, "6 years": 96294, "7 years": 87537, "8 years": 87182, "9 years": 75405}
loan_times_length_pd = pd.DataFrame(list(loan_times_length.items()), columns=["length", "loan_times"])
sns.set_style("whitegrid")
f_loan_times_per_length, ax_loan_times_per_length = plt.subplots(figsize=(15, 10))
# palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_length_pd["length"], loan_times_length_pd["loan_times"], ax=ax_loan_times_per_length,
            palette="Blues_d")
ax_loan_times_per_length.set_title("loan times per length", fontsize=16)
ax_loan_times_per_length.set_xlabel("worked length", fontsize=16)
ax_loan_times_per_length.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
f_loan_times_per_length.savefig("./pictures/loan times per length bar.jpg", dpi=200, bbox_inches="tight")

# histogram of the customer"s annual income with loan figures
# 我们将年收入分为三档:20000以下为low，20000-60000为mid，>60000为high
max_value = loan_data["annual_inc"].max() + 1.0
set_bins = [0.0, 20000.0, 60000.0, max_value]
set_label = ["low", "mid", "high"]
loan_data["income"] = pd.cut(loan_data["annual_inc"], bins=set_bins, labels=set_label)
loan_times_income = loan_data["income"].value_counts().to_dict()
# print(loan_times_income)
# {"high": 1187055, "mid": 912572, "low": 37443}
loan_times_income_pd = pd.DataFrame(list(loan_times_income.items()), columns=["income", "loan_times"])
sns.set_style("whitegrid")
f_loan_times_per_income, ax_loan_times_per_income = plt.subplots(figsize=(15, 10))
# palette为调色板参数,可选项"muted"\"RdBu"\"RdBu_r"\"Blues_d"\"Set1"\"husl"
sns.barplot(loan_times_income_pd["income"], loan_times_income_pd["loan_times"], ax=ax_loan_times_per_income,
            palette="muted")
ax_loan_times_per_income.set_title("loan times per income", fontsize=16)
ax_loan_times_per_income.set_xlabel("income", fontsize=16)
ax_loan_times_per_income.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["income"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_times_per_income.savefig("./pictures/loan times per income bar.jpg", dpi=200, bbox_inches="tight")

# The ratio of good loans and bad loans for each year
# print(loan_data["loan_status"].value_counts().to_dict())
# {"Fully Paid": 962556, "Current": 899615, "Charged Off": 241514, "Late (31-120 days)": 21051, "In Grace Period": 8701,
# "Late (16-30 days)": 3607, "Default": 29}
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count"] = loan_data["loan_status_count"].astype("float")
# print(loan_data["loan_status"].value_counts().to_dict())
# {1.0: 1862171, 0.0: 274902}可以看到正负样本不均衡，在后面我们训练模型预测loan_status时需要注意正负样本不平衡的问题
loan_status_count = loan_data["loan_status_count"].value_counts().to_dict()
if 0 not in loan_status_count.keys():
	loan_status_count["0"] = 0.0
count_sum = 0
for key, value in loan_status_count.items():
	count_sum += value
for key, value in loan_status_count.items():
	value = value / count_sum
	loan_status_count[key] = value
loan_status_count_pd = pd.DataFrame(list(loan_status_count.items()), columns=["loan status", "count_percent"])
# print(loan_status_count_pd)
#    loan status  count_percent
# 0          1.0       0.871365
# 1          0.0       0.128635
loan_data["year"] = pd.to_datetime(loan_data["issue_d"]).dt.year
f_loan_status, ax_loan_status = plt.subplots(1, 2, figsize=(15, 10))
labels = "Good loans", "Bad loans"
ax_loan_status[0].pie(loan_status_count_pd["count_percent"], autopct="%1.2f%%", shadow=True,
                      labels=labels, startangle=70)
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count"] = loan_data["loan_status_count"].map(loan_status_dict)
sns.barplot(x=loan_data["year"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count"], hue_order=labels,
            ax=ax_loan_status[1], estimator=lambda x: len(x) / len(loan_data["loan_status_count"]) * 100)
ax_loan_status[0].set_title("good loans and bad loans percent", fontsize=16)
ax_loan_status[0].set_ylabel("Loans percent", fontsize=16)
ax_loan_status[1].set_title("good loans and bad loans percent per year", fontsize=16)
ax_loan_status[1].set_ylabel("Loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count", "year"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status.savefig("./pictures/good loans and bad loans percent per year.jpg", dpi=200,
                      bbox_inches="tight")

# the map of geographical coordinates of each state"s good loan number and bad loan number
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# 为了便于计算坏账数,我们令坏账为1,好帐为0
loan_status_dict = {"Fully Paid": 0, "Current": 0, "Charged Off": 1, "Late (31-120 days)": 1,
                    "In Grace Period": 1, "Late (16-30 days)": 1, "Default": 1}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_2"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_2"] = loan_data["loan_status_count_2"].astype("float")
# print(loan_data["loan_status_count"].value_counts().to_dict())
# {0.0: 1862171, 1.0: 274902}
loan_status_per_state = loan_data.groupby("addr_state")["loan_status_count_2"].sum().to_dict()
# print(loan_status_per_state)
loan_status_per_state_pd = pd.DataFrame(list(loan_status_per_state.items()),
                                        columns=["state_code", "bad_loan_num"])
loan_status_per_state_pd["state_name"] = None
# print(loan_status_per_state_pd)
for i in range(loan_status_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_status_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Hot"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=True,
             locations=loan_status_per_state_pd["state_code"], z=loan_status_per_state_pd["bad_loan_num"],
             locationmode="USA-states", text=loan_status_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="bad loans num", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="bad loans num per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
loan_data.drop(["loan_status_count_2"], axis=1, inplace=True)
print(loan_data.shape)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="bad loans num per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/bad loans num per state map.png", width=2500, height=1500)

# the map of geographical coordinates of each state"s good loan ratio and bad loan ratio
code_and_name_dict = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
                      "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
                      "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
                      "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
                      "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
                      "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
                      "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
                      "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
                      "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
                      "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
                      "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
# 为了便于计算坏账数,我们令坏账为1,好帐为0
loan_status_dict = {"Fully Paid": 0, "Current": 0, "Charged Off": 1, "Late (31-120 days)": 1,
                    "In Grace Period": 1, "Late (16-30 days)": 1, "Default": 1}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_3"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_3"] = loan_data["loan_status_count_3"].astype("float")
# print(loan_data["loan_status_count"].value_counts().to_dict())
# {0.0: 1862171, 1.0: 274902}
loan_status_per_state = loan_data.groupby("addr_state")["loan_status_count_3"].sum().to_dict()
loan_status_per_state_pd = pd.DataFrame(list(loan_status_per_state.items()),
                                        columns=["state_code", "bad_loan_percent"])
loan_times_per_state_sum_dict = loan_data["addr_state"].value_counts().to_dict()
loan_status_per_state_pd["state_name"] = None
# print(loan_status_per_state_pd)
for i in range(loan_status_per_state_pd.shape[0]):
	state_name = code_and_name_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "state_name"] = state_name
# print(loan_status_per_state_pd)
# print(loan_times_per_state_sum_dict)
for i in range(loan_status_per_state_pd.shape[0]):
	per_state_sum = loan_times_per_state_sum_dict[loan_status_per_state_pd.ix[i, "state_code"]]
	loan_status_per_state_pd.ix[i, "bad_loan_percent"] = float(
		loan_status_per_state_pd.ix[i, "bad_loan_percent"]) / per_state_sum
# print(loan_status_per_state_pd)
# 设立颜色条色彩渐变颜色
# colorscale可选项:["Greys", "YlGnBu", "Greens", "YlOrRd", "Bluered", "RdBu","Reds", "Blues", "Picnic", "Rainbow",
# "Portland", "Jet","Hot", "Blackbody", "Earth", "Electric", "Viridis", "Cividis"]
colorscale = "Reds"
# colorbar为颜色条注释,位置由各州的编号，即缩写表示,z值越高颜色越深
data = [dict(type="choropleth", colorscale=colorscale, autocolorscale=False, reversescale=False,
             locations=loan_status_per_state_pd["state_code"], z=loan_status_per_state_pd["bad_loan_percent"],
             locationmode="USA-states", text=loan_status_per_state_pd["state_name"],
             marker=dict(line=dict(color="rgb(255,255,255)", width=2)),
             colorbar=dict(title="bad loans percent", titlefont=dict(color="rgb(0,0,0)", size=32)))]
layout = dict(title="bad loans percent per state map", titlefont=dict(color="rgb(0,0,0)", size=50),
              geo=dict(scope="usa", projection=dict(type="albers usa")))
fig = dict(data=data, layout=layout)
loan_data.drop(["loan_status_count_3"], axis=1, inplace=True)
print(loan_data.shape)
# filename为网站上个人空间中保存的文件名
py.plot(fig, filename="bad loans percent per state map", auto_open=True)
# filename为本地保存的文件名,plotly本地保存只支持png,svg,jpeg,pdf
py.image.save_as(fig, filename="./pictures/bad loans percent per state map.png", width=2500, height=1500)

# loan purpose word cloud map
loan_times_per_purpose_sum_dict = loan_data["purpose"].value_counts().to_dict()
# print(loan_times_per_purpose_sum_dict)
loan_times_per_purpose_sum_pd = pd.DataFrame(list(loan_times_per_purpose_sum_dict.items()),
                                             columns=["purpose", "loan_times"])
# print(loan_times_per_purpose_sum_pd)
wordcloud = WordCloud(width=1500, height=1000)
wordcloud.add("loan purpose word cloud", loan_times_per_purpose_sum_pd["purpose"],
              loan_times_per_purpose_sum_pd["loan_times"], shape="diamond",
              rotate_step=60, word_size_range=[10, 100])
wordcloud.render(path="./pictures/loan purpose word cloud.html")
wordcloud.render(path="./pictures/loan purpose word cloud.pdf")

# the ratio of good loans and bad loans for each loan purpose
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_4"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_4"] = loan_data["loan_status_count_4"].astype("float")
f_loan_status_purpose, ax_loan_status_purpose = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_4"] = loan_data["loan_status_count_4"].map(loan_status_dict)
sns.barplot(x=loan_data["purpose"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_4"], hue_order=labels,
            ax=ax_loan_status_purpose, estimator=lambda x: len(x) / len(loan_data["loan_status_count_4"]) * 100)
ax_loan_status_purpose.set_title("Loan status per purpose percent", fontsize=16)
ax_loan_status_purpose.set_xticklabels(ax_loan_status_purpose.get_xticklabels(), rotation=45)
ax_loan_status_purpose.set_xlabel("purpose", fontsize=16)
ax_loan_status_purpose.set_ylabel("per purpose loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_4"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_purpose.savefig("./pictures/Loan status per purpose percent bar.jpg", dpi=200,
                              bbox_inches="tight")

# ratio of housing for good loans customer and bad loans customer
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_5"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_5"] = loan_data["loan_status_count_5"].astype("float")
f_loan_status_home, ax_loan_status_home = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_5"] = loan_data["loan_status_count_5"].map(loan_status_dict)
sns.barplot(x=loan_data["home_ownership"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_5"],
            hue_order=labels,
            ax=ax_loan_status_home, estimator=lambda x: len(x) / len(loan_data["loan_status_count_5"]) * 100)
ax_loan_status_home.set_title("Loan status per home percent", fontsize=16)
ax_loan_status_home.set_xlabel("home ownership", fontsize=16)
ax_loan_status_home.set_ylabel("per home ownership loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_5"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_home.savefig("./pictures/Loan status per home ownership percent bar.jpg", dpi=200,
                           bbox_inches="tight")

# ratio of good loans and bad loans for each income level
# 我们将年收入分为三档:20000以下为low，20000-60000为mid，>60000为high
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_6"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_6"] = loan_data["loan_status_count_6"].astype("float")
max_value = loan_data["annual_inc"].max() + 1.0
set_bins = [0.0, 20000.0, 60000.0, max_value]
set_label = ["low", "mid", "high"]
loan_data["income"] = pd.cut(loan_data["annual_inc"], bins=set_bins, labels=set_label)
f_loan_status_income, ax_loan_status_income = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_6"] = loan_data["loan_status_count_6"].map(loan_status_dict)
sns.barplot(x=loan_data["income"], y=loan_data["loan_amnt"], hue=loan_data["loan_status_count_6"], hue_order=labels,
            ax=ax_loan_status_income, estimator=lambda x: len(x) / len(loan_data["loan_status_count_6"]) * 100)
ax_loan_status_income.set_title("Loan status per income percent", fontsize=16)
ax_loan_status_income.set_xlabel("income", fontsize=16)
ax_loan_status_income.set_ylabel("per income loans percent", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_6", "income"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_income.savefig("./pictures/Loan status per income percent bar.jpg", dpi=200,
                             bbox_inches="tight")

# the ratio of good loans and bad loans for each credit rating
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status"].map(loan_status_dict)
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status_count_7"].astype("float")
f_loan_status_grade, ax_loan_status_grade = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data_sorted["loan_status_count_7"] = loan_data_sorted["loan_status_count_7"].map(loan_status_dict)
sns.barplot(x=loan_data_sorted["grade"], y=loan_data_sorted["loan_amnt"], hue=loan_data_sorted["loan_status_count_7"],
            hue_order=labels,
            ax=ax_loan_status_grade, estimator=lambda x: len(x) / len(loan_data_sorted["loan_status_count_7"]) * 100)
ax_loan_status_grade.set_title("Loan status per grade percent", fontsize=16)
ax_loan_status_grade.set_xlabel("grade", fontsize=16)
ax_loan_status_grade.set_ylabel("per grade loans percent", fontsize=16)
plt.show()
plt.close()
print(loan_data.shape)
f_loan_status_grade.savefig("./pictures/Loan status per grade percent bar.jpg", dpi=200,
                            bbox_inches="tight")

# data distribution of loan interest rates
sns.set_style("whitegrid")
f_int_rate, ax_int_rate = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(loan_data["int_rate"], ax=ax_int_rate[0], color="#2F8FF7")
sns.violinplot(y=loan_data["int_rate"], ax=ax_int_rate[1], inner="quartile", palette="Blues")
ax_int_rate[0].set_title("Int rate distribution", fontsize=16)
ax_int_rate[1].set_title("Int rate distribution", fontsize=16)
ax_int_rate[0].set_xlabel("Int rate", fontsize=16)
ax_int_rate[0].set_ylabel("Int rate", fontsize=16)
plt.show()
plt.close()
f_int_rate.savefig("./pictures/Int rate distribution.jpg", dpi=200, bbox_inches="tight")

# average interest rate of good loans and bad loans for each credit ratings
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_8"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_8"] = loan_data["loan_status_count_8"].astype("float")
f_inc_rate_grade, ax_inc_rate_grade = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_8"] = loan_data["loan_status_count_8"].map(loan_status_dict)
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
sns.barplot(x=loan_data_sorted["grade"], y=loan_data_sorted["int_rate"], hue=loan_data_sorted["loan_status_count_8"],
            hue_order=labels, ax=ax_inc_rate_grade, ci=None)
ax_inc_rate_grade.set_title("mean int rate per grade", fontsize=16)
ax_inc_rate_grade.set_xlabel("grade", fontsize=16)
ax_inc_rate_grade.set_ylabel("mean int rate", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_8"], axis=1, inplace=True)
print(loan_data.shape)
f_inc_rate_grade.savefig("./pictures/mean int rate per grade bar.jpg", dpi=200,
                         bbox_inches="tight")

# data distribution of the percentage of the lender"s month-repayments divide the lender"s income
# max_value=loan_data["dti"].max()
# min_value=loan_data["dti"].min()
# print(max_value,min_value)
# 999.0 -1.0 这里的数值应当是百分比
# 该数据中有异常值-1.0,且有很大的离散值,我们需要将其先过滤掉,否则图像效果不好
loan_data_dti = loan_data[loan_data["dti"] <= 100.0]
loan_data_dti = loan_data_dti[loan_data_dti["dti"] > 0.0]
sns.set_style("whitegrid")
f_dti, ax_dti = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(loan_data_dti["dti"], ax=ax_dti[0], color="#F7522F")
sns.violinplot(y=loan_data_dti["dti"], ax=ax_dti[1], inner="quartile", palette="Reds")
ax_dti[0].set_title("dti distribution", fontsize=16)
ax_dti[1].set_title("dti distribution", fontsize=16)
ax_dti[0].set_xlabel("dti", fontsize=16)
ax_dti[1].set_ylabel("dti", fontsize=16)
plt.show()
plt.close()
f_dti.savefig("./pictures/dti distribution.jpg", dpi=200, bbox_inches="tight")

# data distribution of the percentage of the good lender"s month-repayments divide the lender"s income and the percentage of the bad lender"s month-repayments divide the lender"s income
# 请先过滤异常值和极大离散点,只保留0-200之间的数据
loan_data_dti = loan_data[loan_data["dti"] <= 100.0]
loan_data_dti = loan_data_dti[loan_data_dti["dti"] >= 0.0]
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data_dti["loan_status_count_9"] = loan_data_dti["loan_status"].map(loan_status_dict)
loan_data_dti["loan_status_count_9"] = loan_data_dti["loan_status_count_9"].astype("float")
labels = "Bad loans", "Good loans"
# 取出groupby后的分组结果
loans_dti_per_status = dict(list(loan_data_dti.groupby("loan_status_count_9")["dti"]))
good_loan_dti = pd.DataFrame(loans_dti_per_status[1.0], index=None).reset_index(drop=True)
bad_loan_dti = pd.DataFrame(loans_dti_per_status[0.0], index=None).reset_index(drop=True)
# print(good_loan_dti, bad_loan_dti)
# print(good_loan_dti.shape, bad_loan_dti.shape)
sns.set_style("whitegrid")
f_dti_per_loan_status, ax_dti_per_loan_status = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(good_loan_dti["dti"], ax=ax_dti_per_loan_status[0], color="#2F8FF7")
sns.distplot(bad_loan_dti["dti"], ax=ax_dti_per_loan_status[1], color="#F7522F")
ax_dti_per_loan_status[0].set_title("good loans dti distribution", fontsize=16)
ax_dti_per_loan_status[1].set_title("bad loans dti distribution", fontsize=16)
ax_dti_per_loan_status[0].set_xlabel("dti", fontsize=16)
ax_dti_per_loan_status[1].set_ylabel("dti", fontsize=16)
plt.show()
plt.close()
print(loan_data.shape)
f_dti_per_loan_status.savefig("./pictures/dti distribution per loan status.jpg", dpi=200, bbox_inches="tight")

# ratio of short-term and long-term loans for good loans and bad loans
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
loan_data["loan_status_count_10"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_10"] = loan_data["loan_status_count_10"].astype("float")
f_loan_status_term, ax_loan_status_term = plt.subplots(figsize=(15, 10))
labels = "Good loans", "Bad loans"
loan_status_dict = {1.0: "Good loans", 0.0: "Bad loans"}
loan_data["loan_status_count_10"] = loan_data["loan_status_count_10"].map(loan_status_dict)
loan_data_sorted = loan_data.sort_values(by=["grade"], inplace=False)
sns.barplot(x=loan_data_sorted["term"], y=loan_data_sorted["loan_amnt"], hue=loan_data_sorted["loan_status_count_10"],
            hue_order=labels,
            ax=ax_loan_status_term, estimator=lambda x: len(x) / len(loan_data_sorted["loan_status_count_10"]) * 100)
ax_loan_status_term.set_title("loan times per term", fontsize=16)
ax_loan_status_term.set_xlabel("term", fontsize=16)
ax_loan_status_term.set_ylabel("loan times", fontsize=16)
plt.show()
plt.close()
loan_data.drop(["loan_status_count_10"], axis=1, inplace=True)
print(loan_data.shape)
f_loan_status_term.savefig("./pictures/loan times per term bar.jpg", dpi=200,
                           bbox_inches="tight")
```
model_training_and_testing.py
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

loan_data = pd.read_csv("loan_clean_data.csv", low_memory=False)
print(loan_data.shape)

# There is a serious problem of sample imbalance in this dataset
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 用1表示贷款状况良好，用0表示不良贷款
loan_data["loan_status_count_11"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status_count_11"] = loan_data["loan_status_count_11"].astype("float")
loan_data_status_count = loan_data["loan_status_count_11"].value_counts().to_dict()
sum_value = 0.0
for key, value in loan_data_status_count.items():
	sum_value += value
for key, value in loan_data_status_count.items():
	loan_data_status_count[key] = value / sum_value
print(loan_data_status_count)
# {1.0: 0.8713651803190625, 0.0: 0.12863481968093743}
loan_data.drop(["loan_status_count_11"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 87)

# feature:loan_status‘s change to 0 or 1
loan_status_dict = {"Fully Paid": 1, "Current": 1, "Charged Off": 0, "Late (31-120 days)": 0,
                    "In Grace Period": 0, "Late (16-30 days)": 0, "Default": 0}
# 1 is good loan,0 is bad loan
loan_data["loan_status"] = loan_data["loan_status"].map(loan_status_dict)
loan_data["loan_status"] = loan_data["loan_status"].astype("float")
# print(loan_data["emp_length"].value_counts().to_dict)
emp_length_dict = {"10+ years": 10, "2 years": 2, "< 1 year": 0.5, "3 years": 3, "1 year": 1, "5 years": 5,
                   "4 years": 4, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9}
loan_data["emp_length"] = loan_data["emp_length"].map(emp_length_dict)
loan_data["emp_length"] = loan_data["emp_length"].astype("float")
# fill missing value of emp_length to 0
loan_data["emp_length"].fillna(value=0, inplace=True)
# drop some features
loan_data.drop(["emp_title", "title", "zip_code", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"], axis=1,
               inplace=True)
loan_data["month"], loan_data["year"] = loan_data["issue_d"].str.split("-", 1).str
loan_data.drop(["issue_d"], axis=1, inplace=True)
print(loan_data.shape)
# (2137073, 82)


numerical_feature_name = loan_data.columns[(loan_data.dtypes == "float64") | (loan_data.dtypes == "int64")].tolist()
# category_feature_name = loan_data.columns[loan_data.dtypes == "object"].tolist()
# draw pearson correlation coefficient matrix
# 若>0，表明两个变量是正相关,即一个变量的值越大，另一个变量的值也会越大
# 若<0，表明两个变量是负相关，即一个变量的值越大另一个变量的值反而会越小
# 若r=0，表明两个变量间不是线性相关，但有可能是非线性相关
corrmat = loan_data[numerical_feature_name].corr()
f, ax = plt.subplots(figsize=(30, 20))
# vmax、vmin即热力图颜色取值的最大值和最小值,默认会从data中推导
# square=True会将单元格设为正方形
sns.heatmap(corrmat, square=True, ax=ax, cmap="Blues", linewidths=0.5)
ax.set_title("Correlation coefficient matrix", fontsize=16)
ax.set_xlabel("feature names", fontsize=16)
ax.set_ylabel("feature names", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/Correlation coefficient matrix.jpg", dpi=200, bbox_inches="tight")

# split features and labels
X = loan_data.ix[:, loan_data.columns != "loan_status"]
Y = loan_data["loan_status"]
# Category features convert to one_hot code
X = pd.get_dummies(X, drop_first=True)
print(X.shape)
# (2137073, 200)

# divide training sets and testing sets
numerical_feature_name_2 = X.columns[(X.dtypes == "float64") | (X.dtypes == "int64")].tolist()
category_feature_name_2 = X.columns[X.dtypes == "uint8"].tolist()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (1709658, 200) (427415, 200) (1709658,) (427415,)
print(len(numerical_feature_name_2), len(category_feature_name_2))
# 67 133
# 此时类别型特征已经全部变为one_hot编码(k-1列),数值型特征还需要归一化,由于特征值中有异常值,我们使用RobustScaler方法归一化
x_train_num, x_train_cat = x_train[numerical_feature_name_2], x_train[category_feature_name_2]
x_test_num, x_test_cat = x_test[numerical_feature_name_2], x_test[category_feature_name_2]
# get feature names
feature_names = list(x_train_num.columns)
feature_names.extend(list(x_train_cat.columns))
feature_names=np.array(feature_names)
print(feature_names.shape)
# 200

# robust scalar,默认为第一分位数到第四分位数之间的范围计算均值和方差,归一化还是z_score标准化
rob_scaler = RobustScaler()
x_train_num_rob = rob_scaler.fit_transform(x_train_num)
x_test_num_rob = rob_scaler.transform(x_test_num)
x_train_nom_pd = pd.DataFrame(np.hstack((x_train_num_rob, x_train_cat)))
x_test_nom_pd = pd.DataFrame(np.hstack((x_test_num_rob, x_test_cat)))
y_test_pd = pd.DataFrame(y_test)
x_train_sm_np, y_train_sm_np = x_train_nom_pd, y_train
print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# (1709658, 200) (1709658,) (427415, 200) (427415,)


# # we can choose class_weight="balanced" to deal with sample imbalance problem when we use logistic model
# # if we use random foreast model or xgboost model,we don’t need to deal with sample imbalance problem
# # besides,we can also use SMOTE to generate some sample of the category with low number of samples,but it needs over 16GB memory
# # so,if you want to use SMOTE,you can run all these code on a local computer with at least 32GB memory
# # SMOTE算法即对于少数类中的每一个样本a,执行N次下列操作:
# # 从k个最近邻样本中随机选择一个样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
# # n_jobs=-1表示使用所有CPU
# sm = SMOTE(k_neighbors=10, random_state=0, n_jobs=-1)
# x_train_sm_np, y_train_sm_np = sm.fit_sample(x_train_nom_pd, y_train)
# print(x_train_sm_np.shape, y_train_sm_np.shape)


# x_train_sm_pd = pd.DataFrame(x_train_sm_np)
# y_train_sm_pd = pd.DataFrame(y_train_sm_np)
# x_train_sm_pd.to_csv("x_train_sm_np.csv", index=None)
# y_train_sm_pd.to_csv("y_train_sm_np.csv", index=None)
# x_test_nom_pd.to_csv("x_test_nom_np.csv", index=None)
# y_test_pd.to_csv("y_test.csv", index=None)
# x_train_sm_np = np.array(pd.read_csv("x_train_sm_np.csv", low_memory=False))
# y_train_sm_np = np.array(pd.read_csv("y_train_sm_np.csv", low_memory=False))
# x_test_nom_pd = np.array(pd.read_csv("x_test_nom_np.csv", low_memory=False))
# y_test = np.array(pd.read_csv("y_test.csv", low_memory=False))
# print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# # (2980148, 200) (2980148, 1) (427415, 200) (427415, 1)
# # 标签需要降维成一维数组
# y_train_sm_np = y_train_sm_np.ravel()
# y_test = y_test.ravel()
# print(x_train_sm_np.shape, y_train_sm_np.shape, x_test_nom_pd.shape, y_test.shape)
# # (2980148, 200) (2980148,) (427415, 200) (427415,)


# if your computer's memory is not more than 16GB,it is not enough to run three models at the same time,you can choose one model to run.


# use logistic regression model to train and predict
# jobs=-1使用所有CPU进行运算
# sag即随机平均梯度下降，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候
# class_weight="balanced"根据用来训练的样本的各个类别的比例确定权重
print("use logistic model to train and predict")
lr = LogisticRegression(solver="sag", class_weight="balanced", n_jobs=-1)
lr.fit(x_train_sm_np, y_train_sm_np)
lr_y_pred = lr.predict(x_test_nom_pd)
lr_test_acc = accuracy_score(y_test, lr_y_pred)
lr_classification_score = classification_report(y_test, lr_y_pred)
print("Lr model test accuracy:{:.2f}".format(lr_test_acc))
print("Lr model classification_score:\n", lr_classification_score)
lr_confusion_score = confusion_matrix(y_test, lr_y_pred)
f_lr, ax_lr = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
lr_cm_pred_label_sum = lr_confusion_score.sum(axis=0)
lr_cm_true_label_sum = lr_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
lr_model_precision, lr_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
lr_model_precision[0][0], lr_model_precision[1][0] = lr_confusion_score[0][0] / lr_cm_pred_label_sum[0], \
                                                     lr_confusion_score[1][0] / lr_cm_pred_label_sum[0]
lr_model_precision[0][1], lr_model_precision[1][1] = lr_confusion_score[0][1] / lr_cm_pred_label_sum[1], \
                                                     lr_confusion_score[1][1] / lr_cm_pred_label_sum[1]
lr_model_recall[0][0], lr_model_recall[0][1] = lr_confusion_score[0][0] / lr_cm_true_label_sum[0], \
                                               lr_confusion_score[0][1] / lr_cm_true_label_sum[0]
lr_model_recall[1][0], lr_model_recall[1][1] = lr_confusion_score[1][0] / lr_cm_true_label_sum[1], \
                                               lr_confusion_score[1][1] / lr_cm_true_label_sum[1]
sns.heatmap(lr_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_lr[0], square=True, linewidths=0.5)
sns.heatmap(lr_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_lr[1], square=True, linewidths=0.5)
sns.heatmap(lr_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_lr[2], square=True, linewidths=0.5)
ax_lr[0].set_title("lr confusion matrix", fontsize=16)
ax_lr[1].set_title("lr model precision", fontsize=16)
ax_lr[2].set_title("lr model recall", fontsize=16)
ax_lr[0].set_xlabel("Predicted label", fontsize=16)
ax_lr[0].set_ylabel("True label", fontsize=16)
ax_lr[1].set_xlabel("Predicted label", fontsize=16)
ax_lr[1].set_ylabel("True label", fontsize=16)
ax_lr[2].set_xlabel("Predicted label", fontsize=16)
ax_lr[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_lr.savefig("./pictures/lr model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# result
# Lr model test accuracy:0.88
# Lr model classification_score:
#                precision    recall  f1-score   support
#          0.0       0.54      0.73      0.62     55318
#          1.0       0.96      0.91      0.93    372097
#    micro avg       0.88      0.88      0.88    427415
#    macro avg       0.75      0.82      0.78    427415
# weighted avg       0.90      0.88      0.89    427415


# use randomforest model to train and predict
print("use randomforest model to train and predict")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(x_train_sm_np, y_train_sm_np)
rf_y_pred = rf.predict(x_test_nom_pd)
rf_test_acc = accuracy_score(y_test, rf_y_pred)
rf_classification_score = classification_report(y_test, rf_y_pred)
print("Rf model test accuracy:{:.4f}".format(rf_test_acc))
print("rf model classification_score:\n", rf_classification_score)
rf_confusion_score = confusion_matrix(y_test, rf_y_pred)
# print(rf_confusion_score)
f_rf, ax_rf = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
rf_cm_pred_label_sum = rf_confusion_score.sum(axis=0)
rf_cm_true_label_sum = rf_confusion_score.sum(axis=1)
# 计算正样本和负样本的精确率和召回率
rf_model_precision, rf_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
rf_model_precision[0][0], rf_model_precision[1][0] = rf_confusion_score[0][0] / rf_cm_pred_label_sum[0], \
                                                     rf_confusion_score[1][0] / rf_cm_pred_label_sum[0]
rf_model_precision[0][1], rf_model_precision[1][1] = rf_confusion_score[0][1] / rf_cm_pred_label_sum[1], \
                                                     rf_confusion_score[1][1] / rf_cm_pred_label_sum[1]
rf_model_recall[0][0], rf_model_recall[0][1] = rf_confusion_score[0][0] / rf_cm_true_label_sum[0], \
                                               rf_confusion_score[0][1] / rf_cm_true_label_sum[0]
rf_model_recall[1][0], rf_model_recall[1][1] = rf_confusion_score[1][0] / rf_cm_true_label_sum[1], \
                                               rf_confusion_score[1][1] / rf_cm_true_label_sum[1]
sns.heatmap(rf_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_rf[0], square=True, linewidths=0.5)
sns.heatmap(rf_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[1], square=True, linewidths=0.5)
sns.heatmap(rf_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_rf[2], square=True, linewidths=0.5)
ax_rf[0].set_title("rf confusion matrix", fontsize=16)
ax_rf[1].set_title("rf model precision", fontsize=16)
ax_rf[2].set_title("rf model recall", fontsize=16)
ax_rf[0].set_xlabel("Predicted label", fontsize=16)
ax_rf[0].set_ylabel("True label", fontsize=16)
ax_rf[1].set_xlabel("Predicted label", fontsize=16)
ax_rf[1].set_ylabel("True label", fontsize=16)
ax_rf[2].set_xlabel("Predicted label", fontsize=16)
ax_rf[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_rf.savefig("./pictures/rf model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# Rf model test accuracy:0.9828
# rf model classification_score:
#                precision    recall  f1-score   support
#
#          0.0       1.00      0.87      0.93     55318
#          1.0       0.98      1.00      0.99    372097
#
#    micro avg       0.98      0.98      0.98    427415
#    macro avg       0.99      0.93      0.96    427415
# weighted avg       0.98      0.98      0.98    427415


# random forest model feature contribution visualization
feature_importances = rf.feature_importances_
# print(feature_importances)
# y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
indices = np.argsort(feature_importances)[::-1]
# 只取贡献度最高的30个特征来作图
show_indices = indices[0:30]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=feature_importances[show_indices], y=feature_names[show_indices], ax=ax)
ax.set_title("rf model feature importance top30", fontsize=16)
ax.set_xlabel("feature importance score", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/rf model feature importance top30.jpg", dpi=200, bbox_inches="tight")

# use XGBoost model to train and predict
xgb = XGBClassifier(n_estimators=200, nthread=-1)
xgb.fit(x_train_sm_np, y_train_sm_np)
xgb_y_pred = xgb.predict(x_test_nom_pd)
xgb_test_acc = accuracy_score(y_test, xgb_y_pred)
xgb_classification_score = classification_report(y_test, xgb_y_pred)
print("Xgb model test accuracy:{:.4f}".format(xgb_test_acc))
print("Xgb model classification_score:\n", xgb_classification_score)
xgb_confusion_score = confusion_matrix(y_test, xgb_y_pred)
# print(xgb_confusion_score)
f_xgb, ax_xgb = plt.subplots(1, 3, figsize=(15, 10))
# 混淆矩阵的y轴为true label,x轴为pred label
# 精确率,如对正类 ,所有预测为正类样本中中真实的正类占所有预测为正类的比例
# 召回率,如对正类,所有真实的正类样本中有多少被预测为正类的比例
# 分别计算预测预测的正样本数和负样本数以及真实的正样本数和负样本数
xgb_cm_pred_label_sum = xgb_confusion_score.sum(axis=0)
xgb_cm_true_label_sum = xgb_confusion_score.sum(axis=1)
# print(xgb_cm_pred_label_sum,xgb_cm_true_label_sum)
# 计算正样本和负样本的精确率和召回率
xgb_model_precision, xgb_model_recall = np.empty([2, 2], dtype=float), np.empty([2, 2], dtype=float)
xgb_model_precision[0][0], xgb_model_precision[1][0] = xgb_confusion_score[0][0] / xgb_cm_pred_label_sum[0], \
                                                       xgb_confusion_score[1][0] / xgb_cm_pred_label_sum[0]
xgb_model_precision[0][1], xgb_model_precision[1][1] = xgb_confusion_score[0][1] / xgb_cm_pred_label_sum[1], \
                                                       xgb_confusion_score[1][1] / xgb_cm_pred_label_sum[1]
xgb_model_recall[0][0], xgb_model_recall[0][1] = xgb_confusion_score[0][0] / xgb_cm_true_label_sum[0], \
                                                 xgb_confusion_score[0][1] / xgb_cm_true_label_sum[0]
xgb_model_recall[1][0], xgb_model_recall[1][1] = xgb_confusion_score[1][0] / xgb_cm_true_label_sum[1], \
                                                 xgb_confusion_score[1][1] / xgb_cm_true_label_sum[1]
sns.heatmap(xgb_confusion_score, annot=True, fmt="d", cmap="Blues", ax=ax_xgb[0], square=True, linewidths=0.5)
sns.heatmap(xgb_model_precision, annot=True, fmt=".5f", cmap="Blues", ax=ax_xgb[1], square=True, linewidths=0.5)
sns.heatmap(xgb_model_recall, annot=True, fmt=".5f", cmap="Blues", ax=ax_xgb[2], square=True, linewidths=0.5)
ax_xgb[0].set_title("xgb confusion matrix", fontsize=16)
ax_xgb[1].set_title("xgb model precision", fontsize=16)
ax_xgb[2].set_title("xgb model recall", fontsize=16)
ax_xgb[0].set_xlabel("Predicted label", fontsize=16)
ax_xgb[0].set_ylabel("True label", fontsize=16)
ax_xgb[1].set_xlabel("Predicted label", fontsize=16)
ax_xgb[1].set_ylabel("True label", fontsize=16)
ax_xgb[2].set_xlabel("Predicted label", fontsize=16)
ax_xgb[2].set_ylabel("True label", fontsize=16)
plt.show()
plt.close()
f_xgb.savefig("./pictures/xgb model confusion matrix.jpg", dpi=200, bbox_inches="tight")
# Xgb model test accuracy:0.9809
# Xgb model classification_score:
#                precision    recall  f1-score   support
#
#          0.0       1.00      0.85      0.92     55318
#          1.0       0.98      1.00      0.99    372097
#
#    micro avg       0.98      0.98      0.98    427415
#    macro avg       0.99      0.93      0.95    427415
# weighted avg       0.98      0.98      0.98    427415


# xgboost model feature contribution visualization
feature_importances = xgb.feature_importances_
# print(feature_importances)
# y=x.argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
indices = np.argsort(feature_importances)[::-1]
# 只取贡献度最高的30个特征来作图
show_indices = indices[0:30]
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x=feature_importances[show_indices], y=feature_names[show_indices], ax=ax)
ax.set_title("xgb model feature importance top30", fontsize=16)
ax.set_xlabel("feature importance score", fontsize=16)
ax.set_ylabel("feature name", fontsize=16)
plt.show()
plt.close()
f.savefig("./pictures/xgb model feature importance top30.jpg", dpi=200, bbox_inches="tight")
```