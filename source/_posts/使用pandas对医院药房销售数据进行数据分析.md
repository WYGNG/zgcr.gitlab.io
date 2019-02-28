---
title: 使用pandas对医院药房销售数据进行数据分析
date: 2019-02-28 11:32:50
tags:
- 数据挖掘
categories:
- 数据挖掘
---

# 数据集下载
数据集使用某医院药房2018年1月-7月的销售数据，数据集下载地址：
链接：https://pan.baidu.com/s/16ai4TiubDg0rBD0QKR3HTQ 提取码：x9m0 
# 数据分析指标和图表
商品的种类、商品的总销售数量、总销售次数、月均销售次数、总销售金额、平均每月销售金额、平均每单销售金额、最大和最小日销售金额及对应日期、最大和最小日销售数量及对应日期。
各个月单独的每日销售金额直方图、每月销售金额直方图、每月销售金额变化直线图、销量前十和最后十名的药品的直方图。
# 数据分析步骤
导入数据集；
检查数据基本属性；
修改个别表头、舍弃缺失的不完整数据行；
对时间这一列信息进行特殊处理，舍弃星期的信息，然后将时间这一列数据的格式转为时间格式pd.to_datetime()；
“销售数量”、“应收金额”、“实收金额”这三列数据显然不可能有负数，我们要舍弃掉一些异常值的数据行；
对数据按时间升序排序，重置其索引；
计算商品的种类和商品的总销售数量；
计算总销售次数，月份数，月均销售次数；
计算总销售金额，平均每月销售金额，平均每单销售金额；
计算最大和最小日销售金额及对应的日期、最大和最小日销售数量及对应的日期；
画出各个月单独的每日销售金额直方图、每月销售金额直方图、每月销售金额变化直线图、销量前十和最后十名的药品的直方图。
# 代码实现
```python
import os
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl  # 用于画图时显示中文字符

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 使用matplotlib画图时如果出现中文使用仿宋字体
file_data = pd.read_excel("chaoyang2018sale.xlsx")  # 读取数据文件


# 获取文件名
def get_file_name(fl_data):
   # items()函数以列表返回可遍历的(键, 值)元组数组
   # globals()是一个字典，存储了所有的全局变量的名字和对应的变量的值
   # 在这里我们通过判断输入的变量的值与globals()中存储的变量的值是否相等，相等就认为我们输入的变量就是这个globals()中存储的变量
   # 这样我们就返回这个globals()中存储的变量的变量名，即key
   for var_name, value in globals().items():
      if value is fl_data:
         return var_name


# 显示读入的文件数据的一些基本属性
def show_data_basic_description(fl_data):
   describe_label_shape = get_file_name(fl_data) + "文件的shape大小："
   print(describe_label_shape, file_data.shape)
   describe_label_index = get_file_name(fl_data) + "文件的索引开头和末尾："
   print(describe_label_index, file_data.index)
   describe_label_table_head = get_file_name(fl_data) + "文件的表头项："
   print(describe_label_table_head, file_data.columns)
   describe_label_top_five_line = get_file_name(fl_data) + "文件的内容：\n"
   print(describe_label_top_five_line, file_data.head())
   describe_label_data_type = get_file_name(fl_data) + "文件的各项的数据类型：\n"
   print(describe_label_data_type, file_data.dtypes)


print("文件预处理前的各项属性：")
show_data_basic_description(file_data)
# 其中一个列的表头重命名一下
file_data.rename(columns={"购药时间": "销售时间"}, inplace=True)

# 删除所有有缺失数据的行
file_data = file_data.dropna(subset=["销售时间", "社保卡号", "商品编码", "商品名称", "销售数量", "应收金额", "实收金额"], how="any")
# 销售时间这一列数据中的日期和星期使用split函数进行分割，分割后的时间，返回的是Series数据类型
time_list = file_data.loc[:, "销售时间"]
# 我们只需要日期数据,因此舍弃星期数据
data = []
for time in time_list:
   data.append(time.split(" ")[0])

file_data.loc[:, "销售时间"] = data

# 将时间数据转为时间格式,errors = 'coerce':不可扩展，缺失值返回NaT（Not a Time），结果认为DatetimeIndex
file_data.loc[:, "销售时间"] = pd.to_datetime(file_data.loc[:, "销售时间"], format="%Y-%m-%d", errors="coerce")
# 删除无效时间的数据
file_data = file_data.dropna(subset=["销售时间"], how="any")
# 处理异常值：“销售数量”、“应收金额”、“实收金额”这三列数据显然不可能有负数，因此要去掉不合理的数据
pop = file_data.loc[:, "销售数量"] > 0
file_data = file_data.loc[pop, :]
# 对数据按时间排序,ascending=True表示升序排列
file_data = file_data.sort_values(by="销售时间", ascending=True)

# 重置数据的索引
file_data = file_data.reset_index(drop=True)
print("文件预处理后的各项属性：")
show_data_basic_description(file_data)


# 计算商品的种类和商品的总销售数量
def compute_commodity_type_and_sum(fl_data):
   # 去除重复的商品种类
   no_duplicate_file_data = fl_data.drop_duplicates(subset=["商品编码"])
   # 计算商品的种类
   fl_data_total_consumption_type = no_duplicate_file_data.shape[0]
   # 计算商品的总销售数量
   fl_data = fl_data.dropna(subset=["商品编码"], how="any")
   fl_data_total_sum = fl_data.loc[:, "销售数量"].sum()
   return fl_data_total_consumption_type, fl_data_total_sum


file_data_total_consumption_type, file_data_total_sum = compute_commodity_type_and_sum(file_data)
print("销售商品的种类:{} 销售商品的数量:{}".format(file_data_total_consumption_type, int(file_data_total_sum)))


# 计算总销售次数，月份数，月均销售次数，月均销售次数 = 总销售次数 / 月份数
def compute_consumption_num(fl_data):
   # 先删除重复的数据,因为一个人一次消费可能买好几种药
   no_duplicate_file_data = fl_data.drop_duplicates(subset=["销售时间", "社保卡号"])
   # 计算总销售次数
   fl_data_total_consumption_num = no_duplicate_file_data.shape[0]
   # 计算月份数
   # 先给上面删除了重复数据的文件排序，重新组织索引
   no_duplicate_file_data = no_duplicate_file_data.sort_values(by="销售时间", ascending=True)
   no_duplicate_file_data = no_duplicate_file_data.reset_index(drop=True)
   # 最早购买时间，最晚购买时间，计算天数和月份数
   start_time = no_duplicate_file_data.loc[0, "销售时间"]
   end_time = no_duplicate_file_data.loc[fl_data_total_consumption_num - 1, "销售时间"]
   fl_data_days_count = (end_time - start_time).days
   # //表示除以后返回商的整数部分
   fl_data_months_count = fl_data_days_count // 30
   # 计算月均销售次数，用总销售次数除以月数
   fl_data_per_month_average_consumption_num = fl_data_total_consumption_num / fl_data_months_count
   return fl_data_total_consumption_num, fl_data_months_count, fl_data_per_month_average_consumption_num


total_consumption_num, months_count, per_month_average_consumption_num = compute_consumption_num(file_data)
print("总销售次数为:{} 月份数为:{} 月均销售次数为:{:.1f}".format(total_consumption_num, months_count, per_month_average_consumption_num))


# 计算总销售金额，平均每月销售金额，平均每单销售金额
def compute_consumption_money(fl_data, ttl_consumption_num, mth_count):
   # 计算月均销售金额，先计算总销售金额，再除以月数
   fl_data_total_consumption_money = fl_data.loc[:, "实收金额"].sum()
   fl_data_per_month_consumption_money = fl_data_total_consumption_money / mth_count
   # 计算平均每单销售金额，用总销售金额除以总销售次数
   fl_data_per_sale_consumption_money = fl_data_total_consumption_money / ttl_consumption_num
   return fl_data_total_consumption_money, fl_data_per_month_consumption_money, fl_data_per_sale_consumption_money


total_consumption_money, per_month_consumption_money, per_sale_consumption_money = compute_consumption_money(file_data,
                                                                                                             total_consumption_num,
                                                                                                             months_count)
print("总销售金额为:{:.2f} 月均销售金额为:{:.2f} 平均每单销售金额为:{:.2f}".format(total_consumption_money, per_month_consumption_money,
                                                             per_sale_consumption_money))


# 最大和最小销售金额的对应的日期
def max_and_min_consumption_money_day(fl_data):
   fl_data_money = fl_data[['销售时间', '实收金额']]
   fl_data_money.index = fl_data_money['销售时间']
   fl_data_money_group_day = fl_data_money.groupby(fl_data_money.index)
   fl_data_money_group_day = fl_data_money_group_day.sum()
   # 按求最大销售金额和最小销售金额及对应的日期
   fl_data_money_group_day = fl_data_money_group_day.sort_values(by='实收金额', ascending=False)
   max_money_fl_data_group_day = fl_data_money_group_day.iloc[0:1, :]
   fl_data_money_group_day = fl_data_money_group_day.sort_values(by='实收金额', ascending=True)
   min_money_fl_data_group_day = fl_data_money_group_day.iloc[0:1, :]
   return max_money_fl_data_group_day, min_money_fl_data_group_day


max_sum_group_day, min_sum_group_day = max_and_min_consumption_money_day(file_data)
print("最大每日销售金额及对应的日期：\n", max_sum_group_day)
print("最小每日销售金额及对应的日期：\n", min_sum_group_day)


# 最大和最小销售数量的对应的日期
def max_and_min_consumption_sum_day(fl_data):
   fl_data_sum = fl_data[['销售时间', '销售数量']]
   fl_data_sum.index = fl_data_sum['销售时间']
   fl_data_sum_group_day = fl_data_sum.groupby(fl_data_sum.index)
   fl_data_sum_group_day = fl_data_sum_group_day.sum()
   # 按求最大销售金额和最小销售金额及对应的日期
   fl_data_sum_group_day = fl_data_sum_group_day.sort_values(by='销售数量', ascending=False)
   max_sum_fl_data_group_day = fl_data_sum_group_day.iloc[0:1, :]
   fl_data_sum_group_day = fl_data_sum_group_day.sort_values(by='销售数量', ascending=True)
   min_sum_fl_data_group_day = fl_data_sum_group_day.iloc[0:1, :]
   return max_sum_fl_data_group_day, min_sum_fl_data_group_day


max_sum_group_day, min_sum_group_day = max_and_min_consumption_sum_day(file_data)
print("最大每日销售数量及对应的日期：\n", max_sum_group_day)
print("最小每日销售数量及对应的日期：\n", min_sum_group_day)

# 在使用matplotlib画图时使用的数据是file_data的copy，防止画图时影响file_data的数据
file_data_copy = file_data.copy()


# 对文件数据按时间升序排序，并在排序后重置索引
def sort_data_value_as_time(fl_data):
   # 对数据按时间排序,ascending=True表示升序排列
   fl_data = fl_data.sort_values(by="销售时间", ascending=True)
   # 重置数据的索引,重置后索引变成按时间排序后的索引了
   fl_data = fl_data.reset_index(drop=True)
   return fl_data


# 画各个月单独的每日销售金额直方图
def draw_and_save_per_day_per_month_consumption_bar(fl_data):
   # 对文件数据按时间升序排序，并在排序后重置索引
   fl_data = sort_data_value_as_time(fl_data)
   # 如果图片保存路径不存在则创建
   if not os.path.exists("./count_image/"):
      os.mkdir("./count_image")
   fl_data_money = fl_data[['销售时间', '实收金额']]
   fl_data_money.index = fl_data_money['销售时间']
   # 画图
   for index in range(1, 8, 1):
      month_index = "2018-" + str(index)
      per_mont_fl_data_money = fl_data_money[month_index]
      per_mont_fl_data_money_day_group = per_mont_fl_data_money.groupby(per_mont_fl_data_money.index.day)
      per_mont_fl_data_money_day_group = per_mont_fl_data_money_day_group.sum()
      per_mont_fl_data_money_day_group.plot(figsize=(12, 8), kind='bar')
      image_title = str(index) + "月每日销售金额直方图"
      plt.title(image_title)
      plt.xlabel('时间')
      plt.ylabel('实收金额')
      save_path = "./count_image/per_day_consumption_month_" + str(index) + "_bar.jpg"
      plt.savefig(save_path)
      plt.close()


draw_and_save_per_day_per_month_consumption_bar(file_data_copy)


# 画出按月份的销售金额直方图
def draw_and_save_per_month_consumption_bar(fl_data):
   # 对文件数据按时间升序排序，并在排序后重置索引
   fl_data = sort_data_value_as_time(fl_data)
   # 如果图片保存路径不存在则创建
   if not os.path.exists("./count_image/"):
      os.mkdir("./count_image")
   fl_data_money = fl_data[['销售时间', '实收金额']]
   fl_data_money.index = fl_data_money['销售时间']
   fl_data_money_month_group = fl_data_money.groupby(fl_data_money.index.month)
   fl_data_money_month_group_count = fl_data_money_month_group.sum()
   fl_data_money_month_group_count.plot(figsize=(12, 8), kind='bar')
   plt.title('每月销售金额直方图')
   plt.xlabel('月份')
   plt.ylabel('销售金额')
   plt.legend(loc="best")
   plt.savefig("./count_image/per_month_consumption_bar.jpg")
   plt.close()


draw_and_save_per_month_consumption_bar(file_data_copy)


# 画出按月份的销售金额变化直线图
def draw_and_save_per_month_consumption_line(fl_data):
   # 将销售时间聚合按月分组
   fl_data.index = fl_data['销售时间']
   fl_data_month_group = fl_data.groupby(fl_data.index.month)
   months_consumption_count = fl_data_month_group.sum()
   # 画出按月份的销售金额变化直方图
   plt.figure(figsize=(12, 8))
   plt.plot(months_consumption_count['实收金额'])
   plt.title('每月销售金额变化图')
   plt.xlabel('月份')
   plt.ylabel('实收金额')
   plt.savefig("./count_image/per_month_consumption_line.jpg")
   plt.close()


draw_and_save_per_month_consumption_line(file_data_copy)


# 画出销量前十和最后十名的药品的直方图
def draw_and_save_first_ten_medicine_bar(fl_data):
   # 统计各种药品的销售数量
   medicine_data = fl_data[['商品名称', '销售数量']]
   medicine_data_group = medicine_data.groupby('商品名称')[['销售数量']]
   medicine_data_group_count = medicine_data_group.sum()
   # 对统计的每种药品销售数量进行降序排序
   medicine_data_group_count = medicine_data_group_count.sort_values(by='销售数量', ascending=False)
   # 截取销售数量最多和最少的十种药品
   top_ten_medicine = medicine_data_group_count.iloc[0:10, :]
   under_ten_medicine = medicine_data_group_count.iloc[-11:-1, :]
   # 最后十名升序排序
   under_ten_medicine = under_ten_medicine.sort_values(by="销售数量", ascending=True)
   # 画图，kind='bar'表示用条形图
   top_ten_medicine.plot(figsize=(12, 8), kind='bar')
   plt.title('前十名销量的药品')
   plt.xlabel('药品种类')
   plt.ylabel('销售数量')
   plt.legend(loc="best")
   plt.savefig("./count_image/top_ten_medicine_sale_graph.jpg")
   plt.close()

   under_ten_medicine.plot(figsize=(12, 8), kind='bar')
   plt.title('最后十名销量的药品')
   plt.xlabel('药品种类')
   plt.ylabel('销售数量')
   plt.legend(loc="best")
   plt.savefig("./count_image/under_ten_medicine_sale_graph.jpg")
   plt.close()


draw_and_save_first_ten_medicine_bar(file_data_copy)
```
运行结果如下：
```python
文件预处理前的各项属性：
file_data文件的shape大小： (6578, 7)
file_data文件的索引开头和末尾： RangeIndex(start=0, stop=6578, step=1)
file_data文件的表头项： Index(['购药时间', '社保卡号', '商品编码', '商品名称', '销售数量', '应收金额', '实收金额'], dtype='object')
file_data文件的内容：
              购药时间          社保卡号      商品编码     商品名称  销售数量   应收金额    实收金额
0  2018-01-01 星期五  1.616528e+06  236701.0  强力VC银翘片   6.0   82.8   69.00
1  2018-01-02 星期六  1.616528e+06  236701.0  清热解毒口服液   1.0   28.0   24.64
2  2018-01-06 星期三  1.260283e+07  236701.0       感康   2.0   16.8   15.00
3  2018-01-11 星期一  1.007034e+10  236701.0    三九感冒灵   1.0   28.0   28.00
4  2018-01-15 星期五  1.015543e+08  236701.0    三九感冒灵   8.0  224.0  208.00
file_data文件的各项的数据类型：
 购药时间     object
社保卡号    float64
商品编码    float64
商品名称     object
销售数量    float64
应收金额    float64
实收金额    float64
dtype: object
文件预处理后的各项属性：
file_data文件的shape大小： (6509, 7)
file_data文件的索引开头和末尾： RangeIndex(start=0, stop=6509, step=1)
file_data文件的表头项： Index(['销售时间', '社保卡号', '商品编码', '商品名称', '销售数量', '应收金额', '实收金额'], dtype='object')
file_data文件的内容：
         销售时间          社保卡号      商品编码           商品名称  销售数量   应收金额  实收金额
0 2018-01-01  1.616528e+06  236701.0        强力VC银翘片   6.0   82.8  69.0
1 2018-01-01  1.616528e+06  861417.0     雷米普利片(瑞素坦)   1.0   28.5  28.5
2 2018-01-01  1.344823e+07  861507.0  苯磺酸氨氯地平片(安内真)   1.0    9.5   8.5
3 2018-01-01  1.007397e+10  866634.0    硝苯地平控释片(欣然)   6.0  111.0  92.5
4 2018-01-01  1.174343e+07  861405.0  苯磺酸氨氯地平片(络活喜)   1.0   34.5  31.0
file_data文件的各项的数据类型：
 销售时间    datetime64[ns]
社保卡号           float64
商品编码           float64
商品名称            object
销售数量           float64
应收金额           float64
实收金额           float64
dtype: object
销售商品的种类:85 销售商品的数量:15656
总销售次数为:5345 月份数为:6 月均销售次数为:890.8
总销售金额为:304034.97 月均销售金额为:50672.49 平均每单销售金额为:56.88
最大每日销售金额及对应的日期：
                实收金额
销售时间               
2018-04-15  5254.23
最小每日销售金额及对应的日期：
             实收金额
销售时间            
2018-02-08  56.5
最大每日销售数量及对应的日期：
              销售数量
销售时间             
2018-04-15  499.0
最小每日销售数量及对应的日期：
             销售数量
销售时间            
2018-02-08   7.0

Process finished with exit code 0
```
同时还生成了我们需要的图表。