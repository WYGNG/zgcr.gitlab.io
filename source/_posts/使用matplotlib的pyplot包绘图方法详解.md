---
title: 使用matplotlib的pyplot包绘图方法详解
date: 2019-02-28 15:41:30
tags:
- Python
categories:
- Python
---

# 使用Pycharm时matplotlib绘制的图片不弹出独立窗口
在File->Settings->Tools->Python Scientific中去除勾选Show plots in toolwindow即可。
# 引入matplotlib.pyplot包与全局中文字体设置
引用该包语句：
```python
import matplotlib.pyplot as plt
```
pyplot包并不默认支持中文显示，需要.rcParams()方法修改字体来实现。
使用下面的语句：
```python
from pylab import mpl  # 用于画图时显示中文字符

# 定义matplotlib画图时的全局使用仿宋中文字体，如果不定义成中文字体，有中文时不能正常显示
mpl.rcParams['font.sans-serif'] = ['FangSong']
```
# plot快速绘图
适用于只有一个figure对象，且figure对象中只有一个子区域时。此时不需要再创建figure对象和subplot子区域，系统会默认创建。
举例：
```python
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl  # 用于画图时显示中文字符

# 定义matplotlib画图时的全局使用仿宋中文字体，如果不定义成中文字体，那么遇到有中文时不能正常显示
mpl.rcParams['font.sans-serif'] = ['FangSong']
# 创建x数据，np.linspace()默认创建50个数据的等差数列
x = np.linspace(0, 2 * np.pi)
# 创建y数据
y_sin = np.sin(x)
y_cos = np.cos(x)

# 配置figure区域的大小，10和6指1000X600像素
plt.rcParams['figure.figsize'] = (10, 6)

# 系统默认创建一个figure，这个figure默认只有一个子区域，在这个子区域中创建一条曲线
plt.plot(x, y_sin, color='blue', linewidth=3, marker='+', linestyle='-', markersize=12, label=r'$y=sin{x}$')
# 再创建一条曲线
plt.plot(x, y_cos, color='red', linewidth=3, linestyle='-', marker='*', markersize=12, label=r'$y=cos{x}$')
# 给默认子区域创建一个标题
plt.title('sin曲线图', fontsize=20)
# 给默认子区域创建x轴和y轴标签
plt.xlabel('x轴', fontsize=16)
plt.ylabel('y轴', fontsize=16)
# 设置默认子区域的x和y轴刻度取值范围
plt.xlim(0, 2 * np.pi)
plt.ylim(-1, 1)
# 设置默认子区域的x轴和y轴的刻度的标签
plt.xticks([0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi, 2 * np.pi], [r'0', r'0.5π', r'π', r'1.5π', r'2π'])
plt.yticks([-1.0, 0.0, 1.0], [r'min -1.0', r'0.0', r'max 1.0'])
# 给默认子区域的某个位置上增加文字描述，注意前两个参数是描述在x轴和y轴坐标系中的位置，数值是x和y轴的刻度
plt.text(1.2 * np.pi, 0.8, r'$x \in [0.0, \ pi]$', color='red', fontsize=10)
plt.text(1.2 * np.pi, 0.7, r'$y \in [-1.0, \ 1.0]$', color='red', fontsize=10)
# 给默认子区域上某个点创建一个注解
# xy参数设置'被注解点'的坐标,xytext参数设置'注解文字'的位置,可以像xy一样设置绝对位置，也可以像xytext=(+30,-30)设置与xy点相对位置
# xytext=(+30,-30)指在被注解点向右移动30，向下移动30,textcoords='offset points'指以被注解点为起点
# arrowprops参数设置注解文字与被注解点的连接方式,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.1')指弧度曲线，0.1指弧度
# 标签中想要有空格，就再空格前一位加一个\
plt.annotate(r'$y\_sin\ max\ point$', xy=(0.5 * np.pi, np.sin(0.5 * np.pi)), xytext=(0.75 * np.pi, 0.7), fontsize=14,
             color='#090909', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='#090909'))
# 设置默认子区域的图例，loc指图例在子区域中的位置，best是自动配置最佳位置，upper right是在右上角
# 注意如果想创建图例，最好在上面创建曲线时给每条曲线设置曲线的label名
plt.legend(loc="upper right")
# 是否要打开网格线，True打开网格线
plt.grid(True)
# 显示画出来的图像，注意保存不show时也可以保存
plt.show()
# 保存图像,注意图片保存的上一级目录必须存在才能保存
plt.savefig("test.jpg")
# 关闭图像
plt.close()
```
当我们想创建多个figure对象，或每个figure对象中有多个子区域时，不能用上面的快捷方法画图。而是要先创建figure对象（fig = plt.figure()），然后将figure对象划分成多个子区域（ax1 = fig.add_subplot(221)），然后对每个子区域分别绘制这个子区域内的图像。
# 创建figure对象与划分子区域
在任何绘图之前，我们需要一个Figure对象，可以理解成我们需要一张画布才能开始绘图。使用fig = plt.figure()即将一个figure对象命名为fig。
在创建Figure对象之后，我们还要确定这块画布上要绘制几个子图。我们可以使用add_subplot()将画图分割成若干块，并为每一块命名。这样我们就可以对每一个子块区域单独设置其图像的各种属性。另一种方法是plt.subplots(nrows=2, ncols=2)一次性划分出2行X2列一共4个子区域，这个时候fig也一起创建了。
举例：
```python
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
plt.show()

fig2, axes = plt.subplots(2, 2)
axes[0, 0].set(title='Upper Left')
axes[0, 1].set(title='Upper Right')
axes[1, 0].set(title='Lower Left')
axes[1, 1].set(title='Lower Right')
plt.show()
```
fig.subplots_adjust()方法可用于调整子图布局，如：
```python
# 子图之间水平间隔wspace=0.5，垂直间距hspace=0.3，左边距left=0.125 ，右边距right=0.9，上边距top=0.9，下边距bottom=0.1，这里数值都是百分比
fig.subplots_adjust(wspace=0.5, hspace=0.3, left=0.125, right=0.9, top=0.9, bottom=0.1)
```
我们还可以使用fig.tight_layout()进行自动调整，使标题之间不重叠。
# 子区域标题/轴标签/刻度范围/刻度值/文本描述/注解/图例/网格线/显示、保存、关闭图片
对ax1子区域增加一个区域的标题。可同时设置标题字体和字体大小。
```python
ax1.title(‘标题’, fontproperties = 'FangSong', fontsize = 20)
```
对ax1子区域的x和y轴设置轴标签，可同时设置标签字体和字体大小。
```python
ax1.xlabel(‘横轴：时间’, fontproperties = 'FangSong', fontsize = 20)
ax1.ylabel(‘纵轴：数量’, fontproperties = 'FangSong', fontsize = 20)
```
对ax1子区域的x轴和y轴设置刻度的上下限。第一个参数是下限，第二个参数是上限。
```python
ax1.xlim(0, 10)
ax1.ylim(-1, 1)
```
设置ax1的x轴和y轴刻度值标签。第一个参数是要设置标签的刻度值，第二个参数是刻度值对应的标签。注意不是所有的刻度值都一定要设置标签。
```python
ax1.xticks([0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi, 2 * np.pi], [r'0', r'0.5π', r'π', r'1.5π', r'2π'])
ax1.yticks([-1.0, 0.0, 1.0], [r'min -1.0', r'0.0', r'max 1.0'])
```
axes.xaxis/yaxis.set_ticks_position()设置x轴或y轴的刻度标签的位置。
对xaxis参数可以是'top'，'bottom'，'both'，'default'，'none'；对yaxis参数可以是'left'，'right'，'both'，'default'，'none'。
```python
axes.xaxis.set_ticks_position('bottom')  # 设置axes子区域的x轴刻度值标签在x轴下方 
axes.yaxis.set_ticks_position('left')  # 设置axes子区域的y轴刻度值标签在y轴左边
```
在ax1子区域增加一个文本描述。前两个参数指定文本在这个子区域中的位置，第三个参数是文本的内容。
```python
ax1.text(1.2 * np.pi, 0.8, r'$x \in [0.0, \ pi]$', color='red', fontsize=10)
```
在ax1子区域增加一个注解。第一个参数是注解内容。xy是被注解点位置，xytext是注解内容的位置， arrowprops参数设置注解文字与被注解点的连接方式。
```python
ax1.annotate(r'$y\_sin\ max\ point$', xy=(0.5 * np.pi, np.sin(0.5 * np.pi)), xytext=(0.75 * np.pi, 0.7), fontsize=14,
             color='#090909', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='#090909'))
```
在ax1子区域显示图例。设置图例之前最好对每条曲线命名一个标签。loc参数指标签在子区域的哪个位置。
**loc可选参数有：**
'best':0
'upper right':1
'upper left':2
'lower left':3
'lower right':4
'right':5
'center left':6
'center right':7
'lower center':8
'upper center':9
'center':10
```python
ax1.legend(loc="upper right")
ax1.legend(loc=1) # 两种写法等同
```
是否要在ax1子区域显示网格线，为True则显示网格线，为False不显示网格线，默认不显示。
```python
ax1.grid(True)
```
显示、保存、关闭图片。显示图片为plt.show()，保存图片为plt.savefig()，关闭图片为plt.close()。这三个函数的前缀必须是plt，因为这三个操作是对整个figure对象进行的。plt.savefig()保存时上级目录必须存在才可以保存，否则会报错。
```python
plt.show()
plt.savefig("./test_image/test.jpg")
plt.close()
```
# 绘制曲线图(.plot()函数)
ax1.plot()函数的前缀ax1，表示在ax1这个子区域内添加图形。注意如果是线条图，一个子区域内可以同时画多个线条。.plot()函数要需要x轴数据，y轴数据，y轴数据可以通过x轴数据来生成。
```python
import matplotlib.pyplot as plt
import numpy as np

# figsize=(8, 6)即figure窗口大小800X600像素
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# 创建x数据，np.linspace()默认创建50个数据的等差数列
x = np.linspace(0, 2 * np.pi)
# 创建y数据
y_sin = np.sin(x)
y_cos = np.cos(x)
# 在ax1上同时添加两条不同的曲线
ax[0, 0].plot(x, y_cos, '.', x, y_sin, '-', linewidth=2, markersize=8)
# 在ax2上添加一条曲线
ax[0, 1].plot(x, y_cos, '-', color='red', marker='+', markersize=12)
plt.show()
plt.close()
```
# 绘制散点图(.scatter()函数)
.scatter()函数的第一、二个参数是x值和y值，s为点的大小，alpha为点的透明度。
```python
import matplotlib.pyplot as plt
import numpy as np

# figsize=(8, 6)即figure窗口大小800X600像素
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# numpy.random.normal(loc=0.0, scale=1.0, size=None)
# loc:概率分布的均值，对应着整个分布的中心center
# scale:概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
# size:输出的shape，默认为None，只输出一个值
x = np.random.normal(0, 1, 200)
y = np.random.normal(0, 1, 200)
# s指点大小，alpha指透明度
ax[0, 0].scatter(x, y, s=1, color='green', alpha=0.5)
ax[0, 1].scatter(x, y, s=10, color='red', alpha=0.5)
ax[1, 0].scatter(x, y, s=50, color='blue', marker='+', alpha=0.5)
ax[1, 1].scatter(x, y, s=80, color='blue', alpha=0.5)
plt.show()
plt.close()
```
# 条形图与横向条形图(.bar()和.barh()函数)
条形图分两种，一种是纵向条形图(一般都是这种)，还有一种是横向条形图。
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# 创建x数据
np.random.seed(1)
x = np.arange(0, 10, 1)
y = np.random.randint(-5, 5, 10)
# 画一个条形图
ax[0, 0].bar(x, y, facecolor='blue', edgecolor='white', alpha=0.5)
# 画一个横向条形图
ax[0, 1].barh(x, y, color='red', edgecolor='white', alpha=0.5)
# 在条形图的0值分界线画一条线
ax[0, 1].axvline(0, color='black', linewidth=1)
plt.show()
plt.close()
```
# 直方图(.hist()函数)
直方图将数据分成指定数量的分类，y轴可选择显示数量或某类占总数的百分比。
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
np.random.seed(1)
y = np.random.normal(0, 1, 1000)
# 画三个不同的直方图
# 参数：y数据，density=True则y轴是数量占总数量百分比，False则y轴是数量，histtype是直方图类型，alpha为透明度
# bar表示按条形图风格，barstacked表示多分类累计的条形图风格，stepfilled表示去除条形的黑色边框
ax[0, 0].hist(y, 10, density=True, histtype='bar', facecolor='blue', edgecolor="black", alpha=0.5)
ax[0, 1].hist(y, 10, density=False, histtype='barstacked', facecolor='green', edgecolor="black", alpha=0.5)
ax[1, 0].hist(y, 10, density=False, histtype='stepfilled', facecolor='red', edgecolor="black", alpha=0.5)
plt.show()
plt.close()
```
# 饼状图(.pie()函数)
饼图根据数据的百分比画饼。labels是各个块的标签。autopct=%1.1f%%表示各部分所占的百分比的格式化输出，explode表示各部分块与圆心的距离，值越大离圆心越远。pctdistance=1.12表示百分比数字距离圆心的距离，默认是0.6。
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# 创建x数据
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode_1 = (0, 0.1, 0, 0)
explode_2 = (0.5, 0.0, 0, 0)
# .pie()参数：各部分百分比，各部分突出程度(值越大越突出)，各部分标签，显示数字的格式，是否要阴影，起始角度
# pctdistance表示百分比数字距离圆心的距离
# 饼图默认是圆
ax[0, 0].pie(sizes, explode=explode_1, labels=labels, autopct='%1.1f%%', shadow=True, startangle=0)
# bbox_to_anchor=(1.0, 1.0)中两个1.0指对上边和右边的外边距离，borderaxespad = 0.3指图例的内边距
ax[0, 0].legend(labels=labels, loc="upper right", bbox_to_anchor=(1.3, 1.3), borderaxespad=0.3)
ax[0, 1].pie(sizes, explode=explode_2, labels=labels, autopct='%1.2f%%', shadow=True, startangle=45, pctdistance=0.4)
ax[1, 0].pie(sizes, explode=explode_2, labels=labels, autopct='%1.0f%%', shadow=False, startangle=90, pctdistance=0.8)
plt.show()
plt.close()
```
# 箱式图(.boxplot()函数)
箱线图通过数据的四分位数来展示数据的分布情况。箱线图把数据从小到大进行排列并等分成四份，第一分位数（Q1），第二分位数（Q2）和第三分位数（Q3）分别为数据的第25%，50%和75%的数字。
箱线图分为两部分，分别是箱（box）和须（whisker）。箱（box）用来表示从第一分位到第三分位的数据，须（whisker）用来表示数据的范围。
箱线图从上到下各横线分别表示：数据上限（通常是Q3+1.5XIQR），第三分位数（Q3），第二分位数（中位数），第一分位数（Q1），数据下限（通常是Q1-1.5XIQR）。有时还有一些圆点，位于数据上下限之外，表示异常值（outliers）。
```python
import matplotlib.pyplot as plt
import numpy as np

# figsize=(8, 6)即figure窗口大小800X600像素
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
# 创建x数据
y = np.random.normal(0, 1, 1000)
# 画箱式图
ax[0, 0].boxplot(y)
ax[0, 1].boxplot(y, vert=False)
plt.show()
plt.close()
```
# 极坐标图
```python
import matplotlib.pyplot as plt
import numpy as np

# figsize=(8, 6)即figure窗口大小800X600像素
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(121, projection='polar')
ax2 = fig.add_subplot(122, projection='polar')
N = 30
# 将极坐标均分为N份
thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
# 取一个随机的角度
turn_rads = 10 * np.random.rand(N)
# 取一个随机的宽度
widths = np.pi / 4 * np.random.rand(N)
# 从哪个角度开始画，画过多少角度长度，画的扇形的半径，从距离圆心0的地方开始画
ax1.bar(thetas, turn_rads, width=widths, facecolor='green', bottom=0.0, alpha=0.5)
ax2.bar(thetas, turn_rads, width=widths, facecolor='red', bottom=2, alpha=0.5)
plt.show()
plt.close()
```
# 等高线图(.contour()和.contourf()函数)
等高线地图就是将地表高度相同的点连成一环线直接投影到平面形成水平曲线。在机器学习中也会被用在绘制梯度下降算法的图形中。
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 2, figsize=(10, 6))
# 创建x数据和y数据
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
# meshgrid函数使用x和y的数组的值作为x轴和y轴坐标点在平面上确定一个点，这个点的值是Z，Z大小关系到颜色的深浅。
# 具体来说，就是将x这个100个数值的一维向量按行复制100行，变成X其shape=(100,100)的二维向量，y值也是这么处理的
# 注意x和y的shape都是(100,)，而X和Y的shape为(100,100)和(100,100)
X, Y = np.meshgrid(x, y)
# 注意计算Z要用X和Y计算，这样计算出来的Z的shape=(100,100)，画图时数据的shape才能对的上，否则会报错
Z = (1 - X ** 2 + X ** 5 + Y ** 3) * np.exp(-X ** 2 - Y ** 2)
# Z通常表示的是距离X-Y平面的距离，传入X、Y则是控制了绘制等高线的范围。
# contourf会填充轮廓线之间的颜色,而contour不会填充轮廓线之间的颜色
# 10指等高线图划分成10个不同范围的区域，alpha指透明度,cmap=plt.get_cmap('hot')指填充用的颜色域
ax[0, 0].contourf(X, Y, Z, 10, alpha=0.75, cmap=plt.get_cmap('hot'))
# 在ax[0, 0]的contourf图中绘制等高线
C = ax[0, 0].contour(X, Y, Z, 10, colors='black')
# 显示各等高线的数据标签
ax[0, 0].clabel(C, inline=True, fontsize=10)
# ax[0, 1]也是等高线图，但这个图没有填充轮廓线之间的颜色
D = ax[0, 1].contour(X, Y, Z, 10, alpha=0.75, colors='red')
# 显示各等高线的数据标签
ax[0, 1].clabel(D, inline=True, fontsize=10)
ax[1, 0].contour(X, Y, Z, 30, alpha=0.75, colors='blue')
plt.show()
plt.close()
```
# 3D图
3D图在机器学习和深度学习中观察local minima点和global minima点时十分方便。3D图可以旋转角度，以找到最佳的观察角度。
```python
from mpl_toolkits.mplot3d import Axes3D  # 画3D图所需要的包
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
# 创建x数据和y数据
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
# meshgrid函数使用x和y的数组的值作为x轴和y轴坐标点在平面上画网格。
# 具体来说，就是将x这个100个数值的一维向量按行复制100行，变成X其shape=(100,100)的二维向量，y值也是这么处理的
# 注意x和y的shape都是(100,)，而X和Y的shape为(100,100)和(100,100)
X, Y = np.meshgrid(x, y)
# 注意计算Z要用X和Y计算，这样计算出来的Z的shape=(100,100)，画图时数据的shape才能对的上，否则会报错
Z = (1 - X ** 2 + X ** 5 + Y ** 3) * np.exp(-X ** 2 - Y ** 2)
# Z通常表示的是距离X-Y平面的距离，传入X、Y则是控制了绘制等高线的范围。
# rstride=1, cstride=1指x方向和y方向的色块大小,可以不指定
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('hot'))
# 画出3D图对应的等高线图，该等高线图的z轴值全部压缩到-1，也就是说这个图在z=-1的位置，和X-Y平面平行
ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap='hot')
plt.show()
plt.close()
```
# 动态图
动态图有助于我们实时地观察某个变量值的变化情况。
```python
from matplotlib import pyplot as plt
from matplotlib import animation  # 让图动态更新必须的包
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 5 * np.pi, 100)
line, = ax.plot(x, np.sin(x), color="red", linewidth=2)

# 更新图形函数
def animate(i):
   line.set_ydata(np.sin(x + i / 10))
   return line,


# 图形初始化函数
def initial():
   line.set_ydata(np.cos(x))
   return line,


# func是更新图形的函数，frames是总共更新的次数，intit_func是图形初始化函数，interval是更新的间隔时间(ms)
# blit决定是更新整张图的点(Flase)还是只更新变化的点（True）
ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=initial, interval=50, blit=False)
plt.show()
plt.close()
```