---
title: Python pip安装模块各种踩坑教程
date: 2019-02-26 20:06:21
tags:
- 各种环境搭建
categories:
- 各种环境搭建
---

# python中使用pip安装模块的方法
pip安装模块命令：
```python
python -m pip install 模块名
python -m pip install 模块名==模块版本号 # 指定版本号可安装特定版本的模块
```
# 如何知道我们的python模块安装到哪个位置
当我们在cmd.exe中运行Python.exe时，运行的Python是哪一个Python取决于我们的用户环境变量path中哪一个含有python.exe的路径在环境变量path的最前面，在cmd中输入python执行时总是执行path路径中第一个含有python.exe路径中的python.exe。
如果我们使用anaconda prompt，只要我们使用命令：
```python
activate 环境名
python -m pip install 模块名
```
这时模块就会安装在上面activate的那个环境中。
# Cannot uninstall X错误的解决方法
有时我们会遇到Cannot uninstall X这类错误，提示类似下面的代码：
```python
Installing collected packages: numpy
  Found existing installation: numpy 1.8.2
Cannot uninstall 'numpy'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.
```
出现这种错误的原因是我们在新安装B包时，以前安装的A包中含有一些B包中要安装的模块，而Python并不知道这些模块中是A包中的版本比较新还是B包中的版本比较新，所以选择中止安装。
我们可以使用命令：
```python
pip install 模块名 --ignore-installed
```
来解决这个冲突，这样遇到已安装过的模块时就会跳过这个模块。
# pip安装jupyter notebook以及在指定目录下打开jupyter notebook
**pip安装jupyter notebook(请打开代理，有时可能下载不成功)：**
```python
python -m pip install jupyter
```
**在指定目录下打开jupyter notebook：**
首先将你的jupyter-notebook.exe的路径加入到用户变量path中，jupyter-notebook.exe在你的python安装文件夹下的Scripts文件夹下，比如我的路径为：C:\ProgramData\Anaconda3\envs\tensorflow-gpu\Scripts。
然后在指定目录下按住shift，鼠标右键选择在此处打开命令窗口（或者直接打开cmd.exe，用cd命令进入你想进的目录也可），然后输入jupyter-notebook即可。这时我们发现启动jupyter-notebook后的目录变成了当前的指定目录。注意jupyter-notebook运行时不可以关闭上面的cmd窗口。
# windows下pip安装 COCO API(pycocotools)
COCO是一个大型图像数据集，设计用于对象检测，分割，人物关键点检测，填充物分割和字幕生成。该软件包提供了Matlab，Python和Lua API，可帮助加载，解析和可视化COCO中的注释。
原本COCO对Windows是不支持的。不过为了支持 Windows ，有人对 COCO 做了一些修改。
COCO原版本地址： https://github.com/cocodataset/cocoapi  
支持Windows的COCO地址：https://github.com/philferriere/cocoapi 
**windows上pip安装COCO方法：**
在cmd中运行下列命令：
```python
python -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
注意在windows上安装COCO前系统中必须安装bulid tools for VS2015，否则会安装失败。VS2015下载地址在这里：https://www.microsoft.com/en-us/download/details.aspx?id=48159
# pip安装imgaug模块失败
imgaug库可以略微改变图像来扩大我们的数据集。通过对图像进行旋转、模糊、增加噪声等操作来得到一组新的略微改变的图像。
使用下列命令安装imgaug：
```python
python -m pip install imgaug
```
安装失败，出现下列错误提示：OSError：[WinError 126] 找不到指定的模块。command"python setup.py egg_info" failed with error code 1 in C:Users\1234\Appinstall-32335jz3\Shapely\ 。
出现这个错误是因为安装imgaug包时需要安装另一依赖包shapely。而安装shapely包时系统中需要geos.dll，geos_c.dll这两个文件。
geos.dll，geos_c.dll这两个文件是geos库编译后产生的。GEOS的前身是JTS，JTS提供了全功能的，强大的空间操作和空间判断。 后来PostGIS缺少一套完整的空间查询操作，于是就将JTS移植成为C++版本，正式命名为GEOS。GEOS为开源库，它包括了完整的空间查询和一大部分空间操作，是从事图形操作和GIS行业开发人员经常接触的开发库。
我们可以在官网： http://trac.osgeo.org/geos 下载geos最新版本，然后使用vcvars64.bat进行编译。如果系统是32位，那么请使用vcvars32.bat进行编译。
**具体安装过程：**
首先你的系统中要安装Visual Studio2015（注意别安装2017！！geos最新版本还不支持用VS2017来编译），然后找到vcvars64.bat在Visual Studio2015安装目录中的具体目录：C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64。在该目录下按住shift，鼠标右键选择在此处打开命令窗口，然后输入vcvars64.bat初始化编译环境。
然后使用命令：
```
cd C:\Users\1234\Desktop\geos-3.7.1 # cd到你下载的geos文件的目录
autogen.bat
nmake /f makefile.vc
```
等待几分钟至编译完成。编译完成后，会在C:\Users\1234\Desktop\geos-3.7.1\src目录下生成geos.lib, geos_i.lib, geos_c_i.lib, geos.dll, geos_c.dll等五个文件。
将geos.dll和geos_c.dll两个文件粘贴到windows/system32文件夹下，返回命令行窗口，重新运行：
```python
python -m pip install imgaug
```
这次安装成功了。
# Python安装pyecharts包及保存图片为png格式问题
pip安装pyecharts包、pyecharts-snapshot包和地图包:
```python
python -m pip install pyecharts
python -m pip install pyecharts-snapshot
python -m pip install echarts-countries-pypkg
python -m pip install echarts-china-provinces-pypkg
python -m pip install echarts-china-cities-pypkg
python -m pip install echarts-china-counties-pypkg
python -m pip install echarts-china-misc-pypkg
python -m pip install echarts-united-kingdom-pypkg
```
安装完成后应当可以用pyecharts画图了，但此时图片只能用html网页形式保存(网页形式可以进行一些图片交互操作)。
如果我们想以png格式保存图片，我们还需要安装phantomjs。安装命令:
```python
npm install -g phantomjs-prebuilt
```
安装过程中需要下载一个包phantomjs-2.1.1-windows.zip，我们可以从cmd窗口中直接复制下载地址将包下载到本地，放入目录:C:\Users\zgcr6\AppData\Local\Temp\phantomjs\phantomjs-2.1.1-windows.zip，然后关闭cmd窗口，重新打开cmd，运行上面的命令即可。
安装完成后我们还要从官网下载node.js并安装，最后重启计算机即可。测试用图片代码:
```python
data = [
("海门", 9),("鄂尔多斯", 12),("招远", 12),("舟山", 12),("齐齐哈尔", 14),("盐城", 15),
("赤峰", 16),("青岛", 18),("乳山", 18),("金昌", 19),("泉州", 21),("莱西", 21),
("日照", 21),("胶南", 22),("南通", 23),("拉萨", 24),("云浮", 24),("梅州", 25)]

attr, value = geo.cast(data)

geo = Geo("全国主要城市空气质量热力图", "data from pm2.5", title_color="#fff", title_pos="center", width=1200, height=600, background_color='#404a59')

geo.add("空气质量热力图", attr, value, visual_range=[0, 25], type='heatmap',visual_text_color="#fff", symbol_size=15, is_visualmap=True, is_roam=False)
geo.show_config()
geo.render(path="4-04空气质量热力图.png")
```
# Python安装plotly包
安装plotly命令:
```python
python -m pip install plotly
```
注意plotly分在线生成图片和离线生成图片两种形式。
在线生成图片需要在plotly官网上申请一个账户，并在自己的账户中生成一个api_key，然后使用必须加上下面两行语句。注意免费账户在线生成图片时在你的账户里会保存这个图片，免费账户每天限制在线生成100张图片，且sharing不能设为private,world_readable必须设为True。
```
plotly.tools.set_credentials_file(username='', api_key='')
# 免费账户在线生成图片sharing不能设为private,world_readable必须设为True
plotly.tools.set_config_file(world_readable=True)
```
如果用离线生成图片，请加上下面这句:
```python
plotly.offline.init_notebook_mode(connected=True)
```