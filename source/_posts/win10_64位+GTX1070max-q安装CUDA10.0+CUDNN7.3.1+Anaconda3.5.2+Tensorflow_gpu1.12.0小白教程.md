---
title: win10_64位+GTX1070max-q安装CUDA10.0+CUDNN7.3.1+Anaconda3.5.2+Tensorflow_gpu1.12.0小白教程
date: 2019-02-21 16:06:47
tags:
- 各种环境搭建
categories:
- 各种环境搭建
---

# 准备工作：重装系统
机器：神舟精盾t97e(八代i7)，显卡为GTX1070max-q。
为了排除不必要的干扰，我们先重装一个官方纯净版的win10 x64系统。使用微软官方工具来刻录一个系统安装的U盘，工具下载地址在这里：https://www.microsoft.com/zh-cn/software-download/windows10 ，选择立即下载工具，然后按照提示操作即可。注意我们要刻录win10 x64系统的话U盘至少需要8G大小，且刻录时会格式化U盘。
刻录完成后，先备份C盘中的所有资料，然后重启电脑，按F2键(各个型号的笔记本按键可能不同，请自行查阅)进入bios，将U盘设为第一启动盘，然后重启电脑，按照提示安装系统。
由于我的笔记本厂商已经预装了win10 x64家庭中文版系统(注意这个预装的系统一定要先激活)，因此只要我重装的系统也是win10 x64家庭中文版系统，那么系统重装好后就是激活的。如果没有激活，可以使用AIDA64来查看你的oem key：选择主板->ACPI->MSDM->SLS data，其信息就是oem key，你只要用这个key激活自己的win10系统即可。
# 安装tensorflow官方版本的操作步骤
如果你想安装官方版本的tensorflow，请按下面步骤来操作。注意官方版本的tensorflow最高只支持CUDA9和CUDNN9。
## 查看你的GPU计算等级(Compute Capability)
首先你需要知道你的显卡具体型号。然后打开网址：https://developer.nvidia.com/cuda-gpus ，查看显卡对应的计算等级(Compute Capability)。GTX1070max-q的核心与GTX1070的核心相同，计算等级都为6.1。
如果在这个列表中没有查找到你对应的显卡型号，那么直接去查看你的显卡当前安装的NVIDIA驱动所支持的CUDA最高版本。在桌面右键选择NVIDIA控制面板，然后选择系统信息，找到NVCUDA.dll对应的那一行，右边的数字即为支持的CUDA最高版本号。
最后要装哪个版本的CUDA，还要结合tensorflow对CUDA版本的支持情况来看。
## 查看CUDA各版本支持的GPU计算等级
从CUDA-wiki百科中可以查到各个版本的CUDA支持的GPU计算等级范围。网址：https://en.wikipedia.org/wiki/CUDA#GPUs_supported 。
同时，各个版本的CUDA对于NVIDIA驱动也有版本上的要求，CUDA各版本和NVIDIA显卡驱动版本的对应关系在这里可以查到：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 。
我们可以发现CUDA各个版本都支持一定范围内的计算等级，所以我们可以选择的CUDA版本有多个。但因为我们的CUDA还要配合TensorFlow使用，因此选择哪个版本的CUDA还要看TensorFlow对CUDA版本的支持情况。
## 查看TensorFlow各版本支持的CUDA版本
我们的平台是win10 64位系统，而且我们要安装tensorflow-gpu版本，因此我们只看windows平台的Tensorflow各版本支持的CUDA版本，网址在这里：https://www.tensorflow.org/install/source#common_installation_problems 。我们可以发现tensorflow-gpu官方最新版本为1.12.0，最高支持CUDA9和CUDNN7。
## 确定最终安装的tensorflow-gpu、CUDA、CUDNN版本
我们尽量安装更新版本的tensorflow-gpu，因此我们选择安装tensorflow_gpu-1.12.0。这个版本支持CUDA9和CUDNN7。
因此，对于我们的GTX1070max-q显卡，我们可以选择安装CUDA9.0+CUDNN7.0.5+Anaconda3.5.2+tensorflow-gpu1.12.0。
CUDA9.0及其4个更新包下载地址：https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal 。
CUDNN7.0.5下载地址：https://developer.nvidia.com/rdp/cudnn-archive 。
Anaconda3.5.2可在官网上下载，tensorflow-gpu使用pip安装即可，即：
```python
python -m pip install tensorflow-gpu==1.12.0
```
我已经尝试过上面这种组合，在重装系统后仅安装了自带的NVIDIA389.21驱动后安装上述组合，tensorflow-gpu可以正常调用显卡进行计算。注意安装CUDA时提示显卡新于CUDA工具包，此时选择自定义安装，只勾选CUDA项安装即可。
# 10系显卡安装CUDA10+CUDNN7.3.1+Anaconda3.5.2+tensorflow-gpu1.12.0魔改版
由于tensorflow-gpu最高只支持到CUDA9，而CUDA已经发布了最新的10.0版本，国外有一位大神修改了tensorflow-gpu包使其支持CUDA10.0+CUDNN7.3.1。大神的Github仓库地址为：https://github.com/fo40225/tensorflow-windows-wheel 。
要想下载这个魔改的tensorflow-gpu1.12.0的话，地址为：https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.12.0/py36/GPU/cuda100cudnn73avx2 。
## 安装特定版本NVIDIA显卡驱动
注意CUDA10.0版本需要NVIDIA 411.31及之后版本的驱动才能正常工作，我们需要先安装符合版本要求的驱动。NVIDIA驱动下载地址：https://www.nvidia.cn/Download/Find.aspx?lang=cn 。
这里我们选择下载411.63驱动，安装。安装时选择自定义安装，全部勾选。
如果你的系统不是重装后的新系统(新系统什么驱动都没装)，安装这个驱动时可能会失败，提示显卡驱动与windows版本不兼容。需要说明的一点是，重装系统后最好立即安装这个驱动，因为win10系统默认是打开更新的，会自动搜索显卡驱动来安装，而自动搜索的显卡驱动不一定是你需要的版本，如果自动搜索到的显卡驱动被安装好了，那么再安装这个411.63驱动一样会出现提示显卡驱动与windows版本不兼容。
## 下载和安装CUDA
CUDA10.0下载地址：https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal 。我们可以选择直接下载完整安装包。
在CUDA安装过程中请选择自定义选项。全部勾选，安装。安装位置默认即可。
在安装某些版本的CUDA时，有时会提示显卡驱动不兼容该显卡，可以继续安装，但是可能无法使用CUDA。 这种情况是由于显卡新于该工具包造成的。在这种情况下，建议保留现有驱动并安装CUDA工具包的剩余部分(也是选择自定义安装)。 
安装完成后，添加下面四个路径到用户环境变量中：
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
**这四个路径非常重要，如果不添加，会出现import tensorflow时正常不会报错，但当你要执行tf.Session()的时候就会卡住。因为这个时候将会调用cuda，没有路径的话cuda需要的各种lib、dll加载不了。**
上述操作全部完成后打开cmd，输入nvcc -V，如果出现CUDA版本信息，证明路径已经配置好了。
## CUDNN的下载和安装
如果我们安装上面大神修改的魔改版tensorflow-gpu-1.12.0，那么我们必须选择CUDA10+CUDNN7.3.1的组合。CUDNN7.3.1下载地址：https://developer.nvidia.com/rdp/cudnn-archive 。
下载文件。解压缩后得到一个文件夹。将这个目录中所有文件复制粘贴到CUDA安装位置：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0即可。
## Anaconda的下载和安装
我们可以从anaconda官网上下载最新版本的anaconda，如果觉得速度慢的话从这个镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 中下载也可以，注意安装anaconda3-5.2.0，这个版本的anaconda默认安装python3.6。
安装过程按默认设置下一步即可。注意不要勾选add anaconda to the system PATH environment variable，我们待会儿会手动设置。
安装完成后，在用户环境变量中新建一个path，添加下面三个路径：
C:\Users\zgcr6\Anaconda3
C:\Users\zgcr6\Anaconda3\Scripts
C:\Users\zgcr6\Anaconda3\Library\bin
然后打开cmd，输入conda -V测试一下，如果能正常显示版本号，说明已经配置好了。
## Tensorflow-gpu的安装过程
我们在anaconda中先新建一个tensorflow-gpu的环境，该环境安装python3.6。打开anaconda prompt，使用命令：
```python
conda create -n tensorflow-gpu-1.12.0 python=3.6.6
```
然后等待安装完成。该命令创建了一个名为tensorflow-gpu的环境，并安装了python3.6.6。创建的环境位置在：C:\Users\zgcr6\Anaconda3\envs。
然后我们将下面这两个目录添加到用户变量中，注意要放到anaconda的目录之前：
C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0
C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0\Scripts
然后打开cmd.exe，如果我们之前安装的是CUDA9.0或更早版本的CUDA，我们可以安装tensorflow-gpu1.12.0官方版本，使用下面的命令即可：
```python
python -m pip install tensorflow-gpu==1.12.0
```
如果我们之前安装的是CUDA10.0，我们需要安装大神修改的tensorflow-gpu1.12.0。首先我们下载tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl文件，将该文件放在C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0\目录下。然后打开cmd，使用命令：
```python
cd C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0
python -m pip install tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl
```
等待安装完成即可。安装完成后，我们还需要验证tensorflow能否正常调用显卡进行计算。打开cmd.exe，输入python，然后依次输入下面的命令：
```python
import tensorflow as tf
hello = tf.constant("hello, world!")
sess = tf.Session()
sess.run(hello)
```
当输入到第三句sess = tf.Session()时，即开始调用显卡。注意如果是第一次在系统中使用tensorflow来调用显卡时，在adding visible gpu devices:0这里会卡住3-5分钟，之后如果能够继续运行并正常输出hello world，则说明安装成功。
# 如何使用我们安装的tensorflow-gpu1.12.0环境
## 在cmd中使用tensorflow-gpu环境
将tensorflow-gpu-1.12.0环境的路径加入到用户环境变量中，即路径C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0，注意这个路径一定要放在我们刚才添加的路径C:\Users\zgcr6\Anaconda3之前。
此时打开cmd后，输入python运行的就是C:\ProgramData\Anaconda3\envs\tensorflow-gpu文件中的python.exe。
## 在编译器(如pycharm)中使用tensorflow-gpu1.12.0环境
我们以pycharm为例，在pycharm中找到项目的解释器设置(file->settings->project interpreter)，将解释器路径设为我们的C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.12.0\python.exe即可。