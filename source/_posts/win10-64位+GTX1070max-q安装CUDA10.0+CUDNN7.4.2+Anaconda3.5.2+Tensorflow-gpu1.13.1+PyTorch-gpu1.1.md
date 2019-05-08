---
title: win10-64位+GTX1070max-q安装CUDA10.0+CUDNN7.4.2+Anaconda3.5.2+Tensorflow-gpu1.13.1+PyTorch-gpu1.1
date: 2019-02-21 16:06:47
tags:
- 各种环境搭建
categories:
- 各种环境搭建
---

# 准备工作：重装系统
机器：神舟精盾t97e(八代i7)，显卡为GTX1070max-q。tensorflow官方目前已有1.13.1版本，原生支持CUDA10和CUDNN7.4.2。pytorch1.1版本已支持tensorboard，故最好安装这个版本。我们今天就来安装CUDA10.0+CUDNN7.4.2+anaconda3.5.2+tensorflow-gpu1.13.1+pytorch-gpu1.1。
**注意:**
**如果你的win10系统直接能够安装满足CUDA10要求的版本的nvidia驱动，可以不重装系统。**

由于我的win10系统直接安装满足CUDA10要求的版本的nvidia驱动总是失败，因此我们先重装一个官方纯净版的win10 x64系统。使用微软官方工具来刻录一个系统安装的U盘，工具下载地址在这里：https://www.microsoft.com/zh-cn/software-download/windows10 ，选择立即下载工具，然后按照提示操作即可。注意我们要刻录win10 x64系统的话U盘至少需要8G大小，且刻录时会格式化U盘。
刻录完成后，先备份C盘中的所有资料，然后重启电脑，按F2键(各个型号的笔记本按键可能不同，请自行查阅)进入bios，将U盘设为第一启动盘，然后重启电脑，按照提示安装系统。
由于我的笔记本厂商已经预装了win10 x64家庭中文版系统(注意这个预装的系统一定要先激活)，因此只要我重装的系统也是win10 x64家庭中文版系统，那么系统重装好后就是激活的。如果没有激活，可以使用AIDA64来查看你的oem key：选择主板->ACPI->MSDM->SLS data，其信息就是oem key，你只要用这个key激活自己的win10系统即可。
# 安装特定版本NVIDIA显卡驱动
从CUDA-wiki百科中可以查到各个版本的CUDA支持的GPU计算等级范围。网址：https://en.wikipedia.org/wiki/CUDA#GPUs_supported 。然后打开网址：https://developer.nvidia.com/cuda-gpus ，查看显卡对应的计算等级(Compute Capability)。
GTX1070max-q的核心与GTX1070的核心相同，计算等级都为6.1。由此可知我们的GTX-1070MAX-q显卡可以安装CUDA10。关于CUDA各版本对nvidia驱动版本的要求可以看这里:https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 。我们可查到CUDA10.0版本需要NVIDIA 411.31及之后版本的驱动才能正常工作，我们需要先安装符合版本要求的驱动。NVIDIA驱动下载地址：https://www.nvidia.cn/Download/Find.aspx?lang=cn 。
这里我们选择下载411.63驱动，安装。安装时选择自定义安装，全部勾选。如果你的系统不是重装后的新系统(新系统什么驱动都没装)，安装这个驱动时可能会失败，提示显卡驱动与windows版本不兼容。需要说明的一点是，重装系统后最好立即安装这个驱动，因为win10系统默认是打开更新的，会自动搜索显卡驱动来安装，而自动搜索的显卡驱动不一定是你需要的版本，如果自动搜索到的显卡驱动被安装好了，那么再安装这个411.63驱动一样会出现提示显卡驱动与windows版本不兼容。
# 下载和安装CUDA10.0
我们在这个网址可以查到各个版本的tensorflow支持的CUDA和CUDNN版本:https://www.tensorflow.org/install/source#common_installation_problems 。我们要安装tensorflow1.13.1，因此我们需要下载CUDA10.0，下载地址:https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10 。我们可以选择直接下载完整安装包。
在CUDA安装过程中请选择自定义选项。全部勾选，安装。安装位置默认即可。在安装某些版本的CUDA时，有时会提示显卡驱动不兼容该显卡，可以继续安装，但是可能无法使用CUDA。 这种情况是由于显卡新于该工具包造成的。在这种情况下，建议保留现有驱动并安装CUDA工具包的剩余部分(也是选择自定义安装)。 
**安装完成后，添加下面四个路径到用户环境变量中：**
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
**这四个路径非常重要，如果不添加，会出现import tensorflow时正常不会报错，但当你要执行tf.Session()的时候就会卡住。因为这个时候将会调用cuda，没有路径的话cuda需要的各种lib、dll加载不了。**
上述操作全部完成后打开cmd，输入nvcc -V，如果出现CUDA版本信息，证明路径已经配置好了。
# 下载和安装CUDNN7.4.2
由这个网址:https://www.tensorflow.org/install/source#common_installation_problems 。我们还可以查到各个版本的tensorflow需要安装的CUDNN版本。我们要安装tensorflow1.13.1，因此我们要下载CUDNN7.4.2，下载地址:https://developer.nvidia.com/rdp/cudnn-archive 。
下载文件。解压缩后得到一个文件夹。将这个目录中所有文件复制粘贴到CUDA安装位置：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0即可。
# 下载和安装Anaconda3.5.2
我们可以从anaconda官网上下载anaconda，如果觉得速度慢的话也可以从这个镜像：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 中下载也可以，注意安装anaconda3-5.2.0，这个版本的anaconda默认安装python3.6。
安装过程按默认设置下一步即可。注意不要勾选add anaconda to the system PATH environment variable，我们待会儿会手动设置。
**安装完成后，在用户环境变量中新建一个path，添加下面三个路径：**
C:\Users\zgcr6\Anaconda3
C:\Users\zgcr6\Anaconda3\Scripts
C:\Users\zgcr6\Anaconda3\Library\bin
然后打开cmd，输入conda -V测试一下，如果能正常显示版本号，说明已经配置好了。
# 安装Tensorflow-gpu1.13.1
**我们在anaconda中先新建一个名为tensorflow-gpu1.13.1的环境，该环境安装python3.6。打开anaconda prompt，使用命令：**
```python
//首三行代码是将conda安装源换成清华源(永久设置)，因为官方源是pypi，在国内下载速度太慢了
//清华源是官网pypi的镜像，每隔5分钟同步一次，地址为 https://pypi.tuna.tsinghua.edu.cn/simple
//如果使用pip时想临时用一下清华源，使用命令:pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name
//还可以使用阿里源:https://mirrors.aliyun.com/pypi/simple
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda search tensorflow-gpu //查找tensorflow-gpu的所有版本
conda create -n tensorflow-gpu-1.13.1 python=3.6.6 //新建一个名为tensorflow-gpu-1.13.1的环境
```
然后等待安装完成。该命令创建了一个名为tensorflow-gpu-1.13.1的环境，并安装了python3.6.6。创建的环境位置在：C:\Users\zgcr6\Anaconda3\envs。
**然后我们将下面这两个目录添加到用户变量中，注意要放到anaconda的目录之前：**
C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.13.1
C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.13.1\Scripts
**然后打开cmd，使用下面的命令安装tensorflow1.13.1：**
```python
python -m pip install tensorflow-gpu==1.13.1 //安装tensorflow-gpu1.13.1
//如果觉得慢可以临时用一下清华源
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.13.1
```
等待安装完成即可。
**安装完成后，我们还需要验证tensorflow能否正常调用显卡进行计算。打开cmd.exe，输入python，然后依次输入下面的命令：**
```python
import tensorflow as tf
hello = tf.constant("hello, world!")
sess = tf.Session()
sess.run(hello)
```
当输入到第三句sess = tf.Session()时，即开始调用显卡。注意如果是第一次在系统中使用tensorflow来调用显卡时，在adding visible gpu devices:0这里会卡住3-5分钟，之后如果能够继续运行并正常输出hello world，则说明安装成功。
# 安装PyTorch-gpu1.1
我们还可以再创建一个PyTorch-gpu1.1的环境。pytorch1.1新增加了对tensorboard的支持，因此我们安装这个版本。
**在这个地址:https://pytorch.org/get-started/locally/ ，选择stable(1.1)->linux->pip->python3.6->CUDA10，然后得到两行命令:**
```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```
上面的两行命令是在线安装pytorch和torchvision。由于pytorch比较大，我们可以先将whl文件下载到本地，即命令中的:https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-win_amd64.whl  就是下载地址。如果觉得这个下载地址下载的速度太慢也可以从清华镜像站下载:https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/ 。
**然后我们再创建一个名为PyTorch-gpu-1.1的环境，打开anaconda prompt，使用命令：**
```python
conda create -n pytorch-gpu-1.1 python=3.6.6
```
然后等待安装完成。该命令创建了一个名为pytorch-gpu-1.0的环境，并安装了python3.6.6。创建的环境位置在：C:\Users\zgcr6\Anaconda3\envs。
**然后我们将下面这两个目录添加到用户变量中，注意要放到anaconda的目录之前：**
C:\Users\zgcr6\Anaconda3\envs\pytorch-gpu-1.1
C:\Users\zgcr6\Anaconda3\envs\pytorch-gpu-1.1\Scripts
将上面下载的安装包放在C:\Users\zgcr6\Anaconda3\envs\pytorch-gpu-1.1目录下，然后打开cmd，运行下列命令:
```
pip install torch-1.1.0-cp36-cp36m-win_amd64.whl 
pip install torchvision
```
安装完成后，我们要验证一下安装是否成功。在cmd窗口中输入python，然后依次输入下列命令：
```
import torch
print(torch.cuda.is_available())
```
如果输出为True，则说明我们的安装成功了。
# 如何使用我们安装的tensorflow-gpu1.13.1/pytorch-gpu-1.1环境
## 在cmd中使用tensorflow-gpu1.13.1/pytorch-gpu1.1环境
将tensorflow-gpu-1.13.1环境的路径加入到用户环境变量中，即路径C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.13.1，注意这个路径一定要放在我们刚才添加的路径C:\Users\zgcr6\Anaconda3之前。
此时打开cmd后，输入python运行的就是C:\ProgramData\Anaconda3\envs\tensorflow-gpu1.13.1文件中的python.exe。
如果要使用pytorch-gpu-1.1环境其操作也类似，只要将上面的环境变量中pytorch-gpu-1.1目录放在tensorflow-gpu-1.13.1目录之前即可。

## 在编译器(如pycharm)中使用tensorflow-gpu1.13.1/pytorch-gpu1.1环境
我们以pycharm为例，在pycharm中找到项目的解释器设置(file->settings->project interpreter)，将解释器路径设为我们的C:\Users\zgcr6\Anaconda3\envs\tensorflow-gpu-1.13.1\python.exe或C:\Users\zgcr6\Anaconda3\envs\pytorch-gpu-1.1\python.exe即可。