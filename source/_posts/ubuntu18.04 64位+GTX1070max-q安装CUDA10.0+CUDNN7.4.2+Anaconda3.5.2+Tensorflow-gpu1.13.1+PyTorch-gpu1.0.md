---
title: ubuntu18.04 64位+GTX1070max-q安装CUDA10.0+CUDNN7.4.2+Anaconda3.5.2+Tensorflow-gpu1.13.1+PyTorch-gpu1.0
date: 2019-04-27 21:30:56
tags:
- 各种环境搭建
categories:
- 各种环境搭建
---


# 安装Ubuntu18.04系统
**注意:**
本安装方法设置完成后，不需要使用easybcd设置启动项。开机时是ubuntu系统在引导，选择ubuntu即可进入ubuntu18.04系统，选择windows boot manager即可进入win10系统。
我的电脑是神舟精盾t97e，显卡为GTX 1070MAX-Q，原来的win10系统单独安装在一块硬盘中(只有一个分区，即C盘)，而笔记本中一共有两块硬盘。现在我要在另一个硬盘中安装ubuntu18.04系统。
先从官网上下载ubuntu18.04系统镜像(版本ubuntu18.04.02 64位)，下载地址:https://www.ubuntu.com/download/desktop 。如果觉得官网下载速度慢可以去阿里云镜像站下载:http://mirrors.aliyun.com/ubuntu-releases/ 。然后使用UltralSO将系统镜像刻录到一个U盘中(大于8G)，做成一个ubuntu18.04系统安装盘。
重启win10系统，开机不停按F2键进入bios，如果boot mode为uefi，将secure boot设置成disable(secure boot即只可以启动Win8及以上系统，不能启动其他系统，包括linux)。然后调整启动顺序，将U盘调整成第一启动盘。保存设置并重启系统。选择install ubuntu开始安装。这里核芯显卡+N卡的双显卡电脑容易出现卡死在ubuntu logo界面的情况，我们在移动项到install ubuntu时按e进入启动配置参数界面，倒数第二行的quiet splash ---修改为:
```
quiet splash acpi_osi=linux nomodeset
```
然后按F10重启，就可以正常进入安装界面了。进入Ubuntu安装界面后，语言按你的喜好选English或中文简体，需要注意一点的是，即使选择了中文简体，在ubuntu系统的搜索中也是按英文来搜索的，比如搜索终端必须搜索terminal才能搜索到。
然后选择键盘布局keyboard layout，直接默认即可。下一步选择正常安装normal installation和install third-party software for graphics...，这个是保证我们能正常显示桌面和使用wifi的。
下一步选择安装类型，如果我们想把ubuntu18.04与win10系统安装在同一个盘中，就选择第一项安装ubuntu，与win10共存(install ubuntu alongside windows boot manager)，此时安装程序会自动检测你的硬盘剩余空闲空间，并给ubuntu和win10划分适当的大小，这时安装程序会自动给ubuntu系统划分各个挂载分区的大小。由于我们要把ubuntu18.04安装到另外一个盘，因此我们选择其他选项(something else)，这时我们可以手动设置各个挂载分区的大小。
挂载分区的设置:
* 首先从硬盘中删除某个不用的分区，得到一块未分配空间；
* 先设置一个swap分区(交换空间)，这相当于win10系统中的虚拟内存，设置成和你的物理内存一样大即可，选择逻辑分区、空间起始位置，比如我内存16G，swap就设为16384MB；
* 设置一个home分区，这个分区相当于win10中C盘里的user文件夹，选择逻辑分区、空间起始位置和ext4文件日志系统，挂载点/home，你的图片、视频、下载等文件夹都在这个分区里面，如果你是娱乐向用户，建议将这个分区设置的大一些，比如我将这个分区设为40G；
* 设置一个user分区，选择逻辑分区、空间起始位置和ext4文件日志系统，挂载点/usr，这个分区相当于你的软件安装位置，Linux下一般安装第三方软件你是没办法更改安装目录的，系统都会统一地安装到/usr目录下面，所以这个分区我也设为40G大小。对于ubuntu18.04也可以不分这个分区，系统会在/分区下自动创建一个/usr目录，不过这样你就要把/分区划的更大一些；
* 最后设置一个/分区，注意安装ubuntu18.0时这个分区必须最后分，否则前面的分区就不能分了。选择主分区、空间起始位置和ext4文件日志系统，挂载点/，这个分区相当于win10系统中的C盘，即所有除了user和home之外的系统文件都在这里。这个分区最好不要小于30G，比如我将这个分区设为50G；
* 最后请选择安装启动引导器的设备(device for boot loader installation)这一项为windows boot manager，因为我的ubuntu系统装在另一个盘里，这个windows boot manager所在是一个efi分区，在我的win10系统所在的硬盘上。请注意一定不要单独划boot分区，因为我们这里为了能正常地选择win10/ubuntu18.04双系统中的任意一个进入，使用的是EFI引导替代ubuntu的boot的grub。不用担心windows boot manager所在分区大小不够，因为boot需要的空间只有几十M，完全足够。

设置完成后就可以开始安装了，时区选择上海即可，然后自己设置一个账户和密码，等待安装完成。注意账户和密码不要忘记了。
安装完成后，重启电脑，这时我们可以看到efi引导的几个选项，ubuntu在第一项(进入ubuntu18.04系统)，ubuntu高级选项(进入恢复模式下的ubuntu18.04系统)，windows boot manager(进入win10系统)，有10秒选择时间。我们选择ubuntu系统进入后，点击账户，在输入密码页面时会卡住，这在核芯显卡+N卡的双显卡电脑上经常出现，是因为ubuntu的显卡驱动问题造成的。我们可以先把选项移动到ubuntu，然后按e进入启动配置参数界面，倒数第二行的quiet splash $vt_handoff修改为:
```
quiet splash acpi_osi=linux nomodeset
```
再按F10重启电脑，此时选择ubuntu我们就可以正常进入ubuntu系统。但是上面的命令只是本次生效，要想永久生效，打开终端(terminal），使用命令:
```
sudo -s // 临时获取root权限，获取root权限后仍在当前目录下
gedit /boot/grub/grub.cfg
//在文件中找到quiet splash $vt_handoff处全部修改为
quiet splash acpi_osi=linux nomodeset
reboot //重启系统
```
**注意:**
Ubuntu18.04.02最新发行版不建议强制永久获取root权限，会产生一些莫名其妙的问题。使用自身账户时，你的账户的文档、图片、包括免安装软件等资源都放在home分区下面。你的账户只对home分区有完全的读写执行权限，其余分支如usr你只能在终端使用命令来操作文件，因此，建议将home分区多分配一些空间。
**如果安装好系统后觉得不满意，想单独删除ubuntu系统:**
使用DiskGenius，删除swap、home、usr、/等所有ubuntu分区；
使用DiskGenius，进入windows boot manager分区，删除名为Ubuntu文件夹；
删除UEFI启动项，使用CMD命令:
```
bcdedit /enum firmware //找到Ubuntu系统的对应identifier，复制
bcdedit /delete {xxx}
bcdedit /enum firmware //看看删除是否成功
```
# 安装nvidia显卡驱动
经过上面的设置我们可以正常进入系统了，但我们还需要尽快给系统安装最新的nvidia驱动。因为我们这里是全新的系统，有很多依赖包没有安装，自己获取某个版本的nvidia驱动安装很可能因为缺少依赖包导致安装失败，因此这里建议使用标准Ubuntu仓库进行自动化安装。这种安装方法会检测所有缺少的依赖包进行安装，并安装推荐的nvidia驱动(一般是最新的稳定版驱动)。
使用下列命令:
```
sudo -s
add-apt-repository ppa:graphics-drivers/ppa //添加显卡驱动源,有时网络不畅下载会失败，请换一个比较好的网络再试
apt-get purge nvidia-*  //删除可能存在的已有nvidia驱动
ubuntu-drivers devices //列出你的显卡型号和支持的所有版本的nvidia驱动，因为我的显卡是GTX1070 MAX-Q，这里我看到推荐版本的驱动是418版本的nvidia驱动。
ubuntu-drivers autoinstall //自动安装推荐驱动及其依赖包，需要下载较多文件，请找一个比较好的网络，否则经常会有某些文件下载失败
//或使用apt install nvidia-driver-418
reboot //安装完成后，重启
```
重启系统后，打开terminal终端，使用下列命令观察显卡驱动是否正常运行:
```
nvidia-smi //如果出现GPU列表，里面有你的独立显卡，说明显卡已正常运行
nvidia-settings //显示你的显卡信息
```
# Ubuntu18.04系统更换国内软件源、解决ubuntu与win10系统时间差8小时、安装intelligent pinyin中文输入法、设置右键新建文件选项、安装google chrome、安装git、安装网易云音乐、安装最新稳定版nodejs和npm
**更换为国内阿里云的软件源:**
```
sudo -s
gedit /etc/apt/sources.list 
//源地址:https://opsx.alibaba.com/guide?lang=zh-CN，点击文档，点击左侧菜单ubuntu18.04软件源配置手册即可找到源地址
//打开的文件最开头加上以下国内阿里云软件源(ubuntu官方推荐)，然后保存文件
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
```
**解决ubuntu与win10系统时间差8小时:**
UTC即协调世界时，GMT即格林尼治平时，Windows与ubuntu看待系统硬件时间的方式是不一样的。Windows把系统硬件时间当作本地时间(local time)，即操作系统中显示的时间跟BIOS中显示的时间是一样的。ubuntu把硬件时间当作 UTC，操作系统中显示的时间是硬件时间经过换算得来的，如北京时间是GMT+8，则系统中显示时间是硬件时间+8。这样，当PC中同时有多系统共存时，就出现了问题。
我们可以让Ubuntu不使用UTC时间，与Windows保持一致。
```
sudo -s
gedit /etc/default/rcS
//修改下列内容为:
UTC=no
//然后运行依次下面的命令，重启ubuntu系统即可
timedatectl set-local-rtc 1 --adjust-system-clock
//从windows时间服务器对时，然后把时间同步到硬件上
apt install ntpdate 
ntpdate time.windows.com
hwclock --localtime --systohc
```
**安装intelligent pinyin中文输入法:**
打开设置(settings) ，找到区域和语言(Region&Language)，点击管理已安装的语言(Manage Installed Language)，初次进入会安装些字体等相关信息，重启系统后使之生效。然后点击+添加 Chinese(Intelligent Pinyin)。重启系统后使之生效。

**右键新建文件选项:**
用户主目录中找到模板文件夹，进入，在里面右键打开终端，使用下列命令:
```
gedit 新建txt文件
```
然后保存。此时在其他文件夹中右键就可以看到新建文档->新建txt文件了。

**安装google-chrome浏览器:**
```
sudo -s //获取临时root权限
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb //下载安装包
dpkg -i google-chrome-stable_current_amd64.deb //执行安装包，这一步必须要root权限
google-chrome
//上一步会报错:[31560:31560:0207/085601.085852:ERROR:zygote_host_impl_linux.cc(90)] Running as root without --no-sandbox is not supported. See https://crbug.com/638180.
//google-chrome浏览器默认在root权限下的命令行中不能运行，要运行则须按下面修改:
gedit /usr/bin/google-chrome //修改配置
//文件中修改下列内容，这一步也需要root权限
exec -a "$0" "$HERE/chrome" "$@"改为
exec -a "$0" "$HERE/chrome" "$@" --user-data-dir --no-sandbox
//vim进入文件后按i可在光标处修改文件，修改完成后安ESC退回命令行模式，再输入:进入末行模式，输入wq保存并退出vim(不退出terminal)
//这时我们就可以在命令行中使用google-chrome打开chrome浏览器，但是每次登陆账户后下次都要重新登陆，十分不方便，以后每次使用chrome浏览器都要打开terminal，获取root权限，然后使用google-chrome打开chrome浏览器
//我们不进行上面对root账户的设置，使用下面的设置，这样第一次打开浏览器后可将图标添加到收藏夹，以后直接点击图标即可打开chrome浏览器
sudo -s
chmod -R 777 /home/zgcr/.config/google-chrome //刚才几个提示权限不够的文件夹提升权限
google-chrome //默认安装位置/usr/bin/google-chrome
```
**安装网易云音乐:**
从官网下载安装包，下载地址:https://music.163.com/#/download 。
下载完成后，进入下载文件夹，在该文件夹内右键打开终端，运行下列命令:
```
sudo -s
apt-get -f -y install //解决依赖问题
dpkg -i netease-cloud-music_1.2.0_amd64_ubuntu_20190422.deb //注意后面文件名改成你下载下来的文件
netease-cloud-music //打开网易云音乐，打开后添加到收藏夹，之后点击图标即可打开
```
**安装wps office:**
官方下载地址:https://www.wps.cn/product/wpslinux/# 。下载完成后在文件所在目录右键打开终端，运行下列命令安装:
```
sudo -s
dpkg -i wps-office_11.1.0.8392_amd64.deb
```
**最好用的ssr:**
仓库地址:https://github.com/erguotou520/electron-ssr 。建议下载appimage版本（类似windows的绿色版软件），无需安装，右键属性，勾选允许作为程序执行文件即可运行。注意该软件对chrome浏览器有效，但对ubuntu自带的firefox浏览器无效。
**安装git:**
```
sudo -s //必须获取root权限
apt-get install git
git config --global user.name "Your Name" //换成你的git提交名
git config --global user.email "email@example.com" //换成你的git提交时的邮箱名
```
**安装最新稳定版nodejs和npm:**
```
sudo -s //临时获取root权限
apt update
apt install nodejs //安装nodejs
apt install npm //安装npm
npm install -g n //升级nodejs
n stable //选择稳定版nodejs
node -v //查看目前安装的nodejs版本
npm i -g npm //升级npm
npm -v //查看npm版本
//npm国内下载速度太慢，我们可以安装cnpm代替
//cnpm是一个完整npmjs.org镜像，同步频率目前为 10分钟 一次以保证尽量与官方服务同步
npm install -g cnpm --registry=https://registry.npm.taobao.org //安装cnpm
cnpm -v //显示cnpm版本
```
# 安装CUDA10.0
先使用下面的命令观察显卡驱动是否安装并正常工作。
```
nvidia-smi
nvidia-settings
```
这里我看到我的显卡驱动版本为418.56，在这里:https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 可以发现CUDA 10.0.130这个版本要求驱动版本>= 410.48(对linux系统)，所以完全满足。之所以不用CUDA 10.1.105版本是因为tensorflow-gpu1.13版本最高支持CUDA10.0和cuDNN7.4。我们可以从这个网址中查到各个版本的tensorflow-gpu与CUDA和CUDNN的对应关系:https://tensorflow.google.cn/install/source 。
CUDA10.0下载地址:https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal ，注意选择linux->x86_64->ubuntu->18.04->runfile(local)。然后下载。注意这里的文件一定要重新下载，从U盘复制粘贴过来的文件就不能用了。下载完成后在文件所在文件夹中右键打开终端，使用下列命令安装:
```
sudo -s
sh cuda_10.0.130_410.48_linux.run
```
一直按enter将声明读完。然后输入accept。因为我们之前单独安装了418版本显卡驱动，因此下面一项我们选择n（不安装显卡驱动）。Install the CUDA 10.0 Toolkit?选择y。Enter Toolkit Location直接enter使用默认位置（/usr/local/cuda-10.0）即可。
Do you want to install a symbolic link at /usr/local/cuda?选择y。Install the CUDA 10.0 Samples?选择y。Enter CUDA Samples Location直接enter选择默认位置（/home/zgcr）。然后等待安装完成。
最后提示:
```
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 384.00 is required for CUDA 10.0 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run -silent -driver
```
这是因为我们安装CUDA的时候没有选择安装驱动，提示需要安装驱动，忽略即可。
下面我们要添加环境变量，才能正常使用CUDA。
```
sudo -s
gedit ~/.bashrc //~/.bashrc代表的就是 /home/zgcr/.bashrc
//文件中添加下列内容，如果已有，请合并
export CUDA_HOME=/usr/local/cuda-10.0/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
export PATH="$PATH:$LD_LIBRARY_PATH:$CUDA_HOME"
//保存完成后，检验cuda环境是否配置好
source ~/.bashrc // 使环境变量立即生效
nvcc -V //如果能正常显示CUDA版本号，说明我们的路径配置好了
//再运行一个测试看看
cd ~/NVIDIA_CUDA-10.0_Samples/1_Utilities/bandwidthTest
make
./bandwidthTest
//返回Result = PASS代表cuda安装成功
```
# 安装cuDNN7.4.2
从这个网址下载:https://developer.nvidia.com/rdp/cudnn-archive 。选择Download cuDNN v7.4.2 (Dec 14, 2018), for CUDA 10.0。下载cuDNN Library for Linux、cuDNN Runtime Library for Ubuntu18.04 (Deb)、cuDNN Developer Library for Ubuntu18.04 (Deb)、cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)四个文件。
下载完成后，进入到文件所在目录，右键打开终端，运行下列命令:
```
sudo -s
tar -zxvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
cp cuda/include/cudnn.h /usr/local/cuda-10.0/include/ 
cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/ 
chmod a+r /usr/local/cuda-10.0/include/cudnn.h 
chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb
dpkg -i libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb
dpkg -i libcudnn7-doc_7.4.2.24-1+cuda10.0_amd64.deb
//查看CUDNN版本
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
//如果显示出类似下面的说明，则CUDNN安装成功
//#define CUDNN_MAJOR 7
//#define CUDNN_MINOR 4
//#define CUDNN_PATCHLEVEL 2
//--
//#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
```
# 安装anaconda3.5.2
anaconda3.5.2自带的是3.6.5版的Python，比较合适。从这个地址:https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 中下载Anaconda3-5.2.0-Linux-x86_64.sh即可。这个文件一定要下载得到。
下载完成后，进入文件所在文件夹，右键打开终端，使用下列命令:
```
sh Anaconda3-5.2.0-Linux-x86_64.sh  //安装anaconda，同意协议写yes,后面一路有默认选项就用默认选项，没有就选no
```
安装完成后，我们要把系统启动Python的路径设为anaconda中自带的Python的路径，使用下列命令:
```
sudo -s
gedit ~/.bashrc //~/.bashrc代表的就是 /home/zgcr/.bashrc
//文件末尾加入下面这行，然后保存，重启终端，再输入python启动的就是anaconda的python了
export PATH="/home/zgcr/anaconda3/bin:$PATH:......" //注意路径中换成你自己的用户名
source ~/.bashrc // 使环境变量立即生效
```
检查是否可以正常使用anaconda:
```
conda -V //能正常显示conda版本号则正确安装
jupyter notebook //应当能够正常启动anaconda自带的jupyter notebook，注意要在非root账户时运行jupyter notebook
```
# 安装Tensorflow-gpu1.13.1
之所以选择Tensorflow-gpu1.13.1是因为只有这个版本才支持CUDA10，我们必须安装这个版本，才能正常调用CUDA。我们可以从这个网址中查到各个版本的tensorflow-gpu与CUDA和CUDNN的对应关系:https://tensorflow.google.cn/install/source 。
先在anaconda中单独创建一个Tensorflow-gpu环境，然后在该环境中安装Tensorflow-gpu1.13.1。
```
//首三行代码是将conda安装源换成清华源(永久设置)，因为官方源是pypi，在国内下载速度太慢了
//清华源是官网pypi的镜像，每隔5分钟同步一次，地址为 https://pypi.tuna.tsinghua.edu.cn/simple
//如果使用pip时想临时用一下清华源，使用命令:pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name
//如果pip想永久使用清华源，则使用命令:gedit /.pip/pip.conf
//文件中写入:
//[global]
//index-url = https://pypi.tuna.tsinghua.edu.cn/simple
//还可以使用阿里源:https://mirrors.aliyun.com/pypi/simple
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda search tensorflow-gpu //查找tensorflow-gpu的所有版本
conda create -n tensorflow-gpu-1.13.1 python=3.6.6 //新建一个名为tensorflow-gpu-1.13.1的环境
```
下面我们要把默认的Python环境设为新安装的tensorflow-gpu-1.13.1这个文件夹中的Python环境:
```
sudo -s
gedit ~/.bashrc //~/.bashrc代表的就是 /home/zgcr/.bashrc
//文件末尾加入下面这行，然后保存，重启终端，再输入python启动的就是anaconda的python了
export PATH="/home/zgcr/.conda/envs/tensorflow-gpu-1.13.1/bin:......:$PATH:......" //请注意路径的顺序，把第一个搜索的路径放在最前面即可，这样我们就可以通过修改这个参数切换不同位置的Python环境，注意路径中换成你自己的用户名
//我们在anaconda中创建的环境都在/home/zgcr/anaconda3/envs/文件夹下，上面路径中第一个前添加一个路径:/home/zgcr/anaconda3/envs/tensorflow-gpu-1.13.1/bin，然后保存
source ~/.bashrc // 使环境变量立即生效
```
下面我们要用pip安装tensorflow-gpu-1.13.1。打开终端，使用下列命令:
```
which python //查看现在启动的python的安装位置
python -m pip install tensorflow-gpu==1.13.1 //安装tensorflow-gpu1.13.1
//如果觉得慢可以临时用一下清华源
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.13.1
```
安装完成后，测试是否能正常调用显卡。
```
import tensorflow as tf
hello = tf.constant("Hello, tf!")
sess = tf.Session()
print(sess.run(hello))
```
如果打印出b'Hello, tf!'则说明安装成功。
# 安装PyTorch-gpu1.0
在这个地址:https://pytorch.org/get-started/locally/ ，选择stable(1.0)->linux->pip->python3.6->CUDA10，然后得到两行命令:
```
pip install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```
上面的两行命令是在线安装pytorch和torchvision。由于pytorch比较大，我们可以先将whl文件下载到本地，即命令中的:https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl 就是下载地址。如果觉得这个下载地址下载的速度太慢也可以从清华镜像站下载:https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/ ，选择pytorch-nightly-1.0.0.dev20190409-py3.6_cuda10.0.130_cudnn7.4.2_0.tar.bz2下载即可。
然后我们再创建一个名为PyTorch-gpu的环境:
```
conda create -n pytorch-gpu-1.0 python=3.6.6
```
下面我们要把默认的Python环境设为新安装的pytorch-gpu-1.0这个文件夹中的Python环境:
```
gedit ~/.bashrc //~/.bashrc代表的就是 /home/zgcr/.bashrc
//文件末尾加入下面这行，然后保存，重启终端，再输入python启动的就是anaconda的python了
export PATH="/home/zgcr/anaconda3/envs/pytorch-gpu-1.0/bin:/home/zgcr/anaconda3/envs/tensorflow-gpu-1.13.1/bin: …… :$PATH:......" //请注意路径的顺序，路径之间用:隔开，最后一个路径之后也要有:。把第一个搜索的路径放在最前面即可，这样我们就可以通过修改这个参数切换不同位置的Python环境，注意路径中换成你自己的用户名
//我们在anaconda中创建的环境都在/home/zgcr/anaconda3/envs/文件夹下，上面路径中第一个前添加一个路径:/home/zgcr/anaconda3/envs/pytorch-gpu-1.0/bin，然后保存
source ~/.bashrc // 使环境变量立即生效
```
**注意:**
我们必须先安装CUDA才能安装GPU版的pytorch。下面我们要用pip安装下载到本地的torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl。在文件所在目录右键打开终端，使用下列命令:
```
which python //查看现在启动的python的安装位置，看看我们的环境变量设置是否正确
cp torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl /home/zgcr/anaconda3/envs/pytorch-gpu-1.0/bin
cd /home/zgcr/anaconda3/envs/pytorch-gpu-1.0/bin
python -m pip install torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip list //查看已安装的包
rm -i torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl //删除刚才复制的安装包
python -m pip install torchvision
//如果上面的命令因为网络问题安装失败，那么选择下面的命令
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision 
```
最后我们将安装好的两个环境复制到另一个文件夹做备份。
```
pwd //查看当前终端打开时目录的完整路径
//文件夹中有子文件夹，必须加上-r递归拷贝
cp -r /home/zgcr/anaconda3/envs/tensorflow-gpu-1.13.1 /home/zgcr/下载
cp -r /home/zgcr/anaconda3/envs/pytorch-gpu-1.0 /home/zgcr/下载
```