---
title: ubuntu系统基本设置
date: 2019-02-21 16:05:10
tags:
- 各种环境搭建
categories:
- 各种环境搭建
---

# 准备工作
先关闭所有虚拟机系统，然后在VMware软件中选择虚拟机界面，点击编辑虚拟机设置，将网络适配器更改为桥接模式(勾选复制物理网络连接状态)。
# Ubuntu系统关闭休眠和自动锁屏
如果是ubuntu14.04系统，打开system setting，找到brightness&lock项，将screen off设为never，并把lock打到off。
如果是ubuntu18.04系统，打开settings，找到power，将black screen调到never。
# ubuntu开启右键Terminal选项
在search框中输入Terminal，然后打开Terminal终端。输入下列命令：
```
sudo apt-get install nautilus-open-terminal
```
安装完成后，重启系统即可。
上面针对的是ubuntu14.04系统，如果是ubuntu18.04系统，则默认开启右键Terminal选项。
# ubuntu开启root账户登录
在search框中输入Terminal，然后打开Terminal终端。我们先临时获得root账户权限，然后设置root账户密码。使用下面的命令，然后设置root账户密码即可。
```
sudo passwd root 
```
设置完成后，如果我们想在Terminal中切换到root账户，可使用下面的命令，然后输入上一步你设置的root账户密码即可。
```
su root
```
如果我们想在ubuntu系统登录时可以选择用非root账户或root账户登录，可在Terminal终端中使用下列命令(先在终端中切换到root账户)：
```
sudo gedit /usr/share/lightdm/lightdm.conf.d/50-unity-greeter.conf
```
然后在打开的文件中添加下列内容，保存后重启系统。
```
[SeatDefaults]
greeter-session=unity-greeter
user-session=ubuntu
greeter-show-manual-login=true # 手动输入登陆系统的用户名和密码
all-guest=false # 禁用来宾账户
```
有时重启后会出现下面的错误提示：
```
error found when loading /root/.profile:
stdin: is not a tty
```
此时我们打开Terminal终端，使用下面的命令(注意终端中切换到root账户)：
```
sudo gedit /root/.profile
```
然后将内容中mesg n替换成tty -s&&mesg n，再重启系统即可。

如果是ubuntu18.04系统，在按上面要求修改50-unity-greeter.conf文件后重启，登录root账户时可能会出现下面的错误提示：
```
sorry,that didn't work.Please try again.
```
此时我们还要去除gdm登录用户名检测。输入下面的命令：
```
gedit /etc/pam.d/gdm-autologin
```
删除或注释掉文件中的下面这行语句，然后保存。
```
auth required pam_succeed_if.so user != root quiet_success
```
再输入下面的命令：
```
gedit /etc/pam.d/gdm-password
```
也删除或注释掉文件中的上面那行语句。然后保存，重启系统。重启系统后也可能出现类似的error found when loading /root/.profile:错误提示，处理方法和上面类似。将文件最后一行：
```
mesg n || true
```
替换为：
```
tty -s &&mesg n || true
```
# ubuntu安装中文输入法
打开settings，找到language，点击manage installed languages。有自动安装的界面可以直接cancel。然后点击install/remove languages，找到Chinese(Simplified)，点击apply，等待下载和安装完成。keyboard input method system选择IBUS。 这里完成的只是中文语言包的安装，还并不能使用中文输入法。
我们还要安装IBUS框架，使用下面的命令安装和启动框架。
```
sudo apt-get install ibus ibus-clutter ibus-gtk ibus-gtk3 ibus-qt4
im-config -s ibus
```
再安装拼音引擎和设置IBUS，使用下面的命令。
```
sudo apt-get install ibus-pinyin
sudo ibus-setup
```
打开IBUS设置后，点击input method，再点击add添加chinese-pinyin。如果没有找到chinese-pinyin，请重启系统后再试一次。然后打开settings->language(如果是ubuntu14.04系统，打开System Settings–>Text Entry，并将Show current input source in the menu bar勾选上)，点击+号，添加chinese(intelligent pinyin)和english(US）即可。
ubuntu系统快速切换输入发快捷键为win+空格。
# ubuntu中gcc和g++编译与运行
一个程序的编译分为下列几个阶段：
* 编译；
* 汇编；
* 链接；
* 运行。
  编译前先使用：
```
gcc --version
g++ --version
```
可检查ubuntu系统上是否安装了gcc或g++以及所安装的版本。如果未安装gcc或g++，可使用下面的命令安装：
```
apt install gcc
apt install g++
apt-get install g++
```
创建一个名为hello.c的文件，内容如下，将该文件放在download文件夹下，然后右键open in Terminal。
```
#include <stdio.h>
main() {
  printf("hello world\n");
}
```
然后使用gcc或g++编译和链接文件：
```
gcc hello.c -o hello  //hello.c为要编译的.c文件，hello为生成的可执行文件的文件名
g++ hello.cpp -o hello  //hello.cpp为要编译的.cpp文件，hello为生成的可执行文件的文件名
```
然后使用下面的命令即可运行生成的可执行文件：
```
./hello
```
# win10系统与VMware上Ubuntu虚拟机互相复制和粘贴
打开VMware软件，打开ubuntu虚拟机系统，然后选择虚拟机->安装VMware tools，我们可以发现虚拟机系统文件夹中的Devices项下多了一个VMware tools的tar.gz包。选中该包，将该文件复制到任一你指定的文件目录下(比如复制到Downloads目录下)，然后在该目录右键打开terminal，用下面的解压命令解压：
```
tar -xzvf VMwareTools-10.2.0-7259539.tar.gz
```
为了便于后续操作，我们可将解压后的文件夹vmware-tools-distrib改成相对简单的名字，如“vm”。进入目录vm，右键打开terminal，使用下面的命令开始编译，一路按Enter键直到出现Enjoy。
```
./vmware-install.pl 
```
安装完成后务必重启Ubuntu虚拟机系统，使VMware tools生效。重启后，你就可以从windows 复制一个文件，然后粘贴到虚拟机系统中，反之亦然。