---
title: JetBrains CLion/IDEA/Pycharm基本设置
date: 2019-02-20 21:32:38
tags:
- 编译器环境搭建
categories:
- 各种环境搭建
- 编译器环境搭建
---

# CLion设置代码字体大小/设置Tab键退四格/安装cygwin编译器/设置project编译器
打开CLion，选择file->settings->editor->font，选择Font可修改字体，选择Size可修改字体大小，选择Line spacing可修改行距。
选择file->settings->editor->code style->c/c++，勾选Use tab character和Smart tabs，Tab size选择4，这样tab键就是退四格。
我们需要先安装cygwin编译器。首先从官网：https://cygwin.com/install.html 上下载cygwin安装程序。下载完成后，开始安装，选择ininstall from internet，然后指定一个下载目录。代理设置选择use system proxy settings。为了下载更快，我们添加并选择清华镜像源：https://mirrors.tuna.tsinghua.edu.cn/cygwin/ 。然后在search中依次输入gcc-core、gcc-g++、make、gdb、binutils搜索，搜索的结果中点devel文件夹（其他文件夹不用管）左边的+号，然后点击对应项的skip，出现版本号，即表示安装该项。全部选择好后，安装即可。
安装完cygwin后，我们要在Clion中设置cygwin为project的编译器。选择file->settings->editor->build,execution,deployment->toolchains，点击+号可以创建一个新环境。environment下拉框选择你想用的编译器，注意编译器要事先安装好。推荐CLion使用cygwin编译器(也可以用minGW或visual studio)，make、c compiler、c++ compiler的路径系统会自动在你的编译器路径下寻找，你也可以点击右边的...自己去直接找到路径，文件名就是图中的文件名。

# IDEA设置代码字体大小/设置Tab键退四格/设置project解释器/project添加第三方jar包
打开IDEA，选择file->settings->editor->font，选择Font可修改字体，选择Size可修改字体大小，选择Line spacing可修改行距。
选择file->settings->editor->code style->java，勾选Use tab character和Smart tabs，Tab size选择4，这样tab键就是退四格。
我们还要给IDEA设置java解释器。首先下载一个java8的JDK，官网下载地址：https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html 。按提示安装即可。然后打开IDEA，选择file->project structrue->modules，选择你要设置解释器的project，修改 “Sources” 选项卡下的 “Language level” 及 “Dependencies”选项卡下的 “Module SDK” 。因为前面安装的是Java 8 JDK，所以Language level也选择8。dependencies也选择同样的Java 1.8 SDK。
然后我们还要修改Java Compiler。选择File --> Settings --> Build,Execution,Deployment --> Compiler --> java Compiler，将Project bytecode version改成和JDK一样的版本，或者选same as language level。
我们以JDBC-MySQL驱动包为例介绍如何在IDEA project中添加jar包。打开在IntelliJ IDEA中打开要添加jar包的Project，然后选择File->Project Structure->Libraries，点击+号，选中要添加的jar包,点击ok，添加成功。
# PyCharm设置代码字体大小/设置Tab键退四格/设置project解释器
打开PyCharm，file->settings->editor->font，选择Font可修改字体，选择Size可修改字体大小，选择Line spacing可修改行距。
选择file->settings->editor->code style->Python，勾选Use tab character和Smart tabs，Tab size选择4，这样tab键就是退四格。
打开PyCharm，选择file->settings->project:项目名->project interpreter，然后设置这个project的Python解释器即可。

