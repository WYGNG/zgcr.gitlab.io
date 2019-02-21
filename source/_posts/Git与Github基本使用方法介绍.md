---
title: Git与Github基本使用方法介绍
date: 2019-02-21 18:17:05
tags:
- Github
categories:
- Github
---

# Git一些需要注意的要点
如果你使用Windows系统，要在windows系统中某目录下建立本地仓库，为了避免遇到各种莫名其妙的问题，请确保目录名（包括父目录）不包含中文。
包括Git在内的所有的版本控制系统，只能跟踪文本文件的改动，比如TXT文件，网页，所有的程序代码等等。但图片、视频这些二进制文件，没法跟踪文件的变化。版本控制系统只知道图片从100KB改成了120KB，但到底改了什么，版本控制系统不知道。因此如果更新的文件中有图片和视频，那么这个文件只能全部重新上传。
因为各种语言使用不同的编码，比如中文有常用的GBK编码，日文有Shift_JIS编码，为了考虑最大兼容性的问题，强烈建议使用标准的UTF-8编码，该编码包含所有语言，既没有冲突，又被所有平台所支持。
千万不要使用Windows自带的记事本编辑任何文本文件。因为记事本在保存UTF-8编码的文件时会在每个文件开头添加0xefbbbf（十六进制）的字符，这样网页第一行可能会显示一个“?”，正确的程序一编译就报语法错误，等等。
**建议使用notepad++代替记事本，并且把默认编码设置为为UTF-8 without BOM。如果使用其他编译器，也要把默认编码设置为UTF-8 without BOM。**
# Github上新建/删除远程库
点击Github网站上右上角你的账号，选择your repositories，点击new，创建一个库。给新版本库命名并写介绍（description），选择public/private需要收费，点击create repository。
如果想要删除这个库，点击库名下的settings，找到Danger Zone，点击delete this repository。
# 新建/删除本地库
首先安装Git软件。安装完成后在我们的PC中新建一个目录，在这个目录下右键点击Git bash here，使用下面的命令即可新建一个本地库。
```
git init
```
要删除本地库，只要将库目录下的隐藏文件夹.git删除即可。我们可以直接删除.git文件夹，也可使用下面的命令：
```
git branch  # 显示所有本地分支 （初始化时只有一个master分支）
ls -a    # 找到目录下.git文件夹
rm -rf  .git   # 删除.git文件夹
```
# Git基本概念介绍
* 工作区：就是你的本地版本库所在的目录。即.git隐藏文件夹所在的目录。
* 本地版本库：工作区有一个隐藏目录.git，这就是Git的本地版本库。
* 暂存区：在Git的本地版本库中。版本库中还有Git为我们自动创建的第一个分支master，以及指向master的一个指针叫HEAD。
* git add命令将文件修改提交到暂存区；
* git commit -m命令将文件修改进一步加入到本地版本库的当前操作分支；
* 如果文件修改已经git commit -m，这时我们不想要这次修改了，可以使用下面的命令进行版本回退：
  git reset --hard HEAD~1 # 1代表回到第前1个版本
* 分支：在Git中每一次的提交，Git都会把它们串成一条时间线，这条时间线就是一个分支。在Git中，默认会创建一个主分支master，我们的提交都在这个master分支上。HEAD指针指向分支名，分支名指向提交。默认情况下，HEAD指针指向master分支，master分支指向当前提交。
# 查看和修改Git全局用户名和邮箱
使用下面的命令来查看用户名和邮箱：
```
git config --list 
```
如果你没有初始化过用户名和邮箱，那么使用下面的命令来设置：
```
git config --global user.name "输入你的用户名"
git config --global user.email "输入你的邮箱"
```
如果你已经初始化过用户名和邮箱，现在想修改，那么使用下面的命令：
```
git config --global --replace-all user.name "输入你的用户名"
git config --global --replace-all user.email "输入你的邮箱" 
```
# Git创建/切换/删除/合并分支
我们可以使用下面的命令来查看每次提交对应的分支：
```
git log --oneline --graph
```
要创建一个新分支，请使用下面的命令：
```
git branch dev # 建立一个名为dev的分支
git checkout -b dev # 建立一个名为dev的分支并同时将HEAD指针移动到dev分支上，这样在之后我们所做的所有修改都是在dev分支上进行
```
要切换当前操作分支，请使用下面的命令：
```
git checkout dev # 切换到名为dev的分支
```
要删除某个分支，请使用下面的命令：
```
git branch -d dev # 删除名为dev的分支
```
假如要将dev分支(开发的新版本)合并到master分支(客户目前使用的版本)上，使用下面的命令即可：
```
git merge --no-ff -m 'keep merge info' dev # 首先要确定我们在master分支上
```
其中--no-ff表明我们要留下这个merge的信息在log里(不写--no-ff则默认就是--ff模式，不写-m，则本次合并没有标签)。
ff代表Fast-forward信息，即“快进模式”，也就是直接把master指向dev的当前提交，所以合并速度非常快。
如果合并时发现冲突，需要手动编辑产生冲突的文件，保留想留下的内容，然后再保存文件，add并commit。
# 将本地库提交到远程库
如果是首次提交，首先要创建本地库。使用上面的命令新建本地库后，在该目录下进行一些创建/修改/删除操作后，使用下面的命令来提交：
```
git add -A
git commit -m "本次提交描述"
git remote add origin https://github.com/zgcr/111.git # 第一次提交需要本句指令，上面是示例地址，请使用你的Github远程库的https地址
git push -u origin master # 第一次推送时加-u，以后不必加-u
```
**几种git add指令的区别：**
* git add . ：添加新文件(new)和被修改(modified)文件到暂存区（不包括被删除的文件）。
* git add -u ：添加被修改(modified)和被删除(deleted)文件到暂存区（不包括新文件）。
* git add -A：添加所有变化的文件到暂存区(包括新建、修改、删除文件)。

如果不是首次提交，那么使用下面的命令即可：
```
git add -A
git commit -m "本次提交描述"
git push origin master 
```
# 将Github远程版本库的内容拉取到本地库
使用下面的命令即可：
```
git pull origin master # 将origin主机的master分支拉取并合并到本地默认分支上(一般是master分支)
git pull origin master:my_test # 将origin主机的master分支拉取并合并到本地的my_test分支上
```
# Github上某个远程库中下载单个文件或子文件夹
如果要下载单个文件，直接点击该文件名打开进入文件的详细信息的页面，点击右上角Raw即可下载。
如果要下载子文件夹，可以打开这个网站：https://minhaskamal.github.io/DownGit/#/home 。将要下载的子文件夹页面的网址填入，然后点击download即可。(注意该网站可能需要代理才能打开)