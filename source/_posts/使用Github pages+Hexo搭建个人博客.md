---
title: 使用Github pages+Hexo搭建个人博客
date: 2019-02-03 21:42:12
tags:
- 博客搭建
categories:
- 博客搭建
---



# 前言

Github Pages可用来管理用户编写的、托管在github上的静态网页。它相当于一个免费的云服务器，省去了自己搭建个人博客云服务器的麻烦。Github Pages也可以绑定自己的域名。

Hexo是一个快速、简洁且高效的博客框架。Hexo可以简单、快速地生成静态网页。

# 搭建Github Pages

GitHub是一个面向开源及私有软件项目的托管平台，因为只支持Git作为唯一的版本库格式进行托管，故名GitHub。对于托管的开源项目，Github是完全免费的。

## 申请Github账号

打开Github网站: https://github.com/ ，点击sign up for github，申请一个Github账号。根据提示一步一步填写即可，和其他网站注册没什么区别。

## 创建Github Pages的Github仓库

在Github上新建一个仓库，仓库名必须写为:yourname.github.io，Description描述相当于备注，可以不写。仓库类型就是Public不变。注意勾选Initialize this repository with a README。然后点击Create repository创建仓库。

## 开启Github Pages

创建好Repository后，点击Settings，观察Github Pages下面是否显示为：

```
Your site is ready to be published at https://yourname.github.io/
```

如果显示上面的提示，说明你的Github Pages开启了。然后点击choose a theme，选择后会跳转到下面的界面，我们可以从中可以选择一个主题。选择后，点击select theme即可。点击后会跳到edit README.md的界面，这里我们直接commit changes就好了。

此时我们就可以通过：https://username.github.io (username为你的用户名)访问你刚刚创建的这个个人博客了。当然此时的博客页面是默认的，就是你的仓库中的README.md文件。注意个人博客的网站内容是在该仓库的master分支下的文件。

## 域名购买和设置解析

此时我们要访问这个博客只能通过网址：https://username.github.io 来访问，且百度等搜索引擎是不抓取Github Pages生成的博客的网址的，因此我们需要购买一个域名，然后将域名解析到上面的网址，这样就可以通过另一个域名来访问这个博客了。

域名购买可以在阿里云购买，购买后选择控制台->域名->解析来设置域名解析指向我们上面的Github Pages博客的外网IP地址。

首先我们要知道我们的Github Pages个人博客网站的点分十进制IP地址。打开cmd.exe，使用下面的命令就可得到博客的外网IP地址：

```
ping username.github.io # username为你的Github名
```

然后在阿里云域名解析中添加一个记录，记录类型选择A，主机记录就是我们的域名和域名的前缀(域名是你购买好的，前缀可以自己设置)，记录值就是我们的个人博客网站的点分十进制IP地址，TTL10分钟表示设置好后10分钟才生效。生效后我们就可以通过域名(即上图中的主机记录网站)来访问我们这个虚拟机实例了。一般我们设置两个前缀，@表示没有前缀，还有一个就是常用的www前缀。当然你也可以自己设置一个其他的前缀。

我们也可以选择添加一个CNAME类型记录而不是A类型的记录，此时只要把记录值改为username.github.io即可。

添加好记录后，在Github的仓库设置页面Github pages部分设置custom domain为你的个人域名。

## 下载Github桌面客户端

下载这个客户端主要是为了方便管理本地仓库中的资料（存在我们的电脑上和Github网站上我们的账号中的仓库同名的文件夹及其下的所有资料)。

从这个网站：https://desktop.github.com/ 上下载和你的操作系统平台对应的版本，按默认选项安装。

打开Github桌面客户端，点击file->options->sign in登录你的Github网站账号。登录后点击file->clone repository,将你的博客对应的那个仓库从github网站上下载下来到本地。Github桌面客户端可以可视化地追踪你的仓库的修改记录。

# 安装和设置Hexo

## 下载和安装Node.js

使用Hexo需要先安装Node.js。Node.js官网下载地址：https://nodejs.org/en/ 。建议下载推荐安装版本。安装过程一路默认设置安装即可。注意安装完后要将Node.js路径：C:\Program Files\nodejs\ 添加到环境变量中。添加完成后，我们打开cmd.exe，输入node --version，若能正常显示Nodejs版本号则说明安装成功。

## 下载和安装Git

Git是一个免费的开源分布式版本控制系统，我们可以使用Git软件将本地的某个仓库(就是一个目录及其下的文件)与Github上你的账号中的某个在线仓库关联起来，当我们在本地仓库完成修改确认无误后，可以用Git软件将本地仓库的内容push到在线仓库上，我们也可以将在线仓库的内容pull到本地仓库中。

Git下载地址：https://git-scm.com/downloads 。安装时按默认配置安装即可。

## 安装Hexo

新建一个空目录(我们的个人博客的所有静态网页都在这个目录下)，然后在这个目录下鼠标右键选择Git Bash here。依次运行下列命令(一个指令运行完再运行下一个)：

```
npm install hexo-cli -g # 安装hexo
hexo init # 初始化hexo博客
npm install # 安装npm
hexo g 或 hexo generate # 生成博客静态网页
hexo s 或 hexo server # 启动本地服务器
```

这时再打开浏览器，输入http://localhost:4000 就可以查看我们在本地仓库生成的个人博客静态网页。需要注意的是，此时我们用hexo创建的新个人博客还没有部署到Github上对应的仓库中。

**注意：**

如果你使用的是Microsoft Edge浏览器，访问http://localhost:4000/ 时可能会失败，原因是这个浏览器经常偷偷地自动打开使用代理服务器127.0.0.1。关闭后即可正常访问。还有一种可能是你的4000端口被占用了。

## hexo目录介绍和hexo博客配置

```
.
├── .deploy_git
├── public
├── scaffolds
├── scripts
├── source
|   ├── _drafts
|   └── _posts
├── themes
├── _config.yml
└── package.json
```

.deploy_git：执行hexo deploy命令后部署到GitHub/Gitlab上的内容都在这个目录中。
public：执行hexo generate命令，输出的静态网页内容都在这个目录中。
source：站点资源目录，你写的文章，素材等等都是放在这个目录下,包括以后你需要新建的菜单项如about页、tags页、categories页等也是放在这里。
_drafts：草稿文章。
_posts：成功发布的文章都在这个目录下。
themes：主题文件目录。
_config.yml：hexo博客全局配置文件，注意和同名的主题目录下的配置文件区别开。

_config.yml.yml文件中各部分代码的作用如下：

```
# Site
title: Hexo
subtitle:
description:
author: John Doe
language:
timezone:
```
Site配置是对博客站点的描述，title和subtitle分别是博客的标题和副标题，description中内容对搜索引擎收录博客会有帮助。在hexo的大多数主题中，一般只有标题和副标题会显示在页面上。

```
# Extensions
## Plugins: https://hexo.io/plugins/
## Themes: https://hexo.io/themes/
theme: landscape
```

Extensions配置是用来定义主题和配置插件的，官方默认的主题叫做landscape，就是那个天际地平线的主题。这里推荐使用next主题。只要把主题的文件从其Github仓库地址上git clone 来，放到 themes/ 下相应的目录中，然后我们修改博客根目录下的 _config.yml文件的theme:项为相应的主题名就可以使用其他主题了。

## 使用hexo创建文章

使用下面的命令创建文章：

```
hexo new "文章名" #新建文章
```

文章创建后，我们再次执行：

```
hexo clean
hexo g
hexo s
```

在浏览器中输入http://localhost:4000 ，就可以在本地查看我们刚新建的文章了。注意我们新建这个文章时只指定了文章的标题，其内容是空的。文章的内容需要我们自己用markdown编辑器编辑。新建的文章在hexo博客源码目录\source\_posts目录下。

## hexo文章yaml文件头设置

使用任意一种markdown编辑器打开上面的测试文章.md文件。这里我使用typora来编辑.md文件。打开的.md文件首部有一个自动生成的yaml文件头。文件头规定了文章的基本设置信息，文件头是一段代码块的形式。其中title是博文的标题，date是博文的发表日期，tags是博文的标签，categories是文章所属的目录。比如我们在该文章中添加如下内容以设置文章格式：

```
title: 测试文章
date: 2019-01-29 20:10:05
tags:
	- 标签1
	- 标签2
categories: 
	- 分类1
	- 子分类1
```

在这段代码块之后就可以写文章的内容了。编辑完成后，保存.md文件，再次运行：

```
hexo clean
hexo g
hexo s
```

然后在浏览器中打开http://localhost:4000 即可查看到我们新编辑的文章。

## 创建Hexo博客分类页和标签页

使用下面的命令分别创建分类页和标签页，注意着两个页面的.md文件不需要修改。

```
hexo new page categories # 创建分类页
hexo new page tags # 创建标签页
```

## 下载和安装Hexo自定义主题:next主题

打开Hexo目录下的_config.yml文件，找到theme: landscape，即Hexo的默认主题是这个landscape主题。但是这个主题不太好看，现在我们要将默认主题换成第三方主题：next主题。

next主题官网下载地址：http://theme-next.iissnan.com/getting-started.html 。使用下面的命令下载主题。

```
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

下载完成后，打开Hexo目录下的_config.yml文件，找到theme字段，并将其值更改为next。

然后使用下面的命令，再打开浏览器，输入http://localhost:4000 ，就可以看到next主题的网页效果。

```
hexo clean
hexo g
hexo s
```

# hexo部署到Github pages

## 本地博客部署到Github前的设置

编辑根目录下_config.yml文件，修改相应部分为以下内容：

```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo: git@github.com:zgcr/zgcr.github.io.git
  branch: master
```

注意上面的repo改成你自己的Github个人博客仓库的地址。保存后安装一个扩展：

```
npm install hexo-deployer-git --save
```

为了让部署后的博客文章能够正常显示自带的图片，编辑根目录下_config.yml文件：

```
post_asset_folder: True
```

## 检查SSH keys的设置和添加新生成的SSH Key到Github

仍然是Hexo目录下鼠标右键点击Git bash here，依次输入以下命令：

```
cd ~/.ssh
ls # 此时会显示一些文件
mkdir key_backup
cp id_rsa* key_backup
rm id_rsa*
# 以上三步为备份和移除原来的SSH key设置
```

如果第一步cs ~/.ssh出现cd: /c/Users/1234/.ssh: No such file or directory，说明你还没有生成过自己的ssh key(或在windows下查看[c盘->用户->自己的用户名->.ssh]下是否有id_rsa、id_rsa.pub文件)，那么使用命令：

```
git config --global user.name "your_name"
git config --global user.email "your_email"
# --global参数表示你这台机器上所有的Git仓库都会使用这个用户名和邮箱地址的配置，当然也可以对某个仓库指定不同的用户名和Email地址
# 用户名和邮箱地址是本地git客户端的一个变量,每次commit都会用用户名和邮箱纪录,github的contributions统计就是按邮箱来统计的。
ssh-keygen -t rsa -C "your_email@your_email.com #生成新的key文件,邮箱地址填你的Github地址
```

然后一直按enter即可。下面三步就不用执行了。到此为止，我们生成了一个新的SSH keys。

然后打开你的Github，选择settings，再选择SSH and GPG keys，添加一个新的key。我们找到C:\Users\用户名\.ssh\id_rsa.pub文件,以文本方式打开。将其内容全部复制到key中，点击add ssh key即可。

添加完成后测试一下，Git bash中输入下列命令，之后让你选yes/no，选yes就行了。

```
ssh -T git@github.com
```

## 将本地博客部署到Github

使用命令：

```
hexo d
```

这时我们再刷新：https://username.github.io 就可以看到你的线上博客也更新了。

如果你的博客文章中含有图片，这里强烈建议将图片放到云存储上，推荐使用腾讯云存储，目前可以免费使用一年。当然你也可以将图片作为静态网页的内容打包上传，但是这样做的话到后面文章越来越多时，你每次上传一篇新文章都需要重新编译全部静态网页，速度会越来越慢。

## 使用Hexo管理个人博客后的域名解析设置

上面我们已经说明了在Gitpages中管理域名解析设置的方法，我们将一个自己购买的域名解析到我们的个人博客的点分十进制IP地址上。如果我们要使用Hexo来管理个人博客，上面的设置在每一次Hexo deploy后都会失效，要想保持域名解析一直生效，那么在Hexo中也要对域名解析做相应设置。

在Hexo目录下的source文件夹中新建一个名字叫CNAME的文件，用记事本打开，里面输入我们购买的域名。比如我们购买了abcde.com这个域名，那么我们就在里面输入 www.abcde.com 这个域名(前缀要和你在阿里云中设置的解析前缀一致，但开头不要添加http(s)://)，然后保存，文件后缀名也去掉。注意CNAME文件只能填写一个域名，这一步的目的是告知GitHub，我们要使用另一个域名来访问我们的博客，然后gitHub会将这个域名当作我们仓库的主域名，如果访问userName.github.io，会自动跳转至我们设置的域名。

然后我们使用下面的命令，将更改推送到Gitpages上即可。现在我们只要访问自己购买的域名即可访问我们的个人博客，另外此时即使访问 https://username.github.io 也会跳转到自己购买的域名上。

```
hexo clean
hexo g
hexo d
```

## Hexo常用命令

```
hexo clean # 清除缓存文件(db.json) 和已生成的public目录下的静态文件，建议在hexo g命令前运行此命令
hexo g # hexo generate的简化命令，用于生成静态文件
hexo s # hexo server的简化命令，用于启动本地服务器，一般用于测试
hexo d # hexo deploy的简化命令，用将本地内容部署到远程仓库
```

到这里我们已经讲完了搭建Github Pages个人博客的全部操作，现在你可以在你的个人博客里写文章了。







