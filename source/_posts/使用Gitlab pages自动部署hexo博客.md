---
title: 使用Gitlab pages自动部署hexo博客
date: 2019-02-05 14:38:49
tags:
- 博客搭建
categories:
- 博客搭建
---


# Github pages部署hexo博客的缺点

使用Github+hexo搭建个人博客非常简单方便，但是github目前不允许百度蜘蛛的爬取，因此，如果我们希望自己用Github pages+hexo搭建的个人博客能被百度收录，就需要采用其他的方法。网上也有一些利用coding.net进行托管让百度爬取的方法，但是现在coding.net在打开托管的个人pages时会强行加入一个5s的等待跳转页面，这会导致百度的蜘蛛无法正确爬取到博客的内容。

现在我们有了一个更好的选择：使用Gitlab+hexo搭建个人博客。Gitlab搭建的个人博客可以被百度正常收录，另外，在Gitlab上部署hexo博客时需要在Gitlab服务器端完成生成静态网页和部署两个阶段，而Github是在本地生成静态网页的。

具体来说，我们将Hexo博客源码的代码提交到Gitlab，并利用Gitlab服务器的CI/CD功能发布到Gitlab博客，从而实现可以通过 https://username.gitlab.io/projectname 或者 https://username.gitlab.io 访问个人博客。

# 安装和配置环境

本教程系统平台为win10 64位家庭中文版。

## 安装Git

从Git官网下载Git软件，按默认选项安装即可。

## 安装Node.js

从node.js官网下载Node.js软件，推荐下载稳定版本，然后按默认选项安装即可。安装完成后，打开cmd.exe，输入：

```
node -v
```

若能正常显示版本号，则说明安装成功。

## 安装Hexo、预览本地hexo博客

在本地新建一个目录如hexo_blog用来存储hexo博客源码，然后在该目录下打开cmd.exe或Git bash here，依次运行下列命令(一个指令运行完再运行下一个指令)：

```
npm install hexo-cli -g #安装hexo命令行工具
hexo init #下载hexo博客初始源码
npm install #安装npm
hexo s 或 hexo server #启动本地服务器,可预览本地hexo博客
```

如果正常安装完成，在浏览器中访问：http://localhost:4000/ 就可以看到运行在本地服务器上的博客了。

**注意：**

如果你使用的是Microsoft Edge浏览器，访问http://localhost:4000/ 时可能会失败，原因是这个浏览器经常偷偷地自动打开使用代理服务器127.0.0.1。关闭后即可正常访问。还有一种可能是你的4000端口被占用了。

# 将hexo博客部署到Gitlab pages上

## 创建Gitlab pages仓库

首先在Gitlab上注册一个账号，或者直接用Github账号登录也可以。

Gitlab支持project page和user page两种page，只需要创建对应的仓库即可。如果你要创建一个project page，假设project name为blog，那么你的project URL为：https://gitlab.com/username/blog/ 。一旦这个project启用了GitLab pages，并构建了站点，站点访问url为：https://uesrname.gitlab.io/blog  。如果你要创建一个user page，假设此时project name为john.gitlab.io(john为你的Gitlab账号名)，则你的project URL为：https://gitlab.com/john/john.gitlab.io 。一旦这个project启用了GitLab pages，并构建了站点，站点访问url为：https://john.gitlab.io 。

这里我们新建一个名为username.gitlab.io的仓库，username就是你的Gitlab账户名。

## Gitlab账号添加SSH-key
点击你的Gitlab账号右上角头像，选择settings，然后在左侧菜单选择SSH keys。然后检查一下你的本地SSH key的情况，在hexo博客源码目录下点击Git bash here，依次输入以下命令：

```
cd ~/.ssh
ls #此时会显示一些文件
mkdir key_backup
cp id_rsa* key_backup
rm id_rsa*
#以上三步为备份和移除原来的SSH key设置
```

如果第一步cs ~/.ssh出现cd: /c/Users/1234/.ssh: No such file or directory，或者第二步ls后没有文件，则说明你的本地目前没有ssh key(或在windows下查看[c盘->用户->自己的用户名->.ssh]下是否有id_rsa、id_rsa.pub文件)，那么使用命令：

```
git config --global user.name "your_name"
git config --global user.email "your_email"
# --global参数表示你这台机器上所有的Git仓库都会使用这个用户名和邮箱地址的配置来提交，当然也可以对某个仓库指定不同的用户名和Email地址
# 用户名和邮箱地址是本地git客户端的一个变量,每次commit都会用用户名和邮箱纪录,github的contributions统计就是按邮箱来统计的。
ssh-keygen -t rsa -C "your_email@your_email.com #生成新的key文件,邮箱地址填你的Github地址
```

然后按三次enter生成一个新的SSH-key。打开id_rsa.pub文件，将里面的所有内容添加到上面Gitlab账号settings中的SSH-key中。

## hexo博客源码目录下添加.gitlab-ci.yml文件

使用Gitlab部署hexo博客和Github pages不同。在Github上部署博客，需要先在本地生成各种静态网页和文件，然后再推送到Github-pages仓库上就可以直接访问了。使用Gitlab需要在服务器端完成生成和部署两个阶段，需要在本地的hexo博客源码目录下添加一个.gitlab-ci.yml文件用来指导服务器如何处理你提交的源文件。

最新的.gitlab-ci.yml文件官方版本可以从这个仓库中获取：https://gitlab.com/pages/hexo/blob/master/.gitlab-ci.yml 。

.gitlab-ci.yml内容如下：

```
image: node:8.11.2

pages:
  cache:
    paths:
    - node_modules/

  script:
  - npm install hexo-cli -g
  - npm install
  - hexo clean
  - hexo generate
  - hexo deploy
  artifacts:
    paths:
    - public
  only:
  - master
```

## hexo博客本地源码推送到Gitlab pages仓库

在hexo博客源码目录点击鼠标右键Git bash here，依次运行下列命令：

```
git init
git add -A
git commit -m "init blog"
git remote add origin git@gitlab.com:username/username.gitlab.io.git
git push -u origin master
```

这样我们就将hexo博客本地源码推送到Gitlab pages仓库上了。

## 开启Gitlab pages CI/CD生成Gitlab pages页面

上传后，然后Gitlab服务器会自动检查.gitlab-ci.yml脚本是否有效，校验通过后，会自动开始执行脚本。

点击左侧菜单中的CI/CD->pipeline可以查看脚本的执行情况，当脚本的Stages状态变为test:passed&&deploy: passed时，说明构建完成。此时已可以访问我们的个人博客站点：https://username.gitlab.io 。有时构建完成时马上访问可能会出现404页面，这种情况很多人遇到过，这里是该问题的讨论：https://forum.gitlab.com/t/gitlab-pages-404-for-even-the-simplest-setup/5870 。其实这是因为Gitlab的服务器构建速度比较慢，等5-10分钟再重新访问页面就正常了。

## Gitlab pages博客站点绑定个人域名

此时我们要访问这个博客只能通过默认域名：https://username.gitlab.io 来访问，我们可以自己购买一个个人域名，然后将个人域名解析到默认域名上，这样就可以通过个人域名来访问这个博客。

域名购买可以在阿里云购买，购买后我们先点击控制台->域名->管理->免费开启ssl证书，申请ssl证书后，点击下载，选择其他，下载证书。然后打开我们的Gitlab pages仓库，选择左侧面板settings->pages，找到new domain。Domain项填入我们的个人域名，certificate和key分别复制我们下载下来的证书文件内容填入，然后点击create new domain。

随后提示：This domain is not verified. You will need to verify ownership before access is enabled. 我们再打开阿里云的域名控制台，选择->域名->解析来设置一个用来验证域名的DNS解析，具体要求可以看域名的detail页面中的verify ownership链接中的说明。

然后我们打开阿里云控制台->域名->域名解析，添加个人域名指向我们的Gitlab博客站点的默认域名:https://username.gitlab.io 的主机记录。添加一条主机记录，前缀www，记录类型选择CNAME，记录值填写默认域名：username.gitlab.io 。TTL最短10分钟，也就是10分钟后域名解析生效。生效后我们就可以使用个人域名来访问这个博客了。

# 多渠道同时部署hexo博客

## Gitlab Mirror:git push同时将博客代码push到gitlab和github(将push到Gitlab上源码也push到Github上作为备份)

我们还可以修改git push的配置，将代码同时push到gitlab和github上对应的仓库中。打开hexo博客源码目录/.git/config文件，找到下面的代码块：

```
[remote "origin"]
 url = git@gitlab.com:yourname/yourname.gitlab.io.git
 fetch = +refs/heads/*:refs/remotes/origin/*

```

在url这行下面加上新的一行其他远程库的路径，如：

```
 url = git@github.com:yourname/yourname.git

```

然后按照上面部署hexo博客的步骤push即可。注意push前先在github上添加你的ssh-key。

## 在Gitlab和Github上同时部署hexo博客(两个独立站点，内容完全一样)

我们还可以尝试同时在Gitlab和Github上部署hexo博客。在本地hexo博客源码目录鼠标右键选择Git bash here，然后运行下面的命令，安装用于部署hexo博客到Github上的插件：

```
npm install hexo-deployer-git --save
```

这是在Github上部署hexo博客时必须使用的插件。由于Gitlab上部署hexo博客采用CI方式自动部署，因此只在Gitlab上部署hexo博客时不需要安装这个插件。.gitlab-ci.yml文件中不需要加入这条命令。

在Github网站新建一个公开仓库，名为yourname.github.io，然后勾选Initialize this repository with a README，创建仓库，打开该仓库的settings，如果出现提示：Your site is published at https://zgcr.github.io/ ,则说明Github pages开启成功。

然后打开hexo博客源码目录下的_config.yml文件，修改相应代码块为以下内容：

```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy: 
  type: git
  repo: git@github.com:zgcr/zgcr.github.io.git
  branch: master
```

请把上面的repo内容换成你的Github pags仓库的git链接。完成后，使用下面命令：

```
hexo clean
hexo g
hexo d
```

此时可以正常将hexo博客部署到github上了。

要想将博客部署要gitlab上，只需按原有步骤使用下面命令：

```
git add -A
git commit -m "本次提交描述"
git push origin master
```

即可部署hexo博客至Gitlab上。

# 总结

使用Gitlab部署hexo博客时，我们不需要在本地使用hexo generate命令生成博客静态网页，再push到Gitlab仓库，而是直接push了hexo博客的源码到Gitlab仓库，同时增加一个.gitlab-ci.yml文件作为CI/CD脚本，通过该文件在Gitlab服务器生成博客的静态网页，然后自动发布到Gitlab博客站点上。

当我们要在博客上写新文章时，只需把Gitlab仓库中的源码pull下来，然后使用hexo新建文章，使用markdown编辑器(如typora)编辑文章，完成后将源码再push到Gitlab仓库中即可，Gitlab服务器会根据.gitlab-ci.yml文件重新生成博客的静态网页，然后自动发布到Gitlab博客站点上。我们可以点击CD/CI configuration让Gitlab服务器自动检测.gitlab-ci.yml文件，若文件正确则自动运行和发布博客。