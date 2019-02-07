---
title: hexo博客next主题6.0版本个性化设置
date: 2019-02-05 16:29:43
tags:
- 博客搭建
categories:
- 博客搭建
---



# hexo博客源码目录结构

```
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

本教程中大部分效果在修改配置文件后使用hexo g，hexo s命令，再在浏览器中访问:http://localhost:4000 即可看到。有少量效果需要将站点发布到线上后才能正常显示。

# 设置next主题/next主题scheme

next主题是hexo的一个著名的第三方主题，在hexo博客源码目录打开Git bash here，使用以下命令下载next主题：

```
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

使用git clone的方式下载主题后，如果博客是在Gitlab上使用CI方式自动部署，一定要删除hexo博客源码目录/themes/next目录下的.git目录，因为在.git目录下最好不要出现.git目录，这会造成推送到Gitlab上生成的博客页面一片空白，无法正常显示。

**注意：**

在Gitlab上使用CI方式自动部署hexo博客时，只有hexo博客源码目录下的.git目录保留，其他子目录中的.git目录一律都要删掉(比如git clone某个主题后该主题文件夹中就会有.git目录)!!!如果不删掉，会造成Gitlab线上博客能够正常访问，不出现404提示，但是整个博客页面一片空白，无法正常显示。

下载完成后，打开hexo源码根目录下的_config.yml文件，修改相应部分为以下内容：

```
# Extensions
## Plugins: https://hexo.io/plugins/
## Themes: https://hexo.io/themes/
theme: next
```

next主题目前支持四种scheme：

Muse:这是next主题的默认Scheme，黑白主调，大量留白。
Mist：Muse的紧凑版本，整洁有序的单栏外观。
Pisces：双栏Scheme，左侧网站信息及目录，块+片段结构布局。
Gemini：也是双栏Scheme，但右侧要更宽一些。

要修改next主题风格，只需打开hexo博客源码目录下\themes\next目录下的_config.yml文件，找到下面几行配置，将你想启用的scheme前面注释#去除即可(只能有一种scheme配置前面去掉#)。 

```
# Schemes
scheme: Muse
#scheme: Mist
#scheme: Pisces
#scheme: Gemini
```

# 设置站点标题/站点作者/站点语言/站点logo/侧边栏头像/是否加载动画/

要设置站点首页标题和站点作者，打开hexo博客源码目录下的_config.yml文件，找到下面内容：

```
# Site
title: Hexo
subtitle:
description:
keywords:
author: John Doe
language: zh-Hans
timezone:
```

其中title即为标题，修改该项即可。subtitle为副标题。author为站点作者，修改该项即可。description为站点描述，会显示在首页的标题下方。

要设置站点语言，首先在 Hexo博客源码根目录\themes\next\languages找是否有该语言文件，如zh-Hans.yml（中文简体)，然后在Hexo博客源码根目录\_config.yml 中设置 language为zh-Hans。即修改language项。注意如果上面的站点标题和作者中包含中文，请将_config.yml保存为UTF-8编码格式的TXT文档，否则博客上不能正常显示中文。遇到其他配置文件修改时包含中文后网站不能正常显示中文，也可以用这个方法解决。

要设置站点logo，即打开博客站点后在网页标签栏上显示的那个小图，打开hexo博客源码目录\themes\next目录下的_config.yml文件，找到下面内容：

```
favicon:
  small: /images/favicon-16x16-next.png
  medium: /images/favicon-32x32-next.png
  apple_touch_icon: /images/apple-touch-icon-next.png
  safari_pinned_tab: /images/logo.svg
  #android_manifest: /images/manifest.json
  #ms_browserconfig: /images/browserconfig.xml
```

我们可以看到logo可以看到有四种效果，一般我们只需将medium换成我们自己的图标路径就行了。建议把自定义logo放在hexo博客源码目录\themes\next\source\images目录下。logo大小与默认logo大小相同。

要设置首页的侧边栏头像，打开hexo博客源码目录\themes\next目录下的_config.yml文件，找到下面内容：

```
# Sidebar Avatar
# in theme directory(source/images): /images/avatar.gif
# in site  directory(source/uploads): /uploads/avatar.gif
avatar: /images/avatar.gif
```

其中avatar: /images/avatar.gif即为侧边栏头像，头像图片放在Hexo博客源码目录\themes\next\source\images\avatar.gif。只要把我们的自定义头像放在这个目录下，然后将上面配置路径做相应修改即可。图片支持jpg、png、gif等格式。当然你也可以用网上的图片，路径改成http URL即可。

要打开或关闭加载动画，在hexo博客源码目录下\themes\next目录下的_config.yml文件中找到下面内容，将enable项设为true(打开)或false(关闭)即可。

```
# Use velocity to animate everything.
motion:
  enable: true
  async: false
  transition:
    # Transition variants:
    # fadeIn | fadeOut | flipXIn | flipXOut | flipYIn | flipYOut | flipBounceXIn | flipBounceXOut | flipBounceYIn | flipBounceYOut
    # swoopIn | swoopOut | whirlIn | whirlOut | shrinkIn | shrinkOut | expandIn | expandOut
    # bounceIn | bounceOut | bounceUpIn | bounceUpOut | bounceDownIn | bounceDownOut | bounceLeftIn | bounceLeftOut | bounceRightIn | bounceRightOut
    # slideUpIn | slideUpOut | slideDownIn | slideDownOut | slideLeftIn | slideLeftOut | slideRightIn | slideRightOut
    # slideUpBigIn | slideUpBigOut | slideDownBigIn | slideDownBigOut | slideLeftBigIn | slideLeftBigOut | slideRightBigIn | slideRightBigOut
    # perspectiveUpIn | perspectiveUpOut | perspectiveDownIn | perspectiveDownOut | perspectiveLeftIn | perspectiveLeftOut | perspectiveRightIn | perspectiveRightOut
    post_block: fadeIn
    post_header: slideDownIn
    post_body: slideDownIn
    coll_header: slideLeftIn
    # Only for Pisces | Gemini.
    sidebar: slideUpIn
```

# 设置侧边栏头像圆形和旋转效果/侧边栏位置和侧边栏显示时机/侧边栏推荐站点(如友站)/修改标题栏背景颜色

打开hexo博客源码目录\themes\next\source\css\_common\components\sidebar\sidebar-author.styl文件，找到.site-author-image代码块，替换成以下内容：

```
.site-author-image {
  display: block;
  margin: 0 auto;
  padding: $site-author-image-padding;
  max-width: $site-author-image-width;
  height: $site-author-image-height;
  border: $site-author-image-border-width solid $site-author-image-border-color;
  border-radius: 60%;
  transition: 2.5s all;  
}

.site-author-image:hover {
    transform: rotate(360deg);
}
```

默认情况下，侧边栏仅在文章页面（拥有目录列表）时才显示，并放置于右侧位置。我们打开hexo博客源码目录/themes/next/下的_config.yml文件：

```

sidebar:
  # Sidebar Position, available value: left | right (only for Pisces | Gemini).
  position: left
  #position: right

  # Sidebar Display, available value (only for Muse | Mist):
  #  - post    expand on posts automatically. Default.
  #  - always  expand for all pages automatically
  #  - hide    expand only when click on the sidebar toggle icon.
  #  - remove  Totally remove sidebar including sidebar toggle.
  #display: post
  display: always
  #display: hide
  #display: remove
```

上面position中可选left和right(只有Pisces | Gemini两种scheme中才生效)，下面display中有四个选项，选择你想设置的那项，去掉前面的注释即可。

要在侧边栏显示推荐站点，打开hexo博客源码目录下\themes\next目录下的_config.yml文件，找到下列代码：

```
# Blog rolls
links_icon: link
links_title: Links
links_layout: block
#links_layout: inline
#links:
  #Title: http://example.com/
```

将#links:前面的#注册去掉就会显示推荐站点了，同时修改下面的#Title: http://example.com/ ，去掉#注释，比如修改为百度: http://www.baidu.com/ 。links_title项是下面这些链接的说明，比如你链接的如果都是友站，可以写links_title: 友站。

使用Pisces或Gemini主题时，网站标题栏背景颜色是黑色的，我们可以在hexo博客源码目录/themes/next/source/css/_custom目录下的custom.styl文件中添加下面的代码：

```
.site-meta {
  background: $blue; //修改为自己喜欢的颜色
}
```

# 设置显示的菜单项/创建关于页/标签页/分类页

打开hexo博客源码目录下\themes\next目录下的_config.yml文件，找到下面的内容：

```
menu:
  home: / || home # 首页
  #about: /about/ || user # 关于
  #tags: /tags/ || tags # 标签
  #categories: /categories/ || th # 分类
  archives: /archives/ || archive # 归档
  #schedule: /schedule/ || calendar # 日程表
  #sitemap: /sitemap.xml || sitemap # 站点地图
  #commonweal: /404/ || heartbeat # 公益404
```

想显示哪一项菜单，就去掉对应菜单项的#注释。如果对页面的相关简体中文翻译不满意，可以打开hexo博客源码目录/themes/next/languages/zh-Hans.yml，对相关的翻译内容进行修改。你还可以按照上面的格式自己创建一些菜单项，以music为例，在上面的menu字段中添加一项：

```
music: /music/ || music
# 标签名 相应的文件夹名 网站上对应的图标名
```

然后在相关翻译文件zh-Hans.yml中的menu字段中也添加一项对应的翻译。

```
menu:
  home: 首页
  archives: 归档
  categories: 分类
  tags: 标签
  about: 关于
  music: 音乐
  search: 搜索
  schedule: 日程表
  sitemap: 站点地图
  commonweal: 公益404
```

注意上面about、tags、categories、schedule、sitemap、commonweal页都需要另外设置后才可正常访问。

要建立关于页/标签页/分类页，只需使用下面的命令即可：

```
hexo new page about
hexo new page tags
hexo new page categories
```

然后修改hexo博客源码目录/source/目录中about/tages/categories目录中的index.md文件，新增加一行type属性：

```
type: about/tags/categories 
```

# 设置个人社交图标和链接

打开hexo博客源码目录下\themes\next目录下的_config.yml文件，找到下面的内容：

```

#social:
  #GitHub: https://github.com/yourname || github
  #E-Mail: mailto:yourname@gmail.com || envelope
  #Google: https://plus.google.com/yourname || google
  #Twitter: https://twitter.com/yourname || twitter
  #FB Page: https://www.facebook.com/yourname || facebook
  #VK Group: https://vk.com/yourname || vk
  #StackOverflow: https://stackoverflow.com/yourname || stack-overflow
  #YouTube: https://youtube.com/yourname || youtube
  #Instagram: https://instagram.com/yourname || instagram
  #Skype: skype:yourname?call|chat || skype

social_icons:
  enable: true
  icons_only: false
  transition: false

```

想要在侧边栏显示社交图标，就将social前面的注释#去掉，要显示哪几种社交图标，就将对于的社交图标项前面的#去掉，并修改后面的链接为你个人的链接。你还可以自己添加一些自定义社交图标项。格式仿照上面的格式即可。||之后是在图标库中对应的图标，对于自定义社交图标项，在图标库中找到你想设定的图标的名称，填在||之后即可。图标库链接：http://fontawesome.io/icons/ 。

比如我要添加微信和CSDN图标：

```
微信: https://www.yourname.com/about/ || weixin
CSDN: https://blog.csdn.net/yourname || copyright 
```

上面的链接可以填你的博客站点的about页URL和CSDN博客首页的URL。

# 设置站点左上角或者右上角的fork me on github

在：http://tholman.com/github-corners/ 或：https://github.com/blog/273-github-ribbons 选择合适的样式，复制其代码到hexo博客源码目录/themes/next/layout/_layout.swig，添加到div class="headband"下面，如：

```
<a href="https://your-url" class="github-corner" aria-label="View source on GitHub"><svg width="100" height="100" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a><style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>
```

注意把a href="https://github.com/yourname"中的URL改成你的github主页URL。

# 设置hexo博客背景图片/浏览页面时显示浏览进度

要设置博客的背景图片，打开hexo博客源码目录\themes\next\source\css\ _custom\custom.styl文件，添加以下代码：

```
body{
    background:url(/images/background1.jpg);
    background-size:cover;
    background-repeat:no-repeat;
    background-attachment:fixed;
    background-position:center;
}

.main-inner { 
    background: #fff;
    opacity: 0.9;
}
```

其中url可更换为自己喜欢的图片的地址。如果考虑到网上的图片加载可能会比较慢，我们可以把图片放到本地/images/文件夹内，然后url设定为本地图片的路径。repeat即是否重复出现；attachment：定义背景图片随滚动轴的移动方式；position：设置背景图像的起始位置；opacity: 0.9为不透明度。

想在浏览页面时显示浏览进度，打开hexo博客源码目录/themes/next/_config.yml文件，找到下列代码：

```
  # Back to top in sidebar (only for Pisces | Gemini).
  b2t: false

  # Scroll percent label in b2t button.
  scrollpercent: false
```

将scrollpercent改为true即可。如果想把 top按钮放在侧边栏上，则把b2t也改为true。

# 设置站点动态背景/设置站点顶部滚动加载条

目前next主题为6.0版本，这个版本中有4种动态背景：Canvas-nest、three_waves、canvas_lines、canvas_sphere。

打开hexo博客源码目录themes/next/_config.yml文件，从中找到以下内容：

```
# Canvas-nest
canvas_nest: false

# three_waves
three_waves: false

# canvas_lines
canvas_lines: false

# canvas_sphere
canvas_sphere: false
```

想设置哪一种动态背景，设置里需要的动态背景为true即可。

要设置站点顶部动态加载条，打开hexo博客源码目录\themes\next目录下_config.yml文件，找到下面的内容:

```
# Progress bar in the top during page loading.
pace: false
# Themes list:
#pace-theme-big-counter
#pace-theme-bounce
#pace-theme-barber-shop
#pace-theme-center-atom
#pace-theme-center-circle
#pace-theme-center-radar
#pace-theme-center-simple
#pace-theme-corner-indicator
#pace-theme-fill-left
#pace-theme-flash
#pace-theme-loading-bar
#pace-theme-mac-osx
#pace-theme-minimal
# For example
# pace_theme: pace-theme-center-simple
pace_theme: pace-theme-minimal
```

将pace项修改为true即启用加载条。

# 博客底部显示完整时间/隐藏powered By Hexo/显示桃心/显示总访问量/显示网站运行时间

打开hexo博客源码目录/themes/next/layout/_partials/footer.swig，找到下列代码：

```
<div class="copyright">{#
#}{% set current = date(Date.now(), "YYYY") %}{#
#}&copy; {% if theme.footer.since and theme.footer.since != current %}{{ theme.footer.since }} &mdash; {% endif %}{#
#}<span itemprop="copyrightYear">{{ current }}</span>
  <span class="with-love">
    <i class="fa fa-{{ theme.footer.icon }}"></i>
  </span>
  <span class="author" itemprop="copyrightHolder">{{ theme.footer.copyright || config.author }}</span>
```

把"YYYY"改为"YYYY-MM-DD"即可。

要隐藏网页底部显示的powered By Hexo / 强力驱动，打开hexo博客源码目录\themes\next目录下的_config.yml文件，找到下列代码块，删除。

```
{% if theme.footer.powered %}
  <div class="powered-by">{#
  #}{{ __('footer.powered', '<a class="theme-link" target="_blank" href="https://hexo.io">Hexo</a>') }}{#
#}</div>
{% endif %}


{% if theme.footer.powered and theme.footer.theme.enable %}
  <span class="post-meta-divider">|</span>
{% endif %}

{% if theme.footer.theme.enable %}
  <div class="theme-info">{#
  #}{{ __('footer.theme') }} &mdash; {#
  #}<a class="theme-link" target="_blank" href="https://github.com/iissnan/hexo-theme-next">{#
    #}NexT.{{ theme.scheme }}{#
  #}</a>{% if theme.footer.theme.version %} v{{ theme.version }}{% endif %}{#
#}</div>
{% endif %}
```

要将博客底部改为桃心，还是在hexo博客源码目录/themes/next/layout/_partials/footer.swig文件中，找到下列代码块：

```
  <span class="with-love">
    <i class="fa fa-{{ theme.footer.icon }}"></i>
  </span>
```

在图标库：http://fontawesome.io/icons/ 中选择合适的图标，记住图标名称，如我们选择图标heart，那么将上面的代码块修改成下面的代码即可。

```
<span class="with-love" id="animate">
    <i class="fa fa-heart"></i>
  </span>
```

要在博客底部显示总访问量，打开hexo博客源码目录/themes/next目录下的_config.yml文件，找到下列代码块：

```
busuanzi_count:
  # count values only if the other configs are false
  enable: false
  # custom uv span for the whole site
  site_uv: true
  site_uv_header: <i class="fa fa-user"></i>
  site_uv_footer:
  # custom pv span for the whole site
  site_pv: true
  site_pv_header: <i class="fa fa-eye"></i>
  site_pv_footer:
  # custom pv span for one page only
  page_pv: true
  page_pv_header: <i class="fa fa-file-o"></i>
  page_pv_footer:
```
将上面代码修改为：

```
busuanzi_count:
  # count values only if the other configs are false
  enable: true
  # custom uv span for the whole site
  site_uv: true  # 整个网站的访客数
  site_uv_header: 访客数
  site_uv_footer: 人
  # custom pv span for the whole site
  site_pv: true # 整个网站的访问量
  site_pv_header: 本站总访问量
  site_pv_footer: 次
  # custom pv span for one page only
  page_pv: true # 每个页面的访问量
  page_pv_header: <i class="fa fa-file-o"></i> 阅读数
  page_pv_footer:
```

然后打开hexo博客源码目录\themes\next\layout\_third-party\analytics\busuanzi-counter.swig文件，找到下面的代码块：

```
{% if theme.busuanzi_count.enable %}
<div class="busuanzi-count">
  <script async src="https://dn-lbstatics.qbox.me/busuanzi/2.3/busuanzi.pure.mini.js"></script>
```

将上面的代码块修改为下面的代码即可。注意hexo s命令运行时的次数和人数统计不正常，不要紧，将网站发布到线上后就正常了。

```
{% if theme.busuanzi_count.enable %}
<div class="busuanzi-count">
  <script async src="https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
```

要在博客底部添加网站运行时间，打开hexo博客源码目录/themes/next/layout/_partials/footer.swig文件，添加下列代码，注意以UTF-8编码格式保存：

```
<!-- 在网页底部添加网站运行时间 -->
<span id="timeDate">载入天数...</span><span id="times">载入时分秒...</span>
<script>
    var now = new Date();
    function createtime() {
        var grt= new Date("02/01/2019 00:00:00");//此处修改你的建站时间或者网站上线时间
        now.setTime(now.getTime()+250);
        days = (now - grt ) / 1000 / 60 / 60 / 24; dnum = Math.floor(days);
        hours = (now - grt ) / 1000 / 60 / 60 - (24 * dnum); hnum = Math.floor(hours);
        if(String(hnum).length ==1 ){hnum = "0" + hnum;} minutes = (now - grt ) / 1000 /60 - (24 * 60 * dnum) - (60 * hnum);
        mnum = Math.floor(minutes); if(String(mnum).length ==1 ){mnum = "0" + mnum;}
        seconds = (now - grt ) / 1000 - (24 * 60 * 60 * dnum) - (60 * 60 * hnum) - (60 * mnum);
        snum = Math.round(seconds); if(String(snum).length ==1 ){snum = "0" + snum;}
        document.getElementById("timeDate").innerHTML = "网站已运行 "+dnum+" 天 ";
        document.getElementById("times").innerHTML = hnum + " 小时 " + mnum + " 分 " + snum + " 秒";
    }
setInterval("createtime()",250);
</script>
```

注意初始建站时间是你手动设置的。

# hexo创建和编辑文章

创建一篇新文章：

```
hexo new "文章名"
```

文章创建后，想要编辑文章，则使用markdown编辑器(如typora)打开hexo博客源码目录\source\_posts目录下的同名.md文件：

```
title: 测试文章1
date: 2019-02-02 23:45:50
tags:
- tag1
- tag2
categories:
- 类1
- 子类1
```

上面的代码块设置了文章的标题、日期、所属标签、所属分类，这个代码块之下就可以写文章的正文了。

# 开启hexo博客Latex公式支持

首先更换hexo的markdown渲染引擎，使用下面的命令：

```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

hexo-renderer-kramed插件的Github主页在这里：https://github.com/sun11/hexo-renderer-kramed 。这个插件fork了 hexo-renderer-marked项目，并且只针对MathJax支持进行了改进。如果使用Gitlab CI方式自动部署hexo博客，请将上面两个命令加入到.gitlab-ci.yml脚本中。

然后打开hexo博客源码目录/themes/next目录下的_config.yml文件，找到下列代码：

```
# MathJax Support
mathjax:
  enable: false
  per_page: false
  cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
```

将enable改为true。此时hexo博客渲染时只有文件头中含mathjax: true这项时才对这篇文章进行mathjax渲染，这可以提高博客的加载速度。如果想让每个页面都自动开启mathjax渲染，则把per_page设为true。

然后试了一下几个例子，发现Latex公式只能简单地正确显示几秒中，然后就变成了一堆不知道显示什么的小方块。这是上面代码块中的cdn的问题，将上面的cdn那一行换成下面的cdn即可。

```
cdn: //cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML
```

特别要注意的一点是，按上述方法必须要前后都用两个$$符号的形式将Latex公式括起来才可以正常显示。

# 修改hexo博客next主题文章字体大小

打开hexo博客源码目录\themes\next\source\css\_variables目录下的custom.styl文件，添加下列代码：

```
// 标题，修改成你期望的字体族
$font-family-headings = Georgia, sans

// 修改成你期望的字体族
$font-family-base = "Microsoft YaHei", Verdana, sans-serif

// 代码字体
$code-font-family = "Input Mono", "PT Mono", Consolas, Monaco, Menlo, monospace

// 正文字体的大小
$font-size-base = 16px

// 代码字体的大小
$code-font-size = 13px
```

修改font-size-base后面的数值大小即可。

# 设置文章代码高亮主题

NexT使用Tomorrow Theme作为代码高亮，共有5款主题供你选择。NexT默认使用的是 白色的normal主题，可选的值有 normal，night，night blue，night bright，night eighties。

只需打开hexo博客源码目录/themes/next目录下的_config.yml文件，修改其highlight_theme项值即可。

# hexo博客显示自带图片

一般情况下，在hexo博客中要显示图片，我们建议将图片上传到某个在线图床中，然后引用图片的URL即可，不建议将图片打包到网站的source文件夹中，这是因为图片一般都比较大， 且后面博客文章越来越多时，使用这种方式会使整个网站源码文件变得很大，不方便上传。

但是有少量的图片仍然可以采用自带图片的形式，比如我们现在要在about页中插入微信二维码，方便读者扫码加站点作者的微信。

首先在hexo博客源码目录下运行Git bash here，运行下列命令，注意如果采用Gitlab CI方式自动部署hexo博客，该命令还要加入到.gitlab-ci.yml脚本中。

```
npm install hexo-asset-image --save
```

然后打开hexo博客源码目录\themes\next目录下_config.yml文件，修改相应部分为如下代码：

```
post_asset_folder: true
```

现在我们要在about页中插入自带图片，首先用命令hexo new page about新建一个about页，然后打开hexo博客源码目录\source\about目录，在该目录下新建一个和index.md文件同名文件夹index，将要显示在index.md文章中的图片都放在这个文件夹中。然后打开index.md文件中添加：

```
![你想输入的替代文字](index/图片名.jpg)
```

这样该图片就会在这篇文章中显示了。

# 首页设置文章预览(阅读全文)/文章末尾添加版权声明/修改文章内链接样式/文章末尾添加"本文结束"标记

打开hexo博客源码目录\themes\next目录下的_config.yml文件，查找以下代码：

```
auto_excerpt:
  enable: false
  length: 150
```

将enable设为true，length为想要预览的文章字数。建议改为100字。

要在文章末尾添加版权声明，还是在上面的_config.yml文件中查找以下代码：

```
post_copyright:
  enable: false
  license: CC BY-NC-SA 3.0
  license_url: https://creativecommons.org/licenses/by-nc-sa/3.0/
```

将enable改为true即可。

修改文章中的链接样式，修改hexo博客源码目录\themes\next\source\css\_common\components\post\post.styl文件，在末尾添加如下代码即可：

```
// 文章内链接文本样式
.post-body p a{
  color: #0593d3;
  border-bottom: none;
  border-bottom: 1px solid #0593d3;
  &:hover {
    color: #fc6423;
    border-bottom: none;
    border-bottom: 1px solid #fc6423;
  }
}
```

要在文章末尾添加"本文结束"标记，在hexo博客源码目录\themes\next\layout\_macro中新建passage-end-tag.swig文件,并添加以下内容，使用UTF-8编码格式保存(否则中文不能正常显示)：

```
<div>
    {% if not is_index %}
        <div style="text-align:center;color: #ccc;font-size:14px;">----------------本文结束<i class="fa fa-heart"></i>感谢阅读----------------</div>
    {% endif %}
</div>
```

然后打开hexo博客源码目录\themes\next\layout\_macro\post.swig文件，在post-body 之后， post-footer上面的两个DIV之上，即if (theme.alipay or theme.wechatpay or theme.bitcoin) and not is_index之前添加如下代码：

```
<div>
  {% if not is_index %}
    {% include 'passage-end-tag.swig' %}
  {% endif %}
</div>
```

然后打开hexo博客源码目录\themes\next目录下的_config.yml，在末尾添加以下代码，就大功告成了。

```
# 文章末尾添加“本文结束”标记
passage_end_tag:
  enabled: true
```

# 博文加密/博文置顶/文章首部添加置顶标志/文章底部打赏/显示文章字数统计、阅读时长/文章顶部显示更新时间

想加密某篇博文，首先打开hexo博客源码目录\themes\next\layout\_partials\head.swig文件，在meta name="theme-color"下第三行插入以下代码，注意这个文件不要用UTF-8编码格式保存。

```
<script>
    (function(){
        if('{{ page.password }}'){
            if (prompt('please enter password') !== '{{ page.password }}'){
                alert('wrong password!');
                history.back();
            }
        }
    })();
</script>
```

如果某篇文章需要加密，则在文章的head头加上一行password: 123456(123456就是密码)，如：

```
title: 测试文章1
date: 2019-02-02 23:45:50
tags:
- tag2
categories:
- 类1
- 子类1
password: 123456
top: 50
```

想让某篇博文置顶，现在已经有修改后支持置顶的仓库，可以打开Git bash here，直接用以下命令安装：

```
npm uninstall hexo-generator-index --save
npm install hexo-generator-index-pin-top --save
```

注意如果是在Gitlab用CI脚本自动部署hexo博客，则上面两行还要加入到.gitlab-ci.yml文件中。

然后在需要置顶的文章的文件头中加入一项top: 数值即可。数值可以是任意大于等于0的数字。置顶排序时会按数值从大到小的顺序排序。

我们还可以在文章首部添加一个置顶标志。打开hexo博客源码目录/themes/next/layout/_macro目录下的post.swig文件，定位到div class="post-meta"，在其后插入如下代码即可，注意保存时要使用UTF-8编码格式保存：

```
          {% if post.top %}
            <i class="fa fa-thumb-tack"></i>
            <font color=7D26CD>置顶</font>
            <span class="post-meta-divider">|</span>
          {% endif %}
```

要开启文章底部打赏，首先准备好微信和支付宝二维码图片。然后打开hexo博客源码目录\themes\next目录下的_config.yml文件，找到下列代码：

```
# Reward
#reward_comment: Donate comment here
#wechatpay: /images/wechatpay.jpg
#alipay: /images/alipay.jpg
#bitcoin: /images/bitcoin.png
```

如果只打开微信和支付宝打赏，那么把3、4行的注释去掉，然后把wechatpay和alipay后面的路径换成图片的路径，建议先把图片放到themes\next\source\images目录下。

要显示文章字数统计和阅读时长，只需打开hexo博客源码目录\themes\next目录下的_config.yml文件，找到下列代码：

```
# Post wordcount display settings
# Dependencies: https://github.com/willin/hexo-wordcount
post_wordcount:
  item_text: true
  wordcount: false
  min2read: false
  totalcount: false
  separated_meta: true
```

将wordcount、min2read设为true即可。此时部署后会发现字数统计和阅读时长后面没有对应的xxx字，xx分钟等字样。我们再打开hexo博客源码目录\themes\next\layout\_macro\post.swig 文件，找到相应部分修改成下面的代码：

```
                <span title="{{ __('post.wordcount') }}">
                  {{ wordcount(post.content) }} 字
                </span>
```

```
                <span title="{{ __('post.min2read') }}">
                  {{ min2read(post.content) }} 分钟
                </span>
```

注意保存时要以UTF-8编码格式保存。

此时我们还发现统计的字数没有显示，这是因为没有安装 hexo-wordcount 插件，使用下面的命令安装即可：

```
npm i --save hexo-wordcount
```

注意如果在Gitlab上使用CI自动部署hexo博客，则上面的命令也要写入.gitlab-ci.yml文件中。

要在文章顶部显示更新时间，打开hexo博客源码目录\themes\next目录下的 _config.yml 文件，找到下列代码块：

```
# Post meta display settings
post_meta:
  item_text: true
  created_at: true
  updated_at: false
  categories: true
```

把updated_at设为true即可。

# hexo博客添加valine评论系统

Valine是一款极简的评论系统。它的特点是：无后端实现；使用国内后端云服务提供商LeanCloud提供的存储服务；支持表情；支持邮件通知；支持验证码；支持 Markdown格式；支持匿名评论，无需注册和登录账号。

首先在leancloud官网注册一个账号。官网地址: https://leancloud.cn/ 。新账号要先验证邮箱和手机号。然后点击创建应用，自己命名，选择开发版。选择刚创建好的应用，选择左侧菜单中的设置->应用key，保存App ID和App key，待会儿要用到。然后选择左侧菜单中的安全域名，填写你的博客的域名。

打开hexo博客源码目录，使用下面的命令安装valine插件：

```
npm install valine --save
```

如果使用gitlab CI方式部署hexo博客，还要把上面这条命令加入.gitlab-ci.yml文件中。

然后打开hexo博客源码目录/themes/next目录下的_config.yml文件，找到下列代码：

```
valine:
  enable: false
  appid:  # your leancloud application appid
  appkey:  # your leancloud application appkey
  notify: false # mail notifier , https://github.com/xCss/Valine/wiki
  verify: false # Verification code
  placeholder: Just go go # comment box placeholder
  avatar: mm # gravatar style
  guest_info: nick,mail,link # custom comment header
  pageSize: 10 # pagination size
```

修改enable为true，修改appid和appkey为上面保存的App ID和App key。这样博客中的valine评论系统就开启了。

评论的管理在leancloud官网，进入应用项目，选择左侧菜单存储->comment，就可以管理评论了。

# hexo博客添加need more share2分享
这个项目在Github上的仓库地址：https://github.com/revir/need-more-share2 。

我们的next主题中已经集成了这个项目。打开hexo博客源码目录/themes/next目录下的_config.yml文件，找到相应部分代码，修改为以下代码：

```
needmoreshare2:
  enable: true
  postbottom:
    enable: true
    options:
      iconStyle: default
      boxForm: horizontal
      position: bottomCenter
      networks: Weibo,Wechat,Douban,Evernote,Facebook,Twitter
  float:
    enable: true
    options:
      iconStyle: default
      boxForm: horizontal
      position: topRight
      networks: Weibo,Wechat,Douban,Evernote,Facebook,Twitter
```

注意从上到下三个enable分别是开启need more share2，开启底部的need more share2按钮，开启左侧悬浮的need more share2按钮。

由于目前微信二维码不能正确加载，因此我们按照这个issue中最后仓库作者回复的方法修改一下:https://github.com/revir/need-more-share2/issues/4 。

打开hexo博客源码目录\themes\next\source\lib\needsharebutton\needsharebutton.js文件，找到下面的代码：

```
var imgSrc = "https://api.qinco.me/api/qr?size=400&content=" + encodeURIComponent(myoptions.url);
```

修改为下面的代码即可。

```
var imgSrc = 'https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='+encodeURIComponent(myoptions.url);
```

# Pisces Scheme修改内容板块的宽度

打开hexo博客源码目录/themes/next/source/css/_schemes/Picses/_layout.styl文件，在末尾添加下列代码：

```
// 以下为新增代码！！
header{ width: 85% !important; }
header.post-header {
  width: auto !important;
}
.container .main-inner { width: 85%; }
.content-wrap { width: calc(100% - 260px); }

.header {
  +tablet() {
    width: auto !important;
  }
  +mobile() {
    width: auto !important;
  }
}

.container .main-inner {
  +tablet() {
    width: auto !important;
  }
  +mobile() {
    width: auto !important;
  }
}

.content-wrap {
  +tablet() {
    width: 100% !important;
  }
  +mobile() {
    width: 100% !important;
  }
}
```

要调整宽度，只需修改两个75%参数即可。

# Pisces Scheme修改内容板块/菜单栏/站点概况背景颜色和透明度

我们在上面设置背景图片时打开hexo博客源码目录\themes\next\source\css\ _custom\custom.styl文件，添加了以下代码：

```
.main-inner { 
    background: #fff;
    opacity: 0.9;
}
```

这可以使除了菜单栏以外的其他板块都应用其不透明度：0.9的设置。

要修改菜单栏背景颜色和透明度，打开hexo博客源码目录\themes\next\source\css\_schemes\Pisces\_layout.styl文件，找到下列代码块：

```
.header-inner {
  position: absolute;
  top: 0;
  overflow: hidden;
  padding: 0;
  width: 240px;
  background: white;
  box-shadow: $box-shadow-inner;
  border-radius: $border-radius-inner;
```

将background项改为background: rgba(255,255,255,0.9); 0.9是透明度。

# 开启博客站点内部搜索

在hexo博客源码目录，运行Git bash here，运行下面的命令：

```
npm install hexo-generator-searchdb --save
```

注意如果在Gitlab上使用CI自动部署hexo博客，则上面的命令也要写入.gitlab-ci.yml文件中。

打开hexo博客源码目录中的_config.yml文件，在末尾下添加下列代码：

```
search:
  path: search.xml
  field: post
  format: html
  limit: 10000
```

然后打开hexo博客源码目录/themes/next目录下的_config.yml文件，找到下列代码：

```
# Local search
# Dependencies: https://github.com/flashlab/hexo-generator-search
local_search:
  enable: false
  # if auto, trigger search by changing input
  # if manual, trigger search by pressing enter key or search button
  trigger: auto
  # show top n results per article, show all results by setting to -1
  top_n_per_article: 1
```

将enable改为true即可。

# 部署hexo博客
要注意一点的是，我们在上面操作中所有涉及npm安装的操作必须加入到.gitlab-ci.yml文件的script:项中，这样在Gitlab线上才能正确生成博客。

在hexo博客源码目录点击鼠标右键Git bash here，依次运行下列命令：

```
git init
git add -A
git commit -m "init blog"
git remote add origin git@gitlab.com:username/username.gitlab.io.git
git push -u origin master
```

最后一步如果失败，试试先使用命令：

```
git pull origin master --allow-unrelated-histories
```

然后重新进行最后一步。

这样我们就将hexo博客的源码推送到Gitlab pages仓库上了。然后Gitlab服务器会自动检查.gitlab-ci.yml脚本是否有效，校验通过后，会自动开始执行脚本。过5-10分钟执行完成后，我们就可以看到线上的博客更新了。

# Mirror:git push同时将代码push到gitlab和github

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