---
title: MySQL基础知识介绍
date: 2019-06-21 20:32:11
tags:
- 数据库
categories:
- 数据库
---

# MySQL5.7使用命令行登陆
首先将MySQL的安装文件夹\bin路径添加到环境变量中，否则在cmd.exe中我们无法打开mysql。
## 本地登录MySQL数据库
```
mysql -u root -p   
//root是用户名，输入这条命令按回车键后系统会提示你输入密码
//然后你输入正确的密码，再按一下回车键就可以登录MySQL数据库了
```
## 指定端口号登录MySQL数据库
```
mysql -u root -p  -P 3306  
//注意指定端口的字母P为大写，而标识密码的p为小写。MySQL默认端口号为3306
```
## 指定IP地址和端口号登录MySQL数据库
```
mysql -h ip -u root -p -P 3306 
//ip即这个root账号指定的ip地址，只有从该ip地址登陆root账号才能登陆，3306为端口号
//如：mysql -h 127.0.0.1 -u root -p -P 3306
```
# MySQL5.7基本SQL语句
## 基本名词和知识
实例：instance；数据库：database；模式：schema；表：table；视图：view；索引：index。
一个关系数据库管理系统的实例（Instance）中可以建立多个数据库（database）；一个数据库中可以建立多个模式（schema）；一个模式下通常包括多个表（table）、视图（view）和索引（index）等数据库对象。
但是在MySQL中，schema和database是同义词。CREATE SCHEMA和CREATE DATABASE是等效的。在其他的数据库中有所不同。在oracle数据库产品中,schema是database的一部分。
## 启动和停止Mysql服务
```
//cmd.exe中输入
net start 你的mysql服务名 //回车即可启动Mysql服务
net stop 你的mysql服务名 //回车即可关闭Mysql服务
```
## 用户管理
```
create user '用户名'@'localhost'identified by'密码'; //新建用户
//'用户名'@'localhost'意思即为用户名@主机地址，表示该用户名的账号可以在哪些主机地址上登录
//mysql的用户都必须先进行授权，localhost即代表127.0.0.1默认本地地址，如果用%则意味着“任何主机”，说明这个账号可以在所有主机上登录。在对用户账号进行操作时如果只用用户名，那么默认都是用户名@%。

set password for '用户名'@'localhost' ='密码'; //更改密码

show grants for '用户名'@'主机地址'; //查看该用户名的权限
grant all privileges on *.* to 'name'@'localhost'; //给位于localhost地址上的name用户所有数据库的所有权限
grant all privileges on test.* to 'name'@'localhost'; //给位于localhost地址上的name用户名为test的数据库的所有权限
//privileges可以省略，all也可以改为select，update等具体权限

flush privileges; //每当调整权限后，需要执行该语句刷新权限

revoke all privileges on test.* from 'zgcr'@'localhost'; //revoke是grant的反操作，去除权限
```
## 数据库查看、创建、删除
```
show databases; //查看数据库
create database if not exists test2; //创建数据库,test2为数据库名
use test1; //使用名为test1的数据库
drop database test1; //删除名为test1的数据库
```
## 创建/删除表、列、索引、视图
```
create table tabname(col1 type1 [not null] [primary key],col2 type2 [not null],..) //直接创建新表
//如在cos数据库中创建Patron表：
use cos;
create table Patron(
   Patron_id INT NOT NULL AUTO_INCREMENT,
   password VARCHAR(100) NOT NULL,
   address VARCHAR(50) NOT NULL,
   phone_number VARCHAR(50),
   PRIMARY KEY ( Patron_id) 
)ENGINE=InnoDB DEFAULT CHARSET=utf8;
//如果你不想字段为NULL可以设置字段的属性为NOT NULL，在操作数据库时如果输入该字段的数据为NULL ，就会报错。
//AUTO_INCREMENT定义列为自增的属性，一般用于主键，数值会自动加1。
//PRIMARY KEY关键字用于定义列为主键。您可以使用多列来定义主键，列间以逗号分隔。
//ENGINE 设置存储引擎，CHARSET 设置编码。

use cos;//使用名为cos的数据库
create table Patron2 like Patron;//根据旧表创建新表。第一种方法。Patron2为新表名，Patron为旧表名
use cos; //使用名为cos的数据库
create table Patron3 select * from Patron;//根据旧表创建新表。第二种方法。Patron3为新表名，Patron为旧表名
drop table Patron3; //删除表。Patron3为表名

alter table test_table add phone_number varchar(15) NOT NULL; //增加一个列。test_table为表名，phone_number为要增加的列名，后面是属性

use test; //使用名为cos的数据库
create unique index index1 on test_table(test_id); //创建索引。unique表示唯一索引，index1为索引名，test_table为表名，test_id为要创建索引的列名
use test; //使用名为cos的数据库
drop index index1 on test_table; //删除索引。index1索引名，test_table为索引所在的表名。索引是不可更改的，想更改必须删除重新建。

create view view1 as select test_id from test_table; //创建视图，view1为创建的视图名，test_id为列明，test_table为列所在的表名
drop view view1; //删除view1，view1为列名
```
## 其他基本SQL语句
```
选择：select * from table1 where 范围
插入：insert into table1(field1,field2) values(value1,value2)
删除：delete from table1 where 范围
更新：update table1 set field1=value1 where 范围
查找：select * from table1 where field1 like ’%value1%’ （所有包含‘value1’这个模式的字符串）
排序：select * from table1 order by field1,field2 [desc]
总数：select count(*) as totalcount from table1
求和：select sum(field1) as sumvalue from table1
平均：select avg(field1) as avgvalue from table1
最大：select max(field1) as maxvalue from table1
最小：select min(field1) as minvalue from table1[separator]
```
# MySQL主键、外键、索引、唯一索引的区别
## 主键(primary key) 
是能够唯一标识表中某一行的属性或属性组。如，一条记录包括身份正号，姓名，年龄。身份证号是唯一能确定你这个人的，其他都可能有重复，所以，身份证号是主键。 
一个表只能有一个主键，但可以有多个候选索引。主键常与外键构成参照完整性约束，防止出现数据不一致。主键可以保证记录的唯一和主键域非空,数据库管理系统对于主键自动生成唯一索引。
**主键一定是唯一性索引，唯一性索引并不一定就是主键。 一个表中可以有多个唯一性索引，但只能有一个主键。 主键列不允许空值，而唯一性索引列允许空值。** 
事实上，主键和索引都是键，不过主键是逻辑键，索引是物理键，也就是说主键不实际存在，而索引实际存在在数据库中，主键一般都要建，主要是用来避免一张表中有相同的记录，索引一般可以不建，但如果需要对该表进行查询操作，则最好建，这样可以加快检索的速度。 
## 外键（foreign key）
是用于建立和加强两个表数据之间的链接的一列或多列。用于与另一张表的关联。是能确定另一张表记录的字段，用来维护两个表之间数据的一致性。如，A表中的一个字段，是B表的主键，那他就可以是A表的外键。
如，某个电脑生产商，它的数据库中保存着整机和配件的产品信息。用来保存整机产品信息的表叫做PC；用来保存配件供货信息的表叫做Parts。在PC表中有一个字段，用来描述这款电脑所使用的CPU型号；在Parts 表中相应有一个字段，描述的正是CPU的型号，我们可以把它想成是全部CPU的型号列表。显然，这个厂家生产的电脑，其使用的CPU一定是供货信息表(parts)中存在的型号。这时，两个表中就存在一种约束关系(constraint)——PC表中的CPU型号受到Parts表中型号的约束。若要设置外键，在参照表(referencing table，即PC表) 和被参照表 (referencedtable，即parts表) 中，相对应的两个字段必须都设置索引(index)。
简言之，表的外键就是另一表的主键，外键将两表联系起来。一般情况下，要删除一张表中的主键必须首先要确保其它表中的没有相同外键（即该表中的主键没有一个外键和它相关联）。
## 索引(index) 
索引是对表中一个或多个列的值进行排序的结构，用来快速地寻找那些具有特定值的记录。所有MySQL索引都以B-树的形式保存。如果没有索引，执行查询时MySQL必须从第一个记录开始扫描整个表的所有记录，直至找到符合要求的记录。表里面的记录数量越多，这个操作的代价就越高。如果作为搜索条件的列上已经创建了索引，MySQL无需扫描任何记录即可迅速得到目标记录所在的位置。
## 唯一索引
**所谓唯一性索引，这种索引和前面的“普通索引”基本相同，但有一个区别：索引列的所有值都只能出现一次，即必须唯一。**
## 主键和索引的区别
主键一定是唯一性索引，唯一性索引并不一定就是主键；
 一个表中可以有多个唯一性索引，但只能有一个主键；
主键列不允许空值，而唯一性索引列允许空值；
主键可以被其他字段作外键引用，而索引不能作为外键引用。
# MySQL表级锁、行级锁、页级锁、死锁、乐观锁、悲观锁、共享锁、排他锁的概念
**锁:**
锁是在执行多线程时用于强行限制资源访问的同步机制，即用于在并发控制中保证对互斥要求的满足。在DBMS中，可以按照锁的粒度把数据库锁分为行级锁（INNODB引擎）、表级锁（MYISAM引擎）和页级锁（BDB引擎）。
**在MySQL中常用存储引擎的锁机制:**
MySQL在5.5之前默认使用MyISAM存储引擎，之后使用InnoDB存储引擎。
MyISAM和MEMORY采用表级锁（table-level locking）；
InnoDB支持行级锁（row-level locking）和表级锁,默认为行级锁；
BDB采用页面锁（page-level locking）或表级锁，默认为页面锁；
**MySQL中各种锁之间的关系:**
```
                            MySQL
                         /         \
                  存储引擎MyISAM 存储引擎InnoDB
                        |       /           \ 
				    采用表级锁 支持事务     采用行级锁
				   /        \           /         \
				表共享锁   表独占锁     共享锁      排他锁
				                        \         /
				                      悲观锁(mysql自带)
乐观锁需要自己实现
```
**查看当前存储引擎:**
```
show variables like '%storage_engine%';
```
**MyISAM引擎与InnoDB引擎的不同点:**
MyISAM操作数据都是使用表级锁，MyISAM总是一次性获得所需的全部锁，要么全部满足，要么全部等待。所以不会产生死锁，但是由于每操作一条记录就要锁定整个表，导致性能较低，并发不高。
InnoDB与MyISAM的最大不同有两点：一是 InnoDB 支持事务；二是 InnoDB 采用了行级锁。也就是你需要修改哪行，就可以只锁定哪行。在Mysql中，行级锁并不是直接锁记录，而是锁索引。InnoDB 行锁是通过给索引项加锁实现的，索引分为主键索引和非主键索引两种，如果一条sql语句操作了主键索引，Mysql就会锁定这条主键索引；如果一条语句操作了非主键索引，MySQL会先锁定该非主键索引，再锁定相关的主键索引。如果没有索引，InnoDB会通过隐藏的聚簇索引来对记录加锁。也就是说：如果不通过索引条件检索数据，那么InnoDB将对表中所有数据加锁，实际效果跟表级锁一样。
## 表级锁
表级锁是MySQL中锁定粒度最大的一种锁，表示对当前操作的整张表加锁，它实现简单，资源消耗较少，被大部分MySQL引擎支持。最常使用的MYISAM与INNODB都支持表级锁定。表级锁定分为表共享读锁（共享锁）与表独占写锁（排他锁）。特点是开销小，加锁快；不会出现死锁；锁定粒度大，发出锁冲突的概率最高，并发度最低。
## 行级锁
行级锁是Mysql中锁定粒度最细的一种锁，表示只针对当前操作的行进行加锁。行级锁能大大减少数据库操作的冲突。其加锁粒度最小，但加锁的开销也最大。行级锁分为共享锁和排他锁。特点是开销大，加锁慢；会出现死锁；锁定粒度最小，发生锁冲突的概率最低，并发度也最高。
## 页级锁
表级锁是MySQL中锁定粒度介于行级锁和表级锁中间的一种锁。表级锁速度快，但冲突多，行级冲突少，但速度慢。所以取了折衷的页级，一次锁定相邻的一组记录。BDB支持页级锁。特点是开销和加锁时间界于表锁和行锁之间；会出现死锁；锁定粒度界于表锁和行锁之间，并发度一般。
## 死锁
指两个事务或者多个事务在同一资源上相互占用，并请求对方所占用的资源，从而造成恶性循环的现象。 
**出现死锁的原因:**
系统资源不足； 
进程运行推进的顺序不当； 
资源分配不当。 
**产生死锁的四个必要条件:**
互斥条件： 一个资源只能被一个进程使用；
请求和保持条件：进行获得一定资源，又对其他资源发起了请求，但是其他资源被其他线程占用，请求阻塞，但是也不会释放自己占用的资源；
不可剥夺条件： 指进程所获得的资源，不可能被其他进程剥夺，只能自己释放；
环路等待条件： 进程发生死锁，必然存在着进程-资源之间的环形链。
**常见的三种避免死锁的方法:**
如果不同程序会并发存取多个表，尽量约定以相同的顺序访问表，可以大大降低死锁机会；
在同一个事务中，尽可能做到一次锁定所需要的所有资源，减少死锁产生概率；
对于非常容易产生死锁的业务部分，可以尝试使用升级锁定颗粒度，通过表级锁定来减少死锁产生的概率。

**数据库也会发生死锁的现象，数据库系统实现了各种死锁检测和死锁超时机制来解除死锁，锁监视器进行死锁检测，MySQL的InnoDB处理死锁的方式是将持有最少行级排它锁的事务进行回滚。**
## 乐观锁
乐观锁和悲观锁都是为了解决并发控制问题， 乐观锁可以认为是一种在最后提交的时候检测冲突的手段，而悲观锁则是一种避免冲突的手段。 
乐观锁应用系统层面和数据的业务逻辑层次上的（实际上并没有加锁，只不过大家一直这样叫而已），利用程序处理并发， 它假定当某一个用户去读取某一个数据的时候，其他的用户不会来访问修改这个数据，但是在最后进行事务的提交的时候会进行版本的检查，以判断在该用户的操作过程中，没有其他用户修改了这个数据。
乐观锁不是数据库自带的，需要我们自己去实现。乐观锁的实现大部分都是基于版本控制实现的，级别高低是：脏读 < 不可重复读 < 幻读（级别介绍详细见数据库的事务隔离级别）。 除此之外，还可以通过时间戳的方式，通过提前读取，事后对比的方式实现。
## 悲观锁
乐观锁和悲观锁都是为了解决并发控制问题， 乐观锁可以认为是一种在最后提交的时候检测冲突的手段，而悲观锁则是一种避免冲突的手段。 
每次拿数据的时候都认为别的线程会修改数据，所以在每次拿的时候都会给数据上锁。上锁之后，当别的线程想要拿数据时，就会阻塞，直到给数据上锁的线程将事务提交或者回滚。传统的关系型数据库里就用到了很多这种锁机制，比如行锁，表锁，共享锁，排他锁等，都是在做操作之前先上锁。与乐观锁相对应的，悲观锁是由数据库自己实现，要用的时候，我们直接调用数据库的相关语句就可以。
**共享锁和排它锁是悲观锁的不同的实现，它俩都属于悲观锁的范畴。**
**共享锁:**
共享锁又称为读锁，简称S锁，共享锁就是多个事务对于同一数据可以共享一把锁，都能访问到数据，但是只能读不能修改。
比如事务T对数据对象A加上S锁，则事务T只能读A；其他事务只能再对A加S锁，而不能加X锁，直到T释放A上的S锁。这就保证了其他事务可以读A，但在T释放A上的S锁之前不能对A做任何修改。
**排他锁:**
排他锁又称为写锁，简称X锁，排他锁就是不能与其他所并存，如一个事务获取了一个数据行的排他锁，其他事务就不能再获取该行的其他锁，包括共享锁和排他锁，但是获取排他锁的事务是可以对数据就行读取和修改。
比如事物T对数据对象A加上X锁，则只允许T读取和修改A，其它任何事务都不能再对A加任何类型的锁，直到T释放A上的锁。它防止任何其它事务获取资源上的锁，直到在事务的末尾将资源上的原始锁释放为止。

**注意:**
mysql InnoDB引擎默认的修改数据语句，update,delete,insert都会自动给涉及到的数据加上排他锁，select语句默认不会加任何锁类型，如果加排他锁可以使用select ...for update语句，加共享锁可以使用select ... lock in share mode语句。所以加过排他锁的数据行在其他事务种是不能修改数据的，也不能通过for update和lock in share mode锁的方式查询数据，但可以直接通过select ...from...查询数据，因为普通查询没有任何锁机制。