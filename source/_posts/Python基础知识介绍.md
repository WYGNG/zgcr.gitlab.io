---
title: Python基础知识介绍
date: 2019-02-27 16:49:56
tags:
- Python
categories:
- Python
---

# Python的输入与输出、Python输入EOF判断
输入采用input()方法。input输入的元素都是以str形式保存的。输出采用print()方法。
Python中没有代表EOF的字符，因此检测到输入EOF时会直接报错EOFerror。因此我们要采用下面的写法来检测输入到EOF后停止循环输入：
```python
while True:
   try:
      s = int(input())
      print(s)
   except:
      print('input complete')
      break
```
如果在cmd.exe中运行这段代码，输入Ctrl+Z即可停止输入。如果是在Pycharm内建运行框中运行这段代码，输入Ctrl+D即可停止输入。
如果是在Python中读取文件，Python到读取到文件结尾后是返回空字符串的，所以python可以这样判断：
```python
str = ''
with open('readme.txt', 'r', encoding='utf-8') as fp:
   while True:
      s = fp.read(10)
      if s == '':
         break
      str += s
print(str)
```
# Python字符串类型、空值类型、Unicode编码、UTF-8编码、格式化字符串
字符串是以单引号'或双引号"括起来的任意文本。如果字符串内部既包含'又包含"，可以用转义字符\来标识。\n表示换行，\t表示制表符，\\表示\。Python还允许用r' '表示' '内部的字符串默认不转义。
空值是Python里一个特殊的值，用None表示。None不能理解为0，因为0是有意义的，而None是一个特殊的空值。
Unicode编码把所有语言都统一到一套编码里，这样就不会再有乱码问题。Unicode最常用的是用两个字节表示一个字符(很生僻的字符会被编码成4个字节)。比如把ASCII编码的A用Unicode编码表示，只需要在前面补0就可以，因此，A的Unicode编码是00000000 01000001。
UTF-8编码相当于对nicode编码的优化，它一个Unicode字符根据不同的数字大小编码成1-6个字节，常用的英文字母被编码成1个字节，汉字通常是3个字节，只有很生僻的字符才会被编码成4-6个字节。ASCII编码实际上可以被看成是UTF-8编码的一部分，所以，大量只支持ASCII编码的历史遗留软件可以在UTF-8编码下继续工作。
在计算机内存中，统一使用Unicode编码，当需要保存到硬盘或者需要传输的时候，就转换为UTF-8编码。
**注意：**
Python3的字符串类型是str，在内存中以Unicode编码表示。如果要在网络上传输，或者保存到磁盘上，就需要把str变为以字节为单位的bytes。
Python提供了ord()方法来获取某个字符的Unicode编码，而chr()方法把Unicode编码转换成对应的字符。如：
```python
s1 = "A"
s2 = "1"
s3 = "你"
print(ord(s1))
print(ord(s2))
print(ord(s3))
print(chr(66))
print(chr(20500))
```
Python对bytes类型的数据用带b前缀的单引号或双引号表示：
```python
x = b'ABC'
```
Python提供了encode()方法可以将str编码为指定类型的bytes，decode()方法把bytes变为str。如：
```python
s1 = "ABC"
s2 = s1.encode('ascii')
s3 = "你好"
print(s1.encode('ascii'))
print(s3.encode('utf-8'))
print(s2.decode('ascii'))
```
含有中文的str无法用ASCII编码，因为中文编码的范围超过了ASCII编码的范围，Python会报错。为了避免乱码问题，我们应当坚持使用UTF-8编码对str和bytes进行转换。
当Python解释器读取源代码时，为了让它按UTF-8编码读取，我们通常在文件开头写上这两行命令：
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```
第一行注释是为了告诉Linux/OS X系统，这是一个Python可执行程序，Windows系统会忽略这个注释；第二行注释是为了告诉Python解释器，按照UTF-8编码读取源代码，否则，你在源代码中写的中文输出可能会有乱码。除此之外，我们还要确保所使用的IDE正在使用UTF-8 编码。
**格式化字符串的输出方式：**
%运算符就是用来格式化字符串的。%s表示用字符串替换，%d表示用整数替换，有几个%?占位符，后面就跟几个变量或者值，顺序要对应。格式化整数和浮点数还可以指定是否补0和整数与小数的位数。我们还可以用format用法来格式化输出。如：
```python
print("%2d %02d" % (3, 1))
print("%.2f" % 3.1415926)
print("{} {}".format(3, 1.21))
print("{:02d} {:.2f}".format(3, 3.1415926))
```
# Python条件判断、循环
判断语句的形式：
```python
if 判断条件：
    执行语句......
elif 判断条件：
    执行语句......
else：
    执行语句......
```
注意elif、else的判断语句视情况添加。
Python提供了for循环和while循环(在Python中没有do..while循环)，形式如下：
```python
for x in sequence:
    statements(s)

while 判断条件：
    执行语句......
```
break语句会打断当前层的循环，continue语句则会直接结束本次循环，开始下一次循环。
# Python不可变对象和可变对象
不可变对象指该对象所指向的内存中的值不能被改变。当改变某个变量时候，由于其所指的值不能被改变，相当于把新的值放在一个新地址，变量再指向这个新的地址。
可变对象指该对象所指向的内存中的值可以被改变。变量改变后，实际上是其所指的值直接发生改变，并没有发生复制行为，也没有开辟新的地址，通俗点说就是原地改变。
在Python中，数值类型(int和float)、字符串、元组都是不可变类型。而列表、字典、集合是可变类型。
# Python字典与集合
字典可存储任意类型对象，如字符串、数字、元组等其他容器模型。字典的每个键值 key:value对用冒号分割，每个键值对之间用逗号分割，整个字典包括在花括号{}中。一个key只能对应一个value，同一个key放入多个value值后放入的会把前面放入的value值覆盖。
.get()方法可指定key得到对应的value，如果key不存在则返回None。pop()方法可以删除一个key:value。
**注意：**
字典内部存放的顺序和key放入的顺序无关。字典查找和插入的速度极快，不会随着key的增加而变慢，但需要占用大量的内存，内存浪费多。它是一种用空间来换取时间的一种方法。
字典的key必须是不可变对象。字典(Dictionary)根据key来计算value的存储位置，这个通过key计算位置的算法称为哈希算法。要保证hash的正确性，作为key的对象就不能变。在Python中，字符串、整数等都是不可变的，因此，可以放心地作为key。而list是可变的，就不能作为key。

集合可以看成数学意义上的无序和无重复元素的集合。set同样不可以放入可变对象。集合用来存储一组key，但不存储value，在set中，没有重复的key。
我们可以使用大括号{ }或set()函数创建集合。创建一个空集合必须用set()而不是 { }，因为{ }是用来创建一个空字典。
add()方法可以向set中添加元素，remove()方法可以删除元素，两个set还可以做数学意义上的交集、并集等操作。
# Python列表与元组
列表即一个方括号内的逗号分隔值出现。列表的数据项不需要具有相同的类型，并且list里的元素也可以是另一个list。如：
```python
M = ['A', 1, 1.5]
```
len()可求得列表元素个数，.append()末尾增加一个元素，.insert()指定位置插入一个元素，pop()删除指定位置元素。
元组与列表类似，不同之处在于元组的元素不能修改。元组使用小括号( )，列表使用方括号[ ]。元组没有.append()，.insert()，pop()方法。元组中其他访问和获取元素的方法和list是一样的。元组在定义时，其中的元素就必须被确定下来。
只有1个元素的元组定义时必须加一个逗号,，来消除歧义：
```python
T = (1,) # 这是一个元组
```
元组的元素是不可变的，但是如果元组的某个元素是列表时，列表内的元素是可变的。
# Python定义函数、匿名函数、变量作用域
定义一个函数的形式：
```python
def 函数名(x):
    函数体 # 如果是空函数，则函数体直接写pass
```
python中使用lambda表达式来创建匿名函数。所谓匿名就是不再使用def语句这样的标准形式定义一个函数。lambda的主体是一个表达式，而不是一个代码块。我们仅仅能在lambda表达式中封装有限的逻辑进去。lambda 函数拥有自己的命名空间，且不能访问自己参数列表之外或全局命名空间里的参数。如：
```python
M = map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9])
for i in M:
   print(i)
```
Python的作用域一共有4种，分别是：
* L(Local)局部作用域
* E(Enclosing)闭包函数外的函数中
* G(Global)全局作用域
* B(Built-in)内建作用域

举例：
```python
x = int(2.9)  # 内建作用域 
g_count = 0  # 全局作用域
def outer():    
    o_count = 1  # 闭包函数外的函数中    
    def inner():        
        i_count = 2  # 局部作用域
```
Python解释器寻找变量时总是按照L–>E–>G–>B的规则查找，即：在局部找不到，便会去局部外的局部找(例如闭包)，再找不到就会去全局找，再者去内建中找。
当函数在局部作用域内想修改全局作用域的变量时，就要用到global关键字。如果要修改嵌套作用域(闭包函数外的函数中作用域，即外层非全局作用域)中的变量则需要 nonlocal关键字。
# Python切片、迭代、列表生成式、生成器、迭代器
python3支持切片操作的数据类型有list、tuple、string、unicode、range，比如列表(list)中，我们可以用切片取出其中一部分元素。切片操作是指按照步长，截取从起始索引到结束索引，但不包含结束索引（也就是结束索引减1）的所有元素。切片操作返回和被切片对象相同类型对象的副本。如：
```python
M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(M[0:3])
```
Python中，对一个列表或元组通过for in循环来遍历它，这种遍历就叫做迭代。Python中列表、元组、字典、字符串、生成器都是可迭代对象。
默认情况下，字典迭代的是key。如果要迭代value，可以用for value in d.values()，如果要同时迭代key和value，可以用for k, v in d.items()。如：
```python
M = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in M:
   print(i)
```
Python中提供了一个判断函数来判断一个对象是否是可迭代对象。即collections模块里的Iterable类，使用该类的isinstance()函数即可判断。如：
```python
from collections import Iterable

print(isinstance('abc', Iterable))
print(isinstance([1, 2, 3], Iterable))
```
Python内置的enumerate()函数可以把一个list变成索引-元素对，这样就可以在for循环中同时迭代索引和元素本身。如：
```python
S = ["A", "B", "C", "D"]
for i, value in enumerate(S):
   print(i, value)
```
列表生成式是Python内置的用来生成列表的特定语法形式的表达式。通过列表生成式，我们可以用一行语句直接创建一个列表。如：
```python
S = [x * x for x in range(1, 11) if x % 2 == 0]
print(S)
```
类似地，也有字典生成式和集合生成式，区别就是中括号[]改成大括号{}。因为集合自带去重功能，如果计算得出的元素中有重复值，在集合中一个值只会录入一个元素。如：
```python
Dict = {'a': 10, 'b': 34}
Dict = {v: k for k, v in Dict.items()}
print(Dict)

Set = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0}
Set = {i ** 3 for i in Set if i % 2 == 0}
print(Set)
```
生成器是Python中一边循环一边计算的机制。如果列表元素可以按照某种算法推算出来，我们可以在循环的过程中不断推算出后续下一个元素，这样就不必创建完整的列表(list)，从而节省大量的空间。这种方法理论上可以表示元素数量无穷多个的列表。但要注意的是我们不能以下显示出这个列表的所有元素，只能一个一个计算出来。如：
```python
S = (x * x for x in range(10))
for i in S:
   print(next(S))
```
如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个生成器。生成器在执行过程中，遇到yield关键字就会中断执行，下次调用则继续从上次中断的位置(即yield语句的下一条语句)继续执行。如：
```python
def fib(max):
   n, a, b = 0, 0, 1
   while n < max:
      yield b
      a, b = b, a + b
      n = n + 1
   return 'done'


for i in fib(6):
   print(i)
```
迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能前进不能回退。同生成器类似，迭代器不要求准备好整个迭代过程中所有的元素。仅仅是在迭代至某个元素时才计算该元素。Iterator对象可以被next()函数调用并不断返回下一个数据，直到没有数据时抛出StopIteration错误。
可迭代对象(Iterable)有：一类是集合数据类型，如列表(list)、元组(tuple)、字典(Dictionary)、集合(set)、字符串等；一类是生成器(generator)，包括列表生成式改写的生成器和带yield的生成器函数。
collections模块里的Iterable类，使用该类的isinstance()函数可判断一个对象是否是可迭代对象(Iterable)。
凡是可作用于for循环的对象都是Iterable类型；凡是可作用于next()函数的对象都是Iterator类型；集合数据类型如list、dict、str等是可迭代对象但不是迭代器，不过可以通过iter()函数获得一个Iterator对象。
# Python高阶函数：map/reduce、filter、sorted
**一个函数如果它的参数中有些参数可以接收另一个函数作为参数，这种函数就称之为高阶函数。**
map()函数接收两个参数，一个是函数，一个是可迭代对象，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回(Python3中返回Iterator迭代器，Python2中返回列表(list))。如：
```python
def f(x):
   return x * x


r = map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(list(r))
```
reduce() 函数会对参数序列中元素进行累积。用传给reduce中的函数(有两个参数)先对集合中的第1、2个元素进行操作，得到的结果再与第三个数据用函数运算，依次类推直到运算完集合中的最后一个元素，最后返回函数计算结果。如：
```python
from functools import reduce


def add(x, y):
   return x + y


s = reduce(add, [1, 3, 5, 7, 9])
print(s)
```
filter()函数用于过滤序列。filter()函数接收一个函数和一个序列。filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。如：
```python
def odd(n):
   return n % 2 == 1


print(list(filter(odd, [1, 2, 4, 5, 6, 9, 10, 15])))
```
filter()函数返回的是一个Iterator，和map()函数类似，也是一个惰性序列，所以要强迫filter()完成计算结果，需要用list()函数获得所有结果并返回list。
sorted()函数用于排序。sorted()函数排序后的结果另返回副本，原始输入不变。默认情况下，如果用sorted()或sort()对字符串排序，是按照ASCII的大小比较的，由于'Z' < 'a'，结果，大写字母Z会排在小写字母a的前面。
我们还可以设置reverse=True来实现倒序排序。sorted()函数还可以接收一个key函数来实现自定义的排序。key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序。如：
```python
S = [40, 5, -12, 9, -21]
print(sorted(S, key=abs))
print(S)
M = ['ABC', 'zDA', 'bef', 'Zec', 'Mnn', 'abc']
print(sorted(M, key=str.lower, reverse=True))
```
# Python类、实例、构造函数
面向对象最重要的概念就是类和实例，类是抽象的模板，而实例是根据类创建出来的一个个具体的“对象”，每个对象都拥有相同的方法，但各自的数据可能不同。
属性的名称前加上两个下划线，就变成了一个私有变量（private），只有内部可以访问，外部不能访问。想从内部访问私有变量，我们还要在这个类中定义相应的方法。
在Python中，class后面紧接着类名，类名通常是大写开头的单词，(object)表示该类是从哪个类继承下来的，如果没有合适的继承类，就使用object类，这是所有类最终都会继承的类。
类中有一个定义的构造函数__init__()，担负着对类进行初始化的任务。__init__方法的第一个参数永远是self，表示创建的实例本身，因此，在__init__方法内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身。
如：
```python
class Student(object):
   def __init__(self, name, score):
      self.__name = name
      self.__score = score

   def print_score(self):
      print('%s:%s' % (self.__name, self.__score))

   def read_name(self):
      print(self.__name)

   def read_score(self):
      print(self.__score)

   def edit_score(self):
      s = int(input("请输入新的分数："))
      if 0 <= s <= 100:
         self.__score = s
      else:
         print("分数超出范围！")


student1 = Student("zhangsan", 90)
student1.print_score()
student1.read_name()
student1.edit_score()
student1.read_score()
```
Python中也存在多态。当子类覆写了其父类中的某个方法，在子类调用这个方法时，总是会调用其覆写后的方法。
一个子类的实例不仅是子类的数据类型，还是父类的数据类型！但是反过来则不行。
# Python获取对象信息函数：type()、isinstance()、dir()
使用type()函数可以判断对象的类型，如果一个变量指向了函数或类，也可以用type判断。
types模块中提供了四个常量types.FunctionType、types.BuiltinFunctionType、types.LambdaType、types.GeneratorType，分别代表函数、内建函数、匿名函数、生成器类型。如：
```python
import types


class Student(object):
   name = 'Student'


a = Student()
print(type(123))
print(type('abc'))
print(type(None))
print(type(abs))
print(type(a))


def fn():
   pass


print(type(fn) == types.FunctionType)
print(type(abs) == types.BuiltinFunctionType)
print(type(lambda x: x) == types.LambdaType)
print(type((x for x in range(10))) == types.GeneratorType)
```
对于有继承关系的类，我们要判断该类的类型，可以使用isinstance()函数。由于子类的实例不仅是子类的类型，也是继承的父类的类型。故isinstance()判断的是一个对象是否是该类型本身，或者位于该类型的父继承链上。能用type()判断的基本类型也可以用isinstance()判断，并且还可以判断一个变量是否是某些类型中的一种。如：
```python
class Animal(object):
   def run(self):
      print("动物在跑")


class Dog(Animal):
   def eat(self):
      print("狗在吃")


class Cat(Animal):
   def run(self):
      print("猫在跑")


dog1 = Dog()
cat1 = Cat()
print(isinstance(dog1, Dog))
print(isinstance(cat1, Cat))
print(isinstance(cat1, Animal))
print(isinstance(dog1, Animal))

print(isinstance('a', str))
print(isinstance(123, int))
print(isinstance(b'a', bytes))
print(isinstance([1, 2, 3], (list, tuple)))
print(isinstance((1, 2, 3), (list, tuple)))
```
dir()函数可以获得一个对象的所有属性和方法，它返回一个包含字符串的list。如：
```
print(dir('abc'))
```
# __init__.py文件介绍
在Python的每一个包中，都有一个\_\_init\_\_.py文件，这个文件定义了包的属性和方法。然后是一些模块文件和子目录，假如子目录中也有\_\_init\_\_.py，那么它就是这个包的子包。当你将一个包作为模块导入的时候，实际上导入了它的\_\_init\_\_.py 文件。
一个包是一个带有特殊文件 \_\_init\_\_.py 的目录。\_\_init\_\_.py 文件定义了包的属性和方法。它也可以什么也不定义，只是一个空文件，但是必须存在。如果 \_\_init\_\_.py 不存在，这个目录就仅仅是一个目录，而不是一个包，它就不能被导入或者包含其它的模块和嵌套包。
\_\_init\_\_.py 中还有一个重要的变量，叫做\_\_all\_\_。这个变量可设置可不设置。
假如有一个包名为lib，我们设置了\_\_init\_\_.py文件中的\_\_all\_\_变量，那么我们使用类似下面的命令：
```python
from lib import *
```
import就会把注册在lib包\_\_init\_\_.py 文件中\_\_all\_\_变量列表中的子模块和子包导入到当前作用域中来。
# if __name__ == "__main__":的含义
\_\_name\_\_是标识模块名称的一个系统变量。
* 假如当前模块是主模块（也就是调用其他模块的模块），那么此模块的\_\_name\_\_变量值就是\_\_main\_\_；
* 假如此模块是被import调用的，则此模块的\_\_name\_\_变量值为文件名(不加后面的.py)；

这样我们就可以在某些.py文件中写调试代码的时候，使用：
```python
if __name__ == "__main__":
```
作为调试这部分代码是否运行的条件。当外部模块调用的时候遇到这个判断条件会判false，从而不执行我们的调试代码。当我们想排查问题时，直接执行该模块文件，这时候这个判断条件会判ture，从而执行我们的调试代码。
# Python使用pickle保存和提取数据
pickle是一个python中用来压缩、保存、提取文件的模块，可以保存字典和列表。.dump()方法保存数据到某个文件，.load()方法从某个文件加载其中的数据。如：
```python
import pickle

a_dict = {'da': 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)  # 数据保存在file中
file.close()  # 关闭文件

with open('pickle_example.pickle', 'rb')as file:
   a_dict1 = pickle.load(file) # 从文件中读取数据
print(a_dict1)
```
# python字符串前缀u、r、b
前缀u开头的Python字符串表示该字符串为unicode编码(unicode为python解释器内部统一的编码格式)。在python3中，字符串的存储方式都是以Unicode字符来存储的，所以前缀带不带u，其实都一样。
以r或R开头的python中的字符串表示该字符串为非转义的原始字符串，即字符串中默认都是普通字符，没有转义字符。
前缀b开头的Python字符串代表是bytes类型的字符串(注意只有Python3中才是)。这类字符串中的字符必须是十六进制数，或者ASCII字符。
# import模块时Python解释器的搜索顺序
默认情况下，Python解释器会搜索当前目录、所有已安装的内置模块和第三方模块，搜索路径存放在sys模块的path变量中。Python解释器总是按顺序依次搜索path变量中存储的路径中有无import的模块，当搜索到第一个该名称模块时即停止搜索。
我们还可以使用sys.path.append添加自己的搜索目录。这种方法是在.py文件运行时修改，运行结束后失效。
# os.getcwd()、sys.path[0]、sys.argv[0]和__file__的区别
* os.getcwd()函数获得用Python解释器运行某个.py文件的当前目录所在位置。
* sys.argv[0]即当前执行的.py文件所在的目录位置(不带文件名)，由于Python会自动把sys.argv[0]加入sys.path，因此sys.path[0]也是当前执行的.py文件所在的目录位置。
* __file__是所在.py文件的完整路径(带上文件名的)，但是这个变量有时候返回相对路径，有时候返回绝对路径，我们最好用下面这个命令，这个可以保证得到绝对路径：

```python
os.path.realpath(__file__)
```

举个例子，假如文件结构如下面所示：
```python
C:test
|-getpath
    |-path.py
    |-sub
        |-sub_path.py
```
我们在C:\test下面执行python getpath/path.py，这时sub_path.py中上面几种函数用法对应的路径为：
* os.getcwd():"C:\test"，即用Python解释器运行.py文件的当前目录。
* sys.path[0]或sys.argv[0]:"C:\test\getpath"，即被执行脚本path.py的所在目录。
* os.path.realpath(__file__)[0]:"C:\test\sub\sub_path.py"，得到的时sub_path.py的完整路径(含文件名)；如果前面再加上os.path.split，输出的路径就是"C:\test\sub"。

# Python tqdm模块的使用：可视化进度条
tqdm是一个快速，可扩展的Python进度条，可以在Python的for循环中添加一个进度提示信息，进度条可以针对任意迭代器对象。这是一个第三方模块，需要使用下命的命令安装：
```python
python -m pip install tqdm
```
使用示例：
```python
import tqdm
import time

count1 = 0

for i in tqdm.trange(100):
   count1 += i
   time.sleep(0.05)

count2 = 0
x = tqdm.tqdm(range(100))

for i in x:
   count2 += i
   time.sleep(0.05)
   x.set_description("现在是第{}轮".format(i))  # 进度条标题
```