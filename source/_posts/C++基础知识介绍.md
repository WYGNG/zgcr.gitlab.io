---
title: C++基础知识介绍
date: 2019-02-25 18:27:30
tags:
- C++
categories:
- C++
---

# using namespace std
这句话的含义是使用“std”这个命名空间。因为不同的命名空间中可能有相同名称的函数，为了避免冲突，在程序起始加上这个语句，就可以限定使用的命名空间，防止调用到同名的其他函数。
C++常用的cin、cout等方法都在std命名空间中，因此：
```cpp
using namespace std;
```
经常在C++程序中使用。如果不写这句，那么直接使用std中的方法cin、cout等时必须在方法名前加上std::。
# C/C++中把自定义函数放到main函数之后
我们只要在main函数前先声明这个自定义函数，然后写main函数，再在main函数之后定义这个自定义函数即可。如：
```cpp
#include <cstdio>

void fun1(int *); // 声明fun1函数,声明时必须写参数类型,但参数变量名可写可不写,不影响运行
int main() {
   int k = 0;
   fun1(&k); // 调用fun1函数
   printf("%d\n", k);
   return 0;
}

void fun1(int *a) { // 定义fun1函数
   (*a)++;
}
```
# C/C++输入EOF判断
EOF是一个定义在C语言头文件stdio.h中的常量,等于-1。当我们不确定输入的数据数量时，常用输入EOF来判断输入是否结束。如：
```cpp
#include <cstdio>
#include <iostream>

int main() {
   int data;
   while (scanf("%d", &data) != EOF) { // 输入EOF后循环输入停止
      printf("%d", data);
   }
   return 0;
}
```
**如何用键盘输入EOF：**
* windows系统中，如果在cmd.exe中运行C/C++程序，想手动输入EOF，在输入数据结束后先enter换行，然后Ctrl+Z，然后再enter换行。
* linux系统中，如果在terminal中运行C/C++程序，想在输入数据结束后手动输入EOF，也是在输入结束后先enter换行，然后输入Ctrl+D，然后再enter换行。
* Clion中，想在Clion的内建运行框run中实现输入数据结束后手动输入EOF，在输入结束后，先enter换行，然后Ctrl+D即可。一定要先enter换行，否则输出会不完整！(注意我的Clion使用cygwin编译器)
# C++中使用C语言头文件
C语言中的头文件都是带.h后缀的，如果想在C++中使用C语言中头文件，只需要将.h后缀去掉，再在头文件名前加一个c即可。
用法举例：
```cpp
#include <cstdio> // 相当于C语⾔中的#include <stdio.h>
#include <cstring> // 相当于C语⾔中的#include <string.h>
#include <cmath> // 相当于C语⾔中的#include <math.h>
```
# C++输入和输出
C++中的头文件iostream中包含两个基础类型istream和ostream , 分别表示输入流和输出流。C++的输入方法为cin，输出方法为cout，都在头文件iostream中。
用法举例：
```cpp
#include <iostream>

using namespace std;

int main() {
   int n;
   cin >> n;
   cout << n << endl;
   cout << "ok!" << endl;
   return 0;
}
```
cin、cout输入和输出时不需要区分数据类型，endl作用等同于"\n"。cin输入时默认以空格间断。
**注意：**
cin和cout使用更加方便，但是输入输出的效率不如scanf和printf快，所以如果刷题时对时间复杂度要求较高，那么最好使用scanf和printf。
# C++for循环变量声明
在C++中，for循环使用的变量可直接在for循环中定义，然后使用。
用法举例：
```cpp
#include <iostream>

using namespace std;

int main() {
   int n, m;
   cin >> n >> m;
   for (int i = 0; i < n; i++) {
      m++;
      cout << m << endl;
   }
   return 0;
}
```
for循环中的i直接定义后使用。
# C++bool变量
bool型变量只有两个值true和false。如果你声明了一个bool变量后为其赋值一个数字，C++会把非零值解释为true，零值解释为false。
用法举例：
```cpp
#include <iostream>

using namespace std;

int main() {
   bool flag1 = true;
   bool flag2 = 10;
   bool flag3 = 0;
   cout << boolalpha << flag1 << endl;
   cout << boolalpha << flag2 << endl;
   cout << boolalpha << flag3 << endl;
   return 0;
}
```
cout之后加上boolalpha则会把bool型变量输出为true或false。
# C++const定义常量
C语⾔中我们用#define定义常量。C++中使用const这个限定符定义常量，这样做的好处是可以定义常量的类型。
用法举例：
```cpp
#include <iostream>

using namespace std;
const int a = 10;

int main() {
   cout << a << endl;
   return 0;
}
```
一个变量一旦定义为常量，就不可修改。
# C++string类
C++的string类对字符串的定义、拼接、输出、处理变得更加简单了。要注意的是C++的string类只能用cin和cout处理，无法用scanf和printf处理。
用法举例：
```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
   string s1 = "hello";
   string s2 = "world";
   string s3 = s1 + s2;
   string s4;
   cin >> s4;
   cout << s3 << endl;
   cout << s4 << endl;
   string s5;
   getline(cin, s5);
   long len = s5.length();
   cout << len << endl;
   cout << s5 << endl;
   cout << len << endl;
   string s6 = s5.substr(0, 3);
   cout << s6 << endl;
   return 0;
}
```
string的长度可以用s.length()获取，如果想要读入一整行的字符串，需要用getline，geline函数默认以回车间断。substr的作用是截取某个字符串中的子串。(0,3)标识从下标0开始，截取3个字符。
# C++结构体struct和C语言结构体的区别
定义好结构体stu之后，使用这个结构体类型时，在C语言需要写关键字struct，而C++中可以不写。如：
```cpp
struct student {
   int grade;
   float score;
};
struct student class1[10]; // C语⾔中使用结构体类型要加上struct
student clss2[10];// C++中使用结构体类型可以不写struct
```
# C++sort函数的使用
sort函数用来对数组(int数组或vector数组)排序，要使用sort函数必须引入algorithm头文件。sort参数有三个，第一个和第二个参数指定了待排序的元素下标区间，第三个参数指定了具体排序规则函数(即cmp函数，若不指定第三个参数则默认递增排序)。对于vector容器，要用v.begin()和v.end()表示头尾。
用法举例：
```cpp
#include <iostream>
#include <algorithm>

using namespace std;

struct student { // 定义一个结构体student，number表示学号，score表示分数
   int number;
   int score;
};

bool cmp1(student a, student b) { // cmp函数，返回值是bool类型，传入参数的类型是结构体student
   if (a.score != b.score) // 如果学生分数不不同，就按照分数从大到小排列
      return a.score > b.score;
   else // 如果学生分数相同，就按照学号从小到大排列
      return a.number < b.number;
}

bool cmp2(student a, student b) { //使用三目运算符写法更加精炼
   return a.score != b.score ? a.score > b.score : a.number < b.number;
}

int main() {
   student class1[4];
   class1[0] = {1, 100};
   class1[1] = {2, 60};
   class1[2] = {3, 80};
   class1[3] = {4, 60};
   sort(class1, class1 + 4, cmp1);
   for (int i = 0; i < 4; i++)
      cout << class1[i].number << " " << class1[i].score << endl;
   cout << endl;
   return 0;
}
```
**注意：**
上面的例子是对自定义的结构体排序，此时必须定义cmp函数。
# cctype中判断字符的快捷方法
C++中的cctype头文件实际上就是C语言中的ctype.h头文件。假如我们要判断一个字符是否属于字母，可以用更快捷的方法：
```cpp
#include <iostream>
#include <cctype>

using namespace std;

int main() {
   char c;
   cin >> c;
   if (isalpha(c)) {
      cout << "c is alpha";
   }
   return 0;
}
```
类似的，还有isalpha、islower、isupper、isalnum、isblank、isspace这些方法判断字母、小写字母、大写字母、字母+数字、space+\t、space+\t+\r+\n等，这些方法都在cctype头文件中。
还有两个经常用到的方法tolower和toupper，作用是判断+转换，如果某个字符不是小写/大写则将该字符转换成小写/大写。
# C++STL：变长数组vector的使用
C++中的动态数组vector可以在运行时设置数组的长度、数组满后在末尾添加新元素、改变数组中某个空间的元素值，改变数组长度，要使用vector，需要引入vector头文件(也在命名空间std中)。
vector、stack、queue、map、set这些在C++中都叫做容器，这些容器的大小都可以用 .size() 方法获取。
begin和end操作生成指向容器中第一个元素和尾元素之后位置的迭代器。这两个迭代器最常见的用途是形成一个包含容器中所有元素的迭代器范围(一个迭代器范围由一对迭代器表示) 。
用法举例：
```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
   vector<int> v1; //定义一个vector v1,此时vector v1没有分配大小,大小默认为0,vector默认初始化所有元素值为0
   cout << v1.size() << endl;
   v1.resize(10); //给vector重分配大小为10,vector默认初始化所有元素值为0
   cout << v1.size() << " " << v1[0] << endl;
   vector<int> v2(20); //定义一个vector v2,此时vector v2已定义大小为20,vector默认初始化所有元素值为0
   cout << v2.size() << " " << v2[0] << endl;
   vector<int> v3(10, 5); //定义一个vector v3,此时vector v3已定义大小为10,每个元素值都是5
   cout << v3.size() << " " << v3[0] << endl;
   return 0;
}
```
vector如果一开始不定义大小，那么大小默认为0，之后可以用.resize方法改变大小。.size()方法可以查看vector当前大小。如果vector定义时未指定元素值，那么所有元素值默认初始化为0。
**对vector使用for循环的三种写法：**
```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
   vector<int> a(20, 1); //指定vector c大小为20,值为1
   for (int i = 0; i < a.size(); i++) { //传统for语句遍历vector
      cout << a[i] << " ";
   }
   cout << endl;
   for (auto &it : a) { //范围for语句遍历vector
      // 在范围for语句中预存了end()的值,一旦在序列中添加(删除)元素,end函数值可能失效,所以不能在范围for循环中向vector对象添加(删除)元素
      cout << it << " ";
   }
   cout << endl;
   int m = 2;
   for (auto it = a.begin(); it != a.end(); it++) { // 使用迭代器方式遍历vector
      //凡是使用了迭代器的循环体,都不要向迭代器所属的容器添加(删除)元素
      *it = m;
      cout << *it << " ";
   }
   cout << endl;
   return 0;
}
```
# C++STL：集合set的使用
集合set中的元素各不相同，且set会按照元素值大小从小到大进行排序。要使用集合set，需要添加set头文件。insert()方法可以向set中插入元素，.erase()方法删除元素，.find()方法查找元素。
用法举例：
```cpp
#include <iostream>
#include <set>

using namespace std;

int main() {
   set<int> s;
   s.insert(1);
   cout << *(s.begin()) << endl; //*号表示取指针指向的地址中的值
   for (int i = 0; i < 6; i++) {
      s.insert(i);
   }
   for (auto it = s.begin(); it != s.end(); it++) {
      cout << *it << " ";
   }
   cout << endl << boolalpha << (s.find(2) != s.end()) << endl; //查找集合s中的值，如果结果等于s.end()表示未找到
   //s.end()表示s的最后一个元素的下一个元素所在的位置
   cout << boolalpha << (s.find(10) != s.end()) << endl;
   s.erase(1); // 删除s中值1这个元素
   cout << boolalpha << (s.find(1) != s.end()) << endl;
   return 0;
}
```
s.find(10) != s.end()标识在集合s中查找值10这个元素，当查找到s.end()位置时查找结束。由于s.end()指向集合s中最后一个元素的下一个元素所在位置(该位置值为空)，所以此时即未查找到值10。
# C++STL：映射map的使用
map属于关联容器，map 中的元素是一些键值对。map中的键是唯一的。map会自动将所有的键值对按照键从小到大排序，这是由于map内部是使用红黑树实现的(set也是)，在建立映射的过程中会自动实现从小到大的排序。注意如果是字符串到整型的键值对，请使用string而不是char数组。因为char数组作为数组不能作为键值。
用法举例：
```cpp
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
   map<string, int> m; //定义一个空map m,键是string类型,值是int类型
   m["hi"] = 5; //将一个key为"hi",value为1的键值对(key-value)存入map中
   cout << m["hi"] << endl; //打印map中key为"hi"的value,若key不存在,则返回0
   cout << m["world"] << endl;
   m["world"] = 3;
   m[","] = 1;
   for (auto it = m.begin(); it != m.end(); it++) { //迭代器遍历map m,键用it->first获取,值用it->second获取
      cout << it->first << " " << it->second << endl;
   }
   cout << m.begin()->first << " " << m.begin()->second << endl;
   cout << m.rbegin()->first << " " << m.rbegin()->second << endl;
   cout << m.size() << endl; //输出map m的键值对个数
   return 0;
}
```
**注意：**
如果使用set和map做题时发生超时，可尝试使用unordered_map和unordered_set。unordered_map和unordered_set省去了排序，可以缩短代码运行时间。unordered_map在头文件unordered_map 中，unordered_set在头文件unordered_set中。

# C++STL：栈stack的使用
栈的特性是后进先出。要使用栈stack，需要引入stack头文件。.push()方法将元素压入栈，.top()方法取栈顶元素。.pop()方法弹出栈顶元素，.empty()方法检测栈是否为空。
用法举例：
```cpp
#include <iostream>
#include <stack>

using namespace std;

int main() {
   stack<int> s;
   for (int i = 0; i < 6; i++) {
      s.push(i);
   }
   cout << s.top() << endl;
   cout << s.size() << endl;
   s.pop();
   cout << s.top() << endl;
   return 0;
}
```
# C++STL：队列queue的使用
队列的特性是先进先出。要使用队列queue，需要引入queue头文件。.push()方法将元素入队，.pop()方法令队首元素出队，.front()方法和.back()方法分别获得队首元素和队尾元素，.empty()方法判断队列是否为空。
用法举例：
```cpp
#include <iostream>
#include <queue>

using namespace std;

int main() {
   queue<int> q;
   for (int i = 0; i < 6; i++) {
      q.push(i);
   }
   cout << q.front() << " " << q.back() << endl;
   cout << q.size() << endl;
   q.pop();
   cout << q.front() << endl;
   return 0;
}
```
# C++STL：bitset类的使用
C++的bitset类主要用来进行一些位运算，要使用bitset类必须先引入bitset头文件。bitset的每一个元素是整型数值0或1，使用位的方式和数组区别不大，相当于只能存一个位的数组。
用法举例：
```cpp
#include <iostream>
#include <bitset>

using namespace std;

int main() {
   bitset<5> b("11"); //5表示5个二进位，11表示初始化值为11000
   // 多种初始化方法：
   // bitset<5> b; 5个位都为默认值0
   // bitset<5> b(u); u为unsigned int，如果u = 1,则被初始化为10000
   // bitset<5> b(s); s为字符串串，如"1101" -> "10110"
   // bitset<5> b(s, pos, n); 从字符串的s[pos]开始，n位长度
   for (int i = 0; i < 5; i++)
      cout << b[i];
   cout << endl << b.any(); // b中是否存在1
   cout << endl << b.none(); // b中是否不存在1
   cout << endl << b.count(); // b中1的个数
   cout << endl << b.size(); // b中二进制数的个数
   cout << endl << b.test(2); // 下标2的二进制数是否为1
   b.set(4); // 下标4处二进制数值置为1
   b.reset(); // 所有位归零
   b.reset(3); // 下标3处二进制数值置0
   b.flip(); // 所有二进制数取反
   unsigned long a = b.to_ulong(); // b转换为unsigned long类型
   return 0;
}
```
# C++11新特性
## auto声明
即让编译器根据初始值类型直接推断变量的类型。如：
```cpp
auto x = 100; // x是int型变量
auto y = 1.5; // y是double型变量
```
auto声明同样可以用在迭代器中：
```cpp
for (auto it = s.begin(); it != s.end(); it++) {
   cout << *it << " ";
}
```
## to_string方法
to_string方法可以将一个int/float/double等类型的变量转化为string类型的变量。如：
```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
   string s1 = to_string(123); // 将123这个数字转成字符串
   cout << s1 << endl;
   string s2 = to_string(4.5); // 将4.5这个数字转成字符串
   cout << s2 << endl;
   cout << s1 + s2 << endl; // 将s1和s2两个字符串拼接起来并输出
   printf("%s\n", (s1 + s2).c_str()); // 如果想用printf输出string，得加一个.c_str()
   return 0;
}
```
注意上面float型转换后末尾会多出几个0。
## stoi/stod方法
stoi/stod方法可以将字符串string转化为对应的int/double型变量。如：
```cpp
#include <iostream>
#include <string>

using namespace std;

int main() {
   string str = "123";
   int a = stoi(str);
   cout << a << endl;
   str = "123.44";
   double b = stod(str);
   cout << b << endl;
   return 0;
}
```
**stoi遇到非法输入时的情况：**
* 自动截取最前面的数字，直到遇到不是数字为止；
* 如果最前面不是数字，会直接报错。

**stod遇到非法输入时的情况：**

* 自动截取最前面的浮点数，直到遇到不满足浮点数为止；
* 如果最前面不是数字或小数点，会直接报错；
* 如果最前面时小数点，会自动转换后在前面补0。