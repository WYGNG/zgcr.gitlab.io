---
title: Java基础知识介绍
date: 2019-02-22 21:09:08
tags:
- Java
categories:
- Java
---

# Java语言介绍
Java语言的特点：面向对象的跨平台编程语言；以字节码方式运行在虚拟机上；自带功能齐全的类库；非常活跃的开源社区支持。
Java ME、Java SE、Java EE之间的关系：Java ME： Micro Edition；Java SE：Standard Edition；Java EE：Enterprise Edition。SE包含ME，EE包含SE。
JDK：包含编译器和其他开发工具、JVM、Runtime Library。
JRE：JVM、Runtime Library。
# Java hello world
Java编译运行的过程：源代码(hello.java)->字节码(hello.class)->在JVM运行。
使用IDEA创建一个project，然后在src文件夹下新建一个Java class，命名为Hello。
```
public class Hello { 
	public static void main(String[] args) { 
		System.out.println("hello world!"); 
	}
}
```
我们建立的.java文件的名称要与代码中的public类名完全一致。public static void main(String[] args)是主方法，类似C语言中的main()。String[]代表程序可以接收字符串数组作为参数，args是arguments的缩写，是参数的名字。
我们可以直接Ctrl+shift+F10在IDEA中编译运行该.java文件，也可以在cmd中输入javac Hello.java再输入java Hello来编译并运行（cmd中目录要先cd到Hello.java文件所在的目录）
# Java程序基本结构
```
public class 类名{
	public static void 方法名（参数）{
		......
	}
}
```
public表示该类是公类，不写的话也能编译，但无法从命令行运行该类。类名必须是英文字母开头，多个单词构成时每个单词首字母大写，多个单词直接连在一起，如NoteBook。static表示是一个静态方法，方法名首字母要求小写，多个单词构成时第一个单词首字母小写，后面单词首字母大写，多个单词直接连在一起，如goodMorning。
# Java变量/常量/数据类型/短路运算符/三元运算符
变量必须先定义后使用，变量可以被初始化和多次赋值。=是赋值。
常量用final定义，如final double PI=3.14。常量初始化后不可再次复制，常量的名称通常全部大写。
数据类型有：
* 整数类型：byte，short，int，long；
* 浮点类型：float，double；
* 字符类型：char；（注意区分字符类型和字符串类型，两者不同！）
* 布尔类型：boolean。

字符类型与字符串类型的区别：
* 字符类型是基本类型char，在Java中字符类型使用Unicode编码，只要把一个char类型的数据赋值给一个int型变量，得到的值就是这个char类型数据的Unicode编码；
* 字符串类型是引用类型，用" "括起来，字符串是双引号之间的内容。字符串变量是个指针。引用类型的特点是指向而非持有；
* 字符串是不可变的。当我们改变字符串内容时，实际上是创建了一个新的字符串对象，然后让字符串变量指向这个新的字符串对象。

数组类型：数组创建后大小不可变。索引从0开始。数组也是引用类型。我们可以在数组初始化时就指定数组元素。如果是字符串数组，则数组的每个元素都指向一个字符串对象。
对于数字，16进制以0x开头，2进制以0b开头。想对浮点数四舍五入，可在浮点数后加上0.5再进行强制转换。
短路运算符有两种：
* A&&B：如果A已经是false，就不用判断B是true还是false，如果B是个运算式，这个式子也不会进行运算。这就是短路。
* A||B：如果A已经是true，就不用判断B是true还是false，如果B是个运算式，这个式子也不会进行运算。

三元运算符 b ? x : y：如果b为true，计算x并返回值，否则计算y并返回值。
# Java输入/输出/if...else判断/打印数组/数组排序/查看函数方法源码/命令行参数String[] args
输入要引入java.util.Scanner包。
```
import java.util.Scanner
Scanner s = new Scanner(System.in); # 创建一个Scanner对象，控制台会一直等待输入，直到敲回车键结束，把所输入的内容传给Scanner，作为扫描对象。要获取输入的内容，则只需要调用Scanner的nextLine()或nextInt()或nextdouble()等方法。
String name=s.nextLine(); # 读取一个字符串输入，直到回车换行
int age=s.nextInt(); # 或者读取一个int型整数
```
输出方法：
```
System.out.println(); # 输出并换行
System.out.print(); # 输出但不换行
System.out.printf(); # 格式化输出
```
if...else判断格式和C/C++中相同。如：
```
import java.util.Scanner; 

public class test {  
	public static void main(String[] args) { 
		Scanner s=new Scanner(System.in);  
		int n=s.nextInt();  
		if(n>60)   
			System.out.println("n>60");  
		else if(n>80)   
			System.out.println("n>80");  
		else   
			System.out.println("其他情况"); 
	}
}
```
注意：
* 在Java中，浮点数判断相等时，用==判断是不靠谱的。我们可以利用Math.abs(x-0.1)<0.00001这样小于某个临界值即算作相等来判断。
* 如果是引用类型的变量，如字符串变量，==符号判断的是其是否指向同一个字符串对象，而要判断字符串的内容是否相等，要使用equals()函数。如果变量为null，则调用equals()会报错。我们可以通过短路运算符&&来避免这种情况。
* Java中的swtich语句、while循环、do while循环、for循环、break语句、continue语句均与C语言中类似。

Java中打印数组：由于数组变量是引用类型，直接打印数组变量得到的是变量名的地址，要打印数组元素，我们可以用for循环或使用Arrays.toString()方法。
```
import java.util.Arrays; 

public class test {  
	public static void main(String[] args) {  
		int[] ns= {1,2,3,4,5,8};  
		System.out.println(ns);  
		System.out.println(Arrays.toString(ns)); 
	}
}
```
Java中数组排序，以冒泡排序为例：
```
import java.util.Arrays; 

public class test {  
	public static void main(String[] args) {  
		int[] ns= {28,12,89,73,65,18,96,50,8,36};  
		for(int i=0;i<ns.length;i++) {   
			for(int j=i;j<ns.length-1;j++) {    
				if(ns[j]>ns[j+1]) {     
					int temp=ns[j];     
					ns[j]=ns[j+1];     
					ns[j+1]=temp;   
				}  
			} 
		}  
		System.out.println(Arrays.toString(ns)); 
	}
}
```
我们还可以使用Arrays.sort()方法对数组排序，该方法对于元素多的数组进行快速排序，对于元素少的数组进行冒泡排序。
```
import java.util.Arrays; 

public class test {  
	public static void main(String[] args) {  
		int[] ns= {28,12,89,73,65,18,96,50,8,36};  
		Arrays.sort(ns);  
		System.out.println(Arrays.toString(ns)); 
	}
}
```
如何查看函数方法源码：在eclipse/IDEA中，按住CTRL键，点击方法名如sort，就可以查看其方法的源码。
命令行参数String[] args：是一个String[]数组，args为数组变量名，我们可以向这个数组中传入任意数量的参数变量名。命令行参数由JVM虚拟机接收用户的输入并传给Main()方法。
如：我们可以在cmd中javac 文件名.java编译某个.java文件，然后输入java 文件名 -s -t -version。-s -t -version均为传入的参数。检测这些传入的参数变量名，在Main函数中可设置if判断来执行一些操作。
```
public class test {  
	public static void main(String[] args) {  
		System.out.println(args.length);  
		for(String arg:args)   
			System.out.println(arg); 
	}
}
```
# Java对象与实例/方法与数据封装/继承与多态/抽象类与接口/静态字段与方法/包与作用域/classpath与jar/Java核心类
Java对象和实例的关系：class是对象的模板，它定义了如何创建实例，class的名字就是数据类型。一个class里可以有多个字段（field），字段用来描述class的特征；instance是对象的实例，它根据class创建，可以有多个实例，它们的类型相同，但各自的属性可能不同。每个实例都有自己独立的存储空间。使用new操作符就可以创建一个实例。通过变量.字符可以访问或修改实例的字段数据。
方法：什么是方法？方法就是封装了访问实例字段的逻辑。方法是一组执行语句，遇到return返回。void表示不返回任何值（注意区分null）。方法可以让外部代码安全地访问实例字段。
数据封装：一个class里可以有多个字段（field），如果直接把field用public修改，会直接把field暴露给外部，可能会破坏封装。当用private修饰field时，可以拒绝外部访问。此时我们可以在class内定义修改private修饰的field的方法，用public来修饰该方法，这就可以间接设置和获取private修饰的field。通过方法来访问更加安全。
我们通过变量.方法名()来调用实例方法。方法内部可以使用隐式变量this，this指向当前实例，this.field可以访问当前实例的字段。在不引起歧义的情况下，我们可以省略this。在有局部变量名时，局部变量名优先。方法参数是用于接收传递给方法的变量值。可以是基本类型参数，也可以是引用类型参数。
private方法：外部代码不可访问private方法。内部代码可以调用自己的private方法。
方法重载：方法重载即多个方法的方法名相同，但各自的参数个数、类型、位置不同。方法的返回值类型通常是相同的。方法重载的目的是相同功能的方法使用同一名字，便于调用。重载方法主要依靠参数类型和数量区分。注意不要交换参数顺序。
继承：
* Java中所有类都继承自Object类。我们创建一个Person类，然后再创建一个Student类时，就可以让Student类继承Person类。需要注意的是，Person类定义的private字段无法被子类访问。而Person类中用protected修饰的字段可以被子类访问。Java只允许class继承一个类，即一个类有且仅有一个父类（Object类除外）。
* 在Java中，子类的构造方法第一句必须调用父类的构造方法，即子类的构造方法第一行语句必须是super()。super关键字表示父类（超类）。如果没有写super()，编译器会自动生成super()。如果父类没有默认构造方法，子类则必须显示调用super()，并传入super需要的参数。
* instanceof操作符可以判断某个对象是否是某个类型或这个类型的父类。如果一个变量值为null，则instanceof判断结果始终为false。我们可以在向下转型前先用instanceof判断。

多态：
* 子类中重新定义一个与父类中同名的方法，叫做覆写（Override）。子类调用同名的方法时，如果有覆写，则调用的是本子类覆写后的方法。
* 多态指针对某个类型的方法调用，其真正执行的方法取决于运行时期实际类型的方法。对某个类型调用某个方法，执行的方法可能是某个子类的覆写方法。我们可以加上@Override来让编译器帮助检查是否进行了正确的覆写。如果我们想调用父类中的方法而不是子类中覆写的同名方法，可以使用super.方法名()。

final关键字：用final修饰的方法不能被覆写，用final修饰的类不能被继承，用final修饰的字段在初始化后不能被修改（必须创建对象时就初始化）。final修饰符不是访问权限！！！
抽象类：
* 我们已经知道每个子类都可以覆写父类的方法。如果父类的方法没有实际意义，能否去掉方法的执行语句？答案是不能的。但我们可以把父类的方法声明为抽象方法，使用abstract关键字。
* 如果一个class定义了方法，但没有具体的执行代码，这个方法就是抽象方法。抽象方法用abstract修饰，抽象方法也没有任何执行语句。因为无法执行抽象方法，所以这个类也必须申明为抽象类。我们无法实例化一个抽象类。但可以实例化一个抽象类的子类。
* 这个抽象类用于被继承，抽象类可以强迫子类实现其定义的抽象方法。抽象方法实际上相当于定义了“规范”，不需要子类就可以实现业务逻辑（正常编译），具体的业务逻辑由不同的子类来实现，调用者并不关心。
* 抽象方法定义了子类必须实现的接口规范，定义了抽象方法的类就是抽象类，从抽象类继承的子类必须实现抽象方法，如果不实现抽象方法，则该子类仍然是一个抽象类。

接口：
* 如果一个抽象类没有字段，且所有方法都是抽象方法，我们就可以把该抽象类改写为接口（interface）。
* 在Java中，我们使用interface来声明一个接口。interface是Java内置的纯抽象接口，实现interface时要使用implements关键字，我们可以实现多个interface。
* 在Java中，接口特指interface定义的接口，只定义方法签名。在interface中定义的default方法，在子类中就可以不用实现这个default方法而不会产生编译错误。
* 一个interface也可以继承自另一个interface，interface继承自interface使用extends关键字。这相当于扩展了接口的方法。
* 我们需要合理设计interface和abstract class的继承关系。一般我们把公共逻辑放在abstract class中。接口层次代表抽象的程度。

静态字段：
* 用static修饰的字段称为静态字段。普通字段在每个实例中都有自己的一个独立空间，而静态字段只有一个共享“空间”，所有实例都共享该字段。
* 我们不推荐用实例变量访问静态字段，推荐用类名来访问静态字段。我们可以把静态字段理解为描述class本身的字段（非实例的字段）。

静态方法：
* 用static修饰的方法称为静态方法。调用实例方法必须通过实例变量，而调用静态方法不需要实例变量。静态方法就类似C语言中的函数。
* 静态方法不能访问this变量和实例字段（实例字段实际都是通过this变量来访问的）。
* 静态方法可以访问静态字段和调用其他静态方法。
* 静态方法经常用于工具类Arrays.sort()和Math.random()，静态方法经常用于辅助方法，我们Java程序的入口main()也是静态方法。

包：
* 包用来解决同样名字的类名的冲突，Java定义了名字空间：包。包名+类名=完整类名。
* JVM加载class并执行代码时，总是使用class的完整类名，因此只要包名不同，类就不同。包可以是多层结构，但包没有父子关系。编译器编译后的class文件中全部是完整类名。

包的作用域：位于同一个包的类，可以访问包作用域的字段和方法。不用public、protected、private修饰的字段和方法就是包作用域。
引用其他包的类的两种方法：
* 使用完整的类名；
* 使用import语句先import包，再使用类名。

作用域：
* Java的类、接口、字段和方法都可以设置访问权限。访问权限有public、protected、private和package四种。
* 定义为public的field和method可以被其他类访问。如果不确定是否要public，就不声明public，减少对外暴露方法。
* 定义为private的field和method无法被其他类访问。private访问权限限定在class内部，与方法声明顺序无关。一般推荐把private方法写在public方法的后面。
* private也可以修饰class，定义为private的class无法被其他类访问，访问private class被限定在外层class的内部。定义在一个class内部的class成为内部类（inner class）。
* protected作用于继承关系，定义为protected的字段和方法可以被子类访问。
* package作用域指一个类允许访问同一个package的：没有public、private修饰的class；没有public、protected、private修饰的字段和方法。包没有父子关系，所以包名必须完全一致。

局部变量：在方法内部定义的变量称为局部变量。局部变量作用域从变量声明处开始到对应块结束。在编写代码时，我们应当尽可能把局部变量的作用域缩小，并尽可能延后声明局部变量。

注意：一个.java文件只能包含一个public class，但可以包含多个非public class。
classpath：classpath是一个环境变量，指示JVM如何搜索class。设置的搜索路径与操作系统有关。
JVM在读取了某个完整类名后，按下面的顺序来寻找类：
* 当前目录；
* classpath中记录的目录，从左到右依次寻找；
* 在某个目录下找到则不往后继续寻找；
* 如果都没有找到，则报错；
* 我们可以在系统环境变量中设置classpath环境变量（不推荐），或者在启动JVM时设置classpath变量（推荐），如：
```
java -cp 绝对路径 完整类名
```
jar包：jar包是zip格式的压缩文件，包含若干.class文件。jar包相当于目录。jar包还可以包含其他jar包。classpath中可以包含jar文件。
注意：当JVM运行时会自动加载JDK自带的class，JDK自带的class被打包在rt.jar。rt.jar由JVM直接加载，我们不需要在classpath中引用rt.jar。
**Java核心类：**
String：String一旦创建内容不可变。equals(Object)方法可以比较两个String的内容是否相等。equalsIgnoreCase(String)方法忽略大小写来比较两个字符串的区别。
String常用操作：
* 包含子串：ntains/indexOf/lastIndexOf/startsWith/endsWith；
* 去除首尾空白字符：trim。注意trim()不改变字符串内容，而是返回新字符串；
* 提取子串：substring()；
* 替换子串：replace()/replaceAll()；
* 分割：split()；
* 拼接：join()；
* 任意数据转换为String：static String valueof(int/boolean/Object)；
* String转换为其它类型：static int Integer.parselnt(String)或static Integer Integer.ValueOf(String)。

UTF-8编码是一种变长编码，当是英文字符时，UTF-8编码也是一个字节；当是中文时，往往是3个字节。该编码的容错能力很强。当和byte[]互相转换时要注意编码，建议总是使用UTF-8编码。

StringBuilder：
* String可用+拼接，但每次拼接都会创建一个新的字符串对象，这浪费了内存。StringBuilder是一个可变对象，它可以预先分配缓冲区，故可以高效拼接字符串。StringBuilder支持链式操作。
* 当只有一行代码时，不需要特别改写字符串+操作，因为编译器会在内部自动把多个连续的+操作优化为StringBuilder操作。

包装类型：
* Java数据类型分成基本类型（int等）和引用类型（所有class类型），基本类型不能看成对象。但是我们可以定义一个Integer类，包含一个实例字段int，这样就可以把Integer视为int的包装类型（wrapper）。
* JDK为每种基本类型都创建了对应的包装类型；
* 编译器可以自动在int和Integer之间转型：自动装箱：auto boxing，int->Integer；自动拆箱：auto unboxing，Integer->int。自动装箱和自动拆箱只发生在编译阶段(JDK>=1.5)，装箱和拆箱会影响执行效率。编译后的class代码是严格区分基本类型和引用类型的。
* 包装类型还定义了一些有用的静态变量。
* 整数和浮点数的包装类型继承自Number类。

JavaBean：
* 如果一个class定义为若干的private实例字段，然后通过public方法读写实例字段。符合这种命名规范的class被称为JavaBean。
* 通常把一组对应的getter和setter方法称为属性。只有getter的属性称为只读属性。只有setter的属性称为只写属性。属性只需要定义getter和setter方法，不一定要定义对应的field。

枚举类：
* 以前我们一般用static final来定义常量。这样定义有个缺点，就是编译器无法检查定义的常量的内容。Java提供了一个enum关键字来定义常量类型，常量本身带有类型信息，常量可以用==来比较。
* enum定义的类型实际上是class，继承自java.lang.Enum，我们不能通过new来创建实例，所有常量都是唯一实例（引用类型）。可以用于switch语句。
* name()获取常量定义的字符串，注意不要用toString()；ordinal()返回常量定义的顺序。
* 我们可以为enum类编写构造方法、字段和方法，构造方法申明为private。
# Java反射/注解/泛型
反射：
* Java除基本类型外其他都是class类型(包括interface)。JVM为每个加载的class创建对应的Class实例，并在实例中保存该class的所有信息。如果获取了某个Class实例，则可以获取到该实例对应的class的所有信息。通过Class实例获取class信息的方法称为反射。
* 如何获取一个class的Class的实例？
```
Type.class;
getClass();
Class.forName();
```
* Class实例在JVM中是唯一的。我们可以用==比较两个Class实例。注意这种比较和instance of比较不同。Class实例比较只匹配当前类型，instanceof比较不但匹配当前类型，还匹配当前类型的子类。
* 从Class实例可以获取下列class信息：
```
getName(); # 得完整类名
getSimpleName(); # 获得类名
getPackage(); # 获得包名
```
* 从Class实例还可以判断class类型：
```
isInterface();
isEnum();
isArray();
isPrimitive(); # 判断是不是一个基本类型
```
注解：
* 注解（Annotation）是放在Java源码的类、方法、字段、参数前的一种标签。注解本身对代码逻辑没有任何影响，如何使用注解由工具决定。
* 编译器可以使用的注解：
```
@Override：让编译器检查该方法是否正确实现覆写；
@Deprecated：告诉编译器该方法已经被标记为“作废”，其他地方引用将会出现编译警告；
@SuppressWarnings。
```
* 注解可以定义配置参数：配置参数由注解类型定义；配置参数可以包括：所有基本类型、String、枚举类型、数组；配置参数必须是常量。使用注解时，缺少某个配置参数将使用默认值；如果只写常量，相当于省略了value参数；如果只写注解，相当于全部使用默认值。
* 使用@interface可以定义注解：注解的参数类似无参数方法；可以设定一个默认值（推荐）；建议把最常用的参数命名为value（推荐）。
* 使用@Target定义注解可以被应用于源码的哪些位置：
```
类或接口：ElementType.TYPE
字段：ElementType.FIELD
方法：ElementType.METHOD
构造方法：ElementType.CONSTRUCTOR
方法参数：ElementType.PARAMETER
```
* 使用@Retention定义注解的声明周期：
```
仅编译器：RetentionPolicy.SOURCE # 此时Annotation在编译器编译时直接丢弃
仅class文件：RetentionPolicy.CLASS # 此Annotation仅存储在class文件中
仅运行期：RetentionPolicy.RUNTIME # 在运行期间可以读取该Annotation
如果@Retention不存在，则该Annotation默认为CLASS。通常自定义的Annotation都是RUNTIME。
```
* 使用@Repeatable定义Annotation是否可重复(JDK>=1.8)。
* 使用@Inherited定义子类是否可继承父类定义的注解：仅针对@Target为TYPE类型的Annotation；仅针对class的继承，对interface的继承无效。

定义Annotation的步骤：
* 用@interface定义注解；
* 用元注解（meta annotation）配置注解：Target：必须设置；Retention：一般设置为RUNTIME；通常不必写@Inherited, @Repeatable等等。
* 定义注解参数和默认值。

泛型：
* JDK提供了ArrayList，可以看做“可变长度”的数组，它比数组要方便。但是如果用ArrayList存储String类型，则需要强制转型，不方便也容易出错。我们可以通过单独为String编写一种ArrayList来解决，但是我们还需要为其他所有class单独编写一种ArrayList，这非常麻烦。我们必须把ArrayList变成一种模板：ArrayList<T>,T可以是任意的class，这样就实现了编写一个模板就可以存储各种类型的class。所以，泛型（Generic）就是定义一种模板，例如ArrayList<T>。
* 泛型就是编写模板代码来适应任意类型；不必对类型进行强制转换；编译器将对类型进行检查；注意泛型的继承关系不能变。
* 如何编写一个泛型类：按照某种类型（如String）编写类；标记出所有的特定类型（如String）；把特定类型替换为T，并申明<T>。编写泛型时，需要定义泛型类型<T>：public class Pair<T> { … }。
* 静态方法不能引用泛型类型<T>，这会导致编译错误，编译器无法在静态字段或静态方法中使用泛型类型<T>。我们可以使用另一个类型<K>，就可以把静态方法单独改写为“泛型”方法。即形式：public static <K> Pair<K> create(K first, K last) { … }。泛型可以同时定义多种类型<T, K>。
* Java的泛型是采用擦拭法实现的。在泛型代码编译时，编译器实际上把所有泛型类型<T>统一视为Object类型。编译器根据<T>实现安全的强制转型。要注意<T>不能是基本类型，Object字段无法持有基本类型；无法取得带泛型的Class。
* 所有泛型类型，无论T是什么，返回的都是同一个类型的class。
* 泛型可以继承自泛型类，子类可以获取父类的泛型类型<T>。

# Java集合：List、Map、Set、Collections、Queue、Stack、Iterator
集合：
* 一个Java对象可以在内部持有若干其他Java对象，并对外提供访问接口，我们把这种Java对象称为集合。
* 接口和实现相分离：如List接口实现类有ArrayList和LinkedList等；
* 支持泛型：如List<Student> list=new ArrayList<>()；
* 访问集合有统一的方法：迭代器（Iterator）。

JDK自带的java.util包提供了集合类：
* Collection：所有集合类的根接口；
* List：一种有序列表；
* Set：一种无重复元素集合；
* Map：一种通过Key查找Value的映射表集合；

JDK的下列集合类是遗留类，不应该继续使用：
* Hashtable：一种线程安全的Map实现；
* Vector：一种线程安全的List实现；
* Stack：基于Vector实现的LIFO的栈；