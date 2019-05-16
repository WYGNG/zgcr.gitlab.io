---
title: CUDA矩阵计算原理和方法
date: 2019-05-15 11:13:53
tags:
- CUDA编程
categories:
- CUDA编程
---

# 基本概念
##  主机（host）
将CPU及系统的内存（内存条）称为主机。
## 设备（device）
将GPU及GPU本身的显示内存称为设备。
## 流式处理器（SP）
流处理器SP（streaming processor,也叫CUDA core）是最基本的处理单元，最后具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。
## 流式多处理器（SM）
多个SP加上其他的一些资源（warp，scheduler，register，shared memory等）组成一个SM（streaming multiprocessor）。也叫GPU大核。SM可以看做GPU的心脏（对比CPU核心），register和sharedmemory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。
每个SM包含的SP数量依据GPU架构而不同，Fermi架构GF100是32个，GF10X是48个，Kepler架构都是192个，Maxwell都是128个。相同架构的GPU包含的SM数量则根据GPU的中高低端来定。在Maxwell架构中，Nvidia已经把SM改叫SMM。在软件逻辑上是所有SP是并行的，但是物理上并不是所有SP都能同时执行计算，因为有些会处于挂起，就绪等其他状态，这与GPU的线程调度有关。
GPU中每个sm都设计成支持数以百计的线程并行执行，并且每个GPU都包含了很多的SM，所以GPU支持成百上千的线程并行执行。当一个kernel启动后，thread会被分配到这些SM中执行。大量的thread可能会被分配到不同的SM，同一个block中的threads必然在同一个SM中并行（SIMT）执行。每个thread拥有它自己的程序计数器和状态寄存器，并且用该线程自己的数据执行指令，这就是所谓的Single Instruction Multiple Thread。 
一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。Nvidia把32个threads组成一个warp，warp是调度和运行的基本单元。warp中所有threads并行的执行相同的指令。一个warp需要占用一个SM运行，多个warps需要轮流进入SM。由SM的硬件warp scheduler负责调度。目前每个warp包含32个threads（Nvidia保留修改数量的权利）。所以，一个GPU上resident thread最多只有SMxwarp个。 
## 线程（Thread）
一般通过GPU的一个核进行处理。
## 线程块（Block）
由多个线程组成（可以表示成一维，二维，三维）；
各block是并行执行的，block间无法通信，也没有执行顺序；
线程块的数量限制为不超过65535（硬件限制）。
## 线程格（Grid）
由多个线程块组成（可以表示成一维，二维，三维）。
## 线程束（wrap）
在CUDA架构中，线程束是指一个包含32个线程的集合，这个线程集合被“编织在一起”并且“步调一致”的形式执行。在程序中的每一行，线程束中的每个线程都将在不同数据上执行相同的命令。
## 函数修饰符
**在CUDA中，通过函数类型修饰符区分host和device上的函数:**
\_\_global\_\_：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是void，不支持可变参数参数，不能成为类成员函数。注意用\_\_global\_\_定义的kernel是异步的，这意味着host不会等待kernel执行完就执行下一步。
\_\_device\_\_：在device上执行，单仅可以从device中调用，不可以和\_\_global\_\_同时用。
\_\_host\_\_：在host上执行，仅可以从host上调用，一般省略不写，不可以和\_\_global\_\_同时用，但可和\_\_device\_\_，此时函数会在device和host都编译。
# GPU内存的分类
## 全局内存（Global Memory）
通俗意义上的设备内存。
## 共享内存（Shared Memory）
在设备内存上，以关键字\_\_shared\_\_添加到变量声明中。如\_\_shared\_\_ float cache[10]。对于GPU上启动的每个线程块，CUDA C编译器都将创建该共享变量的一个副本。线程块中的每个线程都共享这块内存，但线程却无法看到也不能修改其他线程块的变量副本。这样使得一个线程块中的多个线程能够在计算上通信和协作。
## 常量内存（Constant Memory）
在设备内存上，以关键字\_\_constant\_\_添加到变量声明中。如\_\_constant\_\_ float s[10];。常量内存定义的变量用于保存在核函数执行期间不会发生变化的数据。变量的访问限制为只读。NVIDIA硬件提供了64KB的常量内存。不再需要cudaMalloc()或者cudaFree(),而是在编译时，静态地分配空间。
常量内存其实只是全局内存的一种虚拟地址形式，并没有特殊保留的常量内存块。常量内存有两个特性，一个是高速缓存，另一个是它支持将单个值广播到线程束中的每个线程。当常量内存将数据分配或广播到线程束中的每个线程时（注意，实际上硬件会将单次内存读取操作广播到半个线程束），广播能够在单个周期内发生。当所有16个线程都读取相同地址时，这个功能可以极大提高性能，但当所有16个线程分别读取不同的地址时，它实际上会降低性能。如果半个线程束中的所有16个线程需要访问常量内存中的不同数据，那么这个16次不同的读取操作会被串行化，从而需要16倍的时间来发出请求。但如果从全局内存中读取，那么这些请求就会同时发出。这种情况下，从常量内存读取就会慢于从全局内存中读取。
**注意:**
当我们需要拷贝数据到常量内存中应该使用cudaMemcpyToSymbol()，而cudaMemcpy()会复制到全局内存。
## 纹理内存（Texture Memory）
纹理内存是另一种类型的只读内存，在特定的访问模式中（以下例子并非这种特定的访问模式），纹理内存同样能够提升性能。纹理内存缓存在芯片上，因此在某些情况中，它能够减少对内存的请求并提供更高效的内存带宽。纹理缓存是专门为那些在内存访问模式中存在大量空间局部性(Spatial Locality)的图形应用程序而设计的。在某个计算应用程序中，这意味着一个线程读取的位置可能与邻近线程的读取位置“非常接近”。举个例子，一个2x2矩阵的四个元素在地址上不是连续的，但是在空间位置上是互相相邻的，纹理缓存就是专门为了加速这种访问模式而设计的。如果在这种情况中使用纹理内存而不是全局内存，那么将会获得性能的提升。
纹理变量（引用）必须声明为文件作用域内的全局变量，其形式分为一维纹理内存和二维纹理内存。
**一维纹理内存:**
用texture<类型>类型声明，如texture<float> texIn。通过cudaBindTexture()绑定到纹理内存中，通过tex1Dfetch()来读取纹理内存中的数据，通过cudaUnbindTexture()取消绑定纹理内存。
**二维纹理内存**
用texture<类型,数字>类型声明，如texture<float，2> texIn。通过cudaBindTexture2D()绑定到纹理内存中，通过tex2D()来读取纹理内存中的数据，通过cudaUnbindTexture()取消绑定纹理内存。
## 固定内存
在主机内存上，也称为页锁定内存或者不可分页内存，操作系统将不会对这块内存分页并交换到磁盘上，从而确保了该内存始终驻留在物理内存中。因此操作系统能够安全地使某个应用程序访问该内存的物理地址，因为这块内存将不会破坏或者重新定位。
**优点:**
固定内存可以提高访问速度。由于GPU知道主机内存的物理地址，因此可以通过DMA（直接内存访问，Direct Memory Access)技术来在GPU和主机之间复制数据。由于DMA在执行复制时无需CPU介入。因此DMA复制过程中使用固定内存是非常重要的。
**缺点:**
使用固定内存，将失去虚拟内存的所有功能，系统将更快的耗尽内存。
**注意:**
对cudaMemcpy()函数调用中的源内存或者目标内存，才使用固定内存，并且在不再需要使用它们时立即释放。固定内存通过cudaHostAlloc()函数来分配；通过cudaFreeHost()释放。我们只能以异步方式对固定内存进行复制操作。
# CUDA程序计算原理
## CUDA程序执行过程
分配host内存，并进行数据初始化；
分配device内存，并从host将数据拷贝到device上；
调用CUDA的核函数在device上完成指定的运算；
将device上的运算结果拷贝到host上；
释放device和host上分配的内存。

## 核函数（kernel）与SM
kernel核函数是CUDA中一个重要的概念，kernel核函数是在device上线程中并行执行的函数，用\_\_global\_\_符号声明，在调用时需要用<<<grid, block>>>来指定一个kernel函数要执行的线程数量，在CUDA中，每个线程会分配一个唯一的线程号thread ID，这个ID值可以通过核函数的内置变量threadIdx来获得。
一个线程需要两个内置的坐标变量（blockIdx，threadIdx）来唯一标识，它们都是dim3类型变量，其中blockIdx指明线程所在grid中的位置，而threaIdx指明线程所在block中的位置。
**每个线程有自己的私有本地内存（Local Memory），每个线程块有包含共享内存（Shared Memory）,可以被线程块中所有线程共享，其生命周期与线程块一致。此外，所有的线程都可以访问全局内存（Global Memory），还可以访问一些只读内存块：常量内存（Constant Memory）和纹理内存（Texture Memory）。**
**一个kernel核函数在device上执行时实际上启动了很多线程，一个kernel所启动的所有线程称为一个线程格（grid），同一个线程格上的线程共享相同的全局内存空间；一个线程格又分为很多线程块（block），一个线程块里面包含很多线程。**
一个kernel核函数执行时会启动很多线程，这些线程是逻辑上并行的，但是在物理层上却不一定并行。但是一个GPU中存在很多CUDA核心（即SM），充分利用CUDA核心可以充分发挥GPU的并行计算能力。SM的核心组件包括CUDA核心，共享内存，寄存器等，SM可以并发地执行数百个线程，并发能力就取决于SM所拥有的资源数。
**当一个kernel核函数被执行时，它的网格块（grid）中的线程块（block）被分配到SM上，一个线程块（block）只能在一个SM上被调度。有时一个kernel核函数的各个线程块（block）被分配多个SM，那么网格块（grid）只是逻辑层，而SM才是执行的物理层。**
SM采用的是SIMT（Single-Instruction, Multiple-Thread，单指令多线程）架构，基本的执行单元是线程束（wraps），线程束包含32个线程，这些线程同时执行相同的指令，但是每个线程都包含自己的指令地址计数器和寄存器状态，也有自己独立的执行路径。所以尽管线程束（wraps）中的线程同时从同一程序地址执行，但是可能具有不同的行为，比如遇到了分支结构，一些线程可能进入这个分支，但是另外一些有可能不执行，它们只能死等，因为GPU规定线程束（wraps）中所有线程在同一周期执行相同的指令，线程束（wraps）分化会导致性能下降。
当线程块（block）被划分到某个SM上时，它将进一步划分为多个线程束（wraps），因为这才是SM的基本执行单元，但是一个SM同时并发的线程束（wraps）数是有限的。这是因为资源限制，SM要为每个线程块分配共享内存，而也要为每个线程束（wraps）中的线程分配独立的寄存器。所以SM的配置会影响其所支持的线程块和线程束并发数量。
**从逻辑上划分，一个网格块（grid）包含多个线程块（block），一个线程块（block）包含有多个线程（Threads）。但是一个kernel核函数的所有线程在物理层不一定是同时并发的。因此，kernel核函数的网格块（grid）和线程块（block）的配置不同，性能会出现差异。另外，由于SM的基本执行单元是包含32个线程的线程束，所以线程块（block）大小一般要设置为32的倍数。**
## 核函数（kernel）的调用
**在VS2017的CUDA项目中启动kernel函数时要指定gridsize和blocksize，如:**
```
 dim3 gridsize(2,2);
 dim3 blocksize(4,4);
Kernel << < gridSize, blocksize >> > (A, B, C); 
```
这里的grid和block都是2D的。
**gridsize相当于是一个2x2的block，gridDim.x，gridDim.y，gridDim.z相当于这个dim3的x，y，z方向的维度，这里是2x2x1。序号从0到3，且是从上到下的顺序，即grid中的blockidx序号标注情况为:**
```
0     2
1     3 
```
**blocksize则是指block里面的线程(thread)的情况，blockDim.x，blockDim.y，blockDim.z相当于这个dim3的x，y，z方向的维度，这里是4x4x1.序号是0-15，即block中的threadidx序号标注情况为:**
```
0      4       8      12 
1       5       9       13
2       6       10     14
3       7       11      15
```
**确定线程的global ID:**
```
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;
```
**ThreadID是线性增长的，其目的是用于在硬件和软件上唯一标识每一个线程。CUDA程序中任何一个时刻，每一个线程的ThreadIdx都是特定唯一标识的。Threads的唯一标识ThreadIdx的表达方式随着grid，block的划分维度而不同**。
## 线程同步
线程同步是针对同一个线程块（block）中的所有线程而言的，因为只有同一个线程块（block）中的线程才能在有效的机制中共同访问共享内存（Shared Memory）。由于每一个线程（Thread）的生命周期长度是不相同的，线程（Thread）对共享内存（Shared Memory）的操作可能会导致读写的不一致，因此需要线程的同步，从而保证该block中所有线程同时结束。**
# win10+VS2017+CUDA10.0项目配置
请先安装VS2017，一定要在安装CUDA前安装。
首先从这里下载CUDA10.0:https://developer.nvidia.com/cuda-10.0-download-archive 。按默认选项安装。然后下载CUDNN7.4.2 for CUDA10:https://developer.nvidia.com/rdp/cudnn-archive 。解压后复制到CUDA安装文件夹里。
在安CUDA装过程中，会自动检测本机是否已经安装了配套的VS版本其中之一，如果VS版本和Cuda版本不匹配的话，安装无法进行。
CUDA安装完成后在系统变量中应当会有下面两个变量，如果没有请自己添加上:
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
CUDA_PATH_V10_0 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
```
我们还要在系统变量中添加一个CUDNN变量:
```
CUDNN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
```
我们还需要在用户变量中添加下列变量:
```
CUDA_SDK_PATH = C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.0
CUDA_LIB_PATH = %CUDA_PATH%\lib\x64
CUDA_BIN_PATH = %CUDA_PATH%\bin
CUDA_SDK_BIN_PATH = %CUDA_SDK_PATH%\bin\win64
CUDA_SDK_LIB_PATH = %CUDA_SDK_PATH%\common\lib\x64
```
添加完成后打开cmd，使用下列命令测试CUDA是否正常:
```
nvcc -V //正常显示版本号则说明安装成功
set cuda //可以查看设置的cuda环境变量
cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\demo_suite
deviceQuery.exe
bandwidthTest.exe
// 上面两行命令都返回Rsult=PASS则说明通过，返回Rsult=Fail则需要重新安装
```
打开VS2017，新建一个CUDA10.0 Runtime项目。新建项目时选择NVIDIA->CUDA 10.0->CUDA 10.0 Runtime。
在kernel.cu中添加下列代码：
```cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++) {
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		cudaDriverGetVersion(&driver_version);
		printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
		printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
		printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
		printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
		printf("Warp size:                                      %d\n", deviceProp.warpSize);
		printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
		printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
		printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
		printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
	}
	return 0;
}
```
使用生成->重新生成(项目名)，调试->开始调试，运行结果如下:
```
Device0:"GeForce GTX 1070 with Max-Q Design"
CUDA驱动版本:                                   10.1
CUDA运行时版本:                                 10.0
设备计算能力:                                   6.0
Total amount of Global Memory:                  0 bytes
Number of SMs:                                  16
Total amount of Constant Memory:                65536 bytes
Total amount of Shared Memory per block:        49152 bytes
Total number of registers available per block:  65536
Warp size:                                      32
Maximum number of threads per SM:               2048
Maximum number of threads per block:            1024
Maximum size of each dimension of a block:      1024 x 1024 x 64
Maximum size of each dimension of a grid:       2147483647 x 65535 x 65535
Maximum memory pitch:                           2147483647 bytes
Texture alignmemt:                              32 bytes
Clock rate:                                     1.27 GHz
Memory Clock rate:                              4004 MHz
Memory Bus Width:                               256-bit
```
# CUDA常用函数介绍
## cudaMalloc()函数
函数原型:
```
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
这个函数和C语言中的malloc类似，但是该函数是在device上申请一定字节大小的显存，devPtr是指向所分配内存的指针。可以将cudaMalloc()分配的指针传递给在设备/主机上执行的函数，也可以在设备代码中使用cudaMalloc()分配的指针进行设备内存读写操作。注意不可以在主机代码中使用cudaMalloc()分配的指针进行主机内存读写操作（即不能进行解引用）。
## cudaFree()函数
函数原型:
```
cudaError_t cudaFree (void* devPtr);
```
与c语言中的free()函数一样，只是此函数释放的是cudaMalloc()分配的内存。
## cudaMemcpy()函数
函数原型:
```
cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind);
```
与c语言中的memcpy函数一样，只是此函数可以在主机内存和GPU内存之间互相拷贝数据。cudaMemcpyKind kind表示数据拷贝方向，若kind赋值为cudaMemcpyDeviceToHost表示数据从设备内存拷贝到主机内存。
该函数以同步方式执行，即当函数返回时，复制操作就已经完成了，并且在输出缓冲区中包含了复制进去的内容。相应的有个异步方式执行函数cudaMemcpyAsync()。
##  cudaMallocManaged()函数
函数原型:
```
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flag=0);
```
在最初的CUDA编程中，我们往往单独在host和device上进行内存分配，并且要进行将数据从host拷贝到device上，这很容易出错的。CUDA 6.0版本之后引入统一内存（Unified Memory）来避免这种麻烦，简单来说就是使用一个托管内存来共同管理host和device中的内存，并且自动在host和device中进行数据传输。CUDA中使用cudaMallocManaged()函数分配托管内存。
**注意:**
kernel核函数的执行是与host异步的，我们要在执行完kernel核函数后用cudaDeviceSynchronize()函数保证device和host同步，这样后面才可以正确访问kernel计算的结果。
# CUDA矩阵乘法实例
我们要实现两个矩阵的乘法，设输入矩阵为A和B，要得到 C=AxB 。实现思路是每个线程计算C的一个元素值Cij，对于矩阵运算，应该选用grid和block为2-D。
**计算步骤:**
分配host内存，并进行数据初始化，分配device内存，并从host将数据拷贝到device上，实际代码中使用cudaMallocManaged()函数进行内存托管；
调用CUDA的kernel核函数在device上完成指定的运算；
同步device上的运算结果到host上，使用cudaDeviceSynchronize()函数来同步；
释放device和host上分配的内存，这步由前面定义的cudaMallocManaged()函数自动管理。
**完整代码:**
```cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

// 打印显卡各项信息
void GetCudaImformation() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++) {
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				cout << endl;
		cout << "使用GPU device " << dev << ": " << deviceProp.name << endl;
		cudaDriverGetVersion(&driver_version);
		cout << "CUDA驱动版本:" << driver_version / 1000 << "." << (driver_version % 1000) / 10 << endl;
		cudaRuntimeGetVersion(&runtime_version);
		cout << "CUDA运行时版本:" << runtime_version / 1000 << "." << (runtime_version % 1000) / 10 << endl;
		cout << "设备计算能力:" << deviceProp.major << "." << deviceProp.minor << endl;
		cout << "显卡时钟频率:" << deviceProp.clockRate * 1e-6f << " GHz" << endl;
		cout << "内存时钟频率:" << deviceProp.memoryClockRate * 1e-3f << " MHz" << endl;
		cout << "内存总线带宽:" << deviceProp.memoryBusWidth << " bit" << endl;
		cout << "总显存大小:" << deviceProp.totalGlobalMem / (1024.0*1024.0) << " MB" << endl;
		cout << "总常量内存大小:" << deviceProp.totalConstMem / 1024.0 << " KB" << endl;
		cout << "SM数量:" << deviceProp.multiProcessorCount << endl;
		cout << "每个SM最大线程数:" << deviceProp.maxThreadsPerMultiProcessor << endl;
		cout << "每个线程块(block)共享内存大小:" << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
		cout << "每个线程块(block)的最大线程数:" << deviceProp.maxThreadsPerBlock << endl;
		cout << "每个线程块(block)的最大可用寄存器数:" << deviceProp.regsPerBlock << endl;
		cout << "线程束(wrap)尺寸:" << deviceProp.warpSize << endl;
		cout << "每个线程块(block)各个维度最大尺寸:" << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl;
		cout << "每个线程格(grid)各个维度最大尺寸" << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << endl;
		cout << "最大存储间距:" << deviceProp.memPitch / (1024.0*1024.0) << " MB" << endl;
	}
}

// 矩阵类型,行优先,M(row, col) = *(M.elements + row * M.width + col)
struct Matrix {
	int width;
	int height;
	float *elements;
};

// 获取矩阵A的位置为(row, col)元素
__device__ float getElement(Matrix *A, int row, int col) {
	return A->elements[row * A->width + col];
}

// 为矩阵A的位置为(row, col)的元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value) {
	A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel函数,2-D,每个线程计算一个元素Cij
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C) {
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i) {
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, Cvalue);
}

int main() {
	GetCudaImformation();
	int width = 1 << 10;
	int height = 1 << 10;
	Matrix *A, *B, *C;
	// 申请托管内存
	cudaMallocManaged((void**)&A, sizeof(Matrix));
	cudaMallocManaged((void**)&B, sizeof(Matrix));
	cudaMallocManaged((void**)&C, sizeof(Matrix));
	int nBytes = width * height * sizeof(float);
	cudaMallocManaged((void**)&A->elements, nBytes);
	cudaMallocManaged((void**)&B->elements, nBytes);
	cudaMallocManaged((void**)&C->elements, nBytes);
	// 初始化A\B\C矩阵的宽度和高度
	A->height = height;
	A->width = width;
	B->height = height;
	B->width = width;
	C->height = height;
	C->width = width;
	// 初始化A矩阵所有元素为1.0,B矩阵所有元素为2.0
	for (int i = 0; i < width * height; ++i) {
		A->elements[i] = 1.0;
		B->elements[i] = 2.0;
	}
	// 定义kernel的blocksize为(32, 32)，那么grid大小为(32, 32)
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);
	// 执行kernel
	matMulKernel << < gridSize, blockSize >> > (A, B, C);
	// 同步device数据保证结果能正确访问
	cudaDeviceSynchronize();
	// 检查执行结果
	float maxError = 0.0;
	for (int i = 0; i < width * height; ++i)
		maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
	cout << endl;
	cout << "最大误差: " << maxError << endl;
	return 0;
}
```