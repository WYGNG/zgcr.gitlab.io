---
title: 70-Climbing Stairs
date: 2019-04-10 15:02:37
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:
```
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

Example 2:
```
Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
3. 1 step + 1 step + 1 step
4. 1 step + 2 steps
5. 2 steps + 1 step
```
# C/C++解法
和求斐波那契数列第n个值几乎完全相同的题目，三种solution对应分治法(递归)->动态规划->动态规划优化空间复杂度为O(1)三种解法。
```cpp
#include <cstdio>
#include <vector>

using namespace std;

// 若要提交到leetcode只需提交class Solution
// 这道题和求斐波那契数列第n个值非常相似，其运用算法的思想可以完全套用到这道题种，即分治法->动态规划->动态规划改进存储空间为O(1)
class Solution {
public:
   // 还是动态规划做法，但我们不需要用数组存储所有结果,只要用两个变量存储下一次计算时需要的两个结果,在每次循环时更新这两个结果即可
   int climbStairs(int n) {
      if(n==1)
         return 1;
      if(n==2)
         return 2;
      int var1=1,var2=2;
      for(int i=0;i<=n-3;i++){
         var2=var1+var2;
         var1=var2-var1;
      }
      return var2;
   }
};


// 若要提交到leetcode只需提交class Solution
//class Solution {
//public:
// // 动态规划做法，从最开始计算的子问题开始依次计算子问题并把结果记录到数组中
// int climbStairs(int n) {
//    //注意边界条件,假如n=1,没有下面两个if,创建数组初始化最后两个元素时会越界
//    if(n==1)
//       return 1;
//    if(n==2)
//       return 2;
//    //数组中每个下标i的含义是从第i个台阶开始上到第n个台阶有多少种方式
//    int record[n];
//    record[n - 1] = 1, record[n - 2] = 2;
//    for (int i = n - 3; i >= 0; i--)
//       record[i] = record[i + 1] + record[i + 2];
//    return record[0];
// }
//};

// 若要提交到leetcode只需提交class Solution
//class Solution {
//public:
// //每次上一步或两步,有多少种方式可以上到第n个台阶
// //i为当前在第i个台阶
// //递归做法，递归做法造成了子问题的大量重复计算
// int climbStairs(int n) {
//    climb_Stairs(0, n);
// }
// // return中i每次加1或加2，然后判断i+1或i+2是否大于等于n,等于n则说明是一种方式，大于则不是
// // 如果i+1或i+2还小于n,则继续递归调用。
// int climb_Stairs(int i, int n) {
//    if (i > n)
//       return 0;
//    if (i == n)
//       return 1;
//    return climb_Stairs(i + 1, n) + climb_Stairs(i + 2, n);
// }
//};

int main() {
   int n = 10;
   Solution s;
   int sum = s.climbStairs(n);
   printf("%d ", sum);
   return 0;
}
```