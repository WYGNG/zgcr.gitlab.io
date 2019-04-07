---
title: 461. Hamming Distance
date: 2019-04-07 16:21:40
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers x and y, calculate the Hamming distance.

Note:

0 ≤ x, y < 2^31.


Example:
```
Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

The above arrows point to positions where the corresponding bits are different.
```
# C/C++解法
汉明距离即两个整数的二进制表示形式中，相同位置上对应位不同的数量。这里用bitset类将int整数转化位二进制数，然后遍历看同样位置上的数字是否相等即可。
```cpp
# include <string>
# include <iostream>
# include <bitset>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   int hammingDistance(int x, int y) {
      //bitset类是C++中的二进制操作类
      bitset<31> x_b(x);
      bitset<31> y_b(y);
      int hamming_num=0;
      for(int i=0;i<x_b.size();i++){
         if(x_b[i]!=y_b[i])
            hamming_num++;
      }
      return hamming_num;
   }
};

int main() {
   int x=1,y=4;
   Solution s;
   int num_1 = s.hammingDistance(x, y);
   cout << num_1 << endl;
   return 0;
}
```