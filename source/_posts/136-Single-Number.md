---
title: 136-Single Number
date: 2019-04-07 21:50:00
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:
```
Input: [2,2,1]
Output: 1
```

Example 2:
```
Input: [4,1,2,1,2]
Output: 4
```
# C/C++解法
```cpp
# include <string>
# include <iostream>
# include <vector>
# include <queue>

using namespace std;

struct TreeNode {
   int val;
   TreeNode *left;
   TreeNode *right;

   TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   //异或是相同为0，不同为1  
   //如果我们把两个相同的数字异或，0与0异或是0,1与1异或也是0，那么我们会得到0。而且异或满足交换律,那么只出现一次的数字在哪个位置都不影响
   //根据这个特点，我们把数组中所有的数字都作异或乘法，则每对相同的数字都会得0，然后最后剩下来的数字就是那个只出现1次的数字。
   int singleNumber(vector<int> &nums) {
      int res = 0;
      for (int num : nums) {
         res ^= num;
      }
      return res;
   }

};

int main() {
   // vector数组中按完全二叉树层次遍历的顺序来排列元素,中间没有元素的节点令其元素为0,建立树时这些节点不会建立
   vector<int> a = {2, 2, 1};
   vector<int> b = {4, 1, 2, 1, 2};
   Solution s;
   int single_number_a = s.singleNumber(a);
   int single_number_b = s.singleNumber(b);
   cout << single_number_a << " " << single_number_b << endl;
   return 0;
}
```