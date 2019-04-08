---
title: 283-Move Zeroes
date: 2019-04-08 19:08:14
tags:
- Leetcode
categories:
- Leetcode
---

# 题目

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Example:
```
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
```
Note:
```
You must do this in-place without making a copy of the array.
Minimize the total number of operations.
```

# C/C++解法
非零元素需要保持原有顺序。不能复制数组。我们用一个for循环寻找第一个不为0的数的下标，找到后与j位置交换。j跳到下一个位置。如此循环。j始终在一个全是0的序列的开头第一个0的位置(每次交换都是把非零值与全0序列的第一个0交换，这样整个全0序列就在逐渐后移)。每交换一次，j当前位置变成非零值，所以j要加1。
```cpp
# include <string>
# include <iostream>
# include <vector>
# include <queue>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   void moveZeroes(vector<int> &nums) {
      for(int i=0,j=0;i<nums.size();i++){
         // 找到第一个不为0的数下标i
         if(nums[i]!=0) {
            //如果i不等于j,互换位置,然后j跳到下一个位置
            //如果是初始情况i=j=0,且nums[0]不为0时,不交换,j++
            if (i != j) {
               nums[j] = nums[i];
               nums[i] = 0;
            }
            j++;
         }
      }
   }
};

int main() {
   // 建立的数组必须是完全二叉树,且最后一个节点必须是父节点的右孩子
   vector<int> a = {0, 1, 0, 3, 12};
   for (auto i:a) {
      cout << i << " ";
   }
   cout << endl;
   Solution s;
   s.moveZeroes(a);
   for (auto i:a) {
      cout << i << " ";
   }
   cout << endl;
   return 0;
}
```