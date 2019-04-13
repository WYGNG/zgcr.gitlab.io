---
title: 1-Two Sum
date: 2019-04-13 11:46:47
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
```
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```
# C/C++解法
暴力法即两层遍历。依次按顺序从数组取一个数，然后遍历整个数组看看有无等于target-取出的数的值。
我们这里用哈希表，哈希表的键是数组里的值，哈希表键的值是数组中元素的下标。用一次for循环，每轮取出一个位置的数，检查哈希表中是否有target-取出的数的键，若有，取其对应值(即下标位置)。
注意实现时哈希表对应C++中的unordered_map，注意unordered_map和map的区别。map内部是用红黑树实现的，其键值对按二叉搜索树的形式存储，其键是有序的，即左孩子<根节点<右孩子，寻找键的时间是O(logn)。而unordered_map是单纯的哈希表，其键是无序的，寻找键的时间是O(1)。
```cpp
#include<unordered_map>
#include <vector>
#include <iostream>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   //本题暴力法当然可解,但需要O(n2),我们这里利用hash表来做,只需要O(n)
   //注意题目要求,每个输入target只有一个解,且同一个下标的元素不能用两次
   //但是如果数组中不同的两个下标值相同时,这两个相同值加起来等于target是满足条件的解
   //利用hash表,将nums数组中的值作为键,值下标作为键的值依次存入hash表,存一个就先假设这个值是我们要找的元素之一
   //然后查找hash表中是否有target-nums[i]这个值,如果有,那么这两个元素就是我们要找的元素,否则继续循环
   //这里最好使用unordered_map而不是map
   //map内部实现了一个红黑树，对元素自动排序，因此map内部的所有元素都是有序的，红黑树的每一个节点都代表着map的一个元素
   //对于map进行的查找，删除，添加等一系列的操作都相当于是对红黑树进行的操作
   //map中的元素是按照二叉搜索树(键值:左孩子<根节点<右孩子)存储的,查找时间O(logn)
   //unordered_map内部实现了一个哈希表查找的时间复杂度可达到O(1),其元素的排列顺序是无序的
   //map底层是红黑树实现的，因此它的find函数时间复杂度O(logn)
   //unordered_map底层是哈希表,因此它的find函数时间复杂度O(l)
   vector<int> twoSum(vector<int> &nums, int target) {
      unordered_map<int, int> m;
      for (int i = 0; i < nums.size(); i++) {
         if (m.find(target - nums[i]) != m.end()) {
            return {m[target - nums[i]], i};
         }
         m[nums[i]] = i;
      }
      return {};
   }
};

int main() {
   vector<int> nums = {3, 3};
   int target = 6;
   Solution s;
   vector<int> index = s.twoSum(nums, target);
   for (auto i:index)
      cout << i << endl;
   return 0;
}
```