---
title: 206-Reverse Linked List
date: 2019-04-08 21:46:37
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

Example:
```
Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]
```
# C/C++解法
两次循环。第一次循环每个位置上的值作为下标(请取绝对值再减1，防止已经被变成负数)，查询下标位置的数是否是正数(初始为正数)，是则取其相反数，得一负数，如果已经是负数，则不变。这样我们第一轮遍历下标abs(nums[i]-1)位置都会置该位置上数为负数，那么我们第二轮遍历所有数组时只要发现某个位置为正数，则对应下标的数就没有出现过。
```cpp
# include <string>
# include <iostream>
# include <vector>
# include <queue>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   //这道题让我们找出数组中所有消失的数，跟Find All Duplicates in an Array极其类似，那道题让找出所有重复的数字，这道题让找不存在的数
   // 这类问题的一个重要条件就是1 ≤ a[i] ≤ n (n = size of array)，不然很难在O(1)空间和O(n)时间内完成
   vector<int> findDisappearedNumbers(vector<int> &nums) {
      vector<int> res;
      //对于每个数字nums[i],如果其对应的nums[nums[i] - 1]是正数,我们就赋值为其相反数,如果已经是负数就不变
      //第二次循环把nums[i]中为正数的位置对应的下标加1存入res中即可
      for(int i=0;i<nums.size();i++){
         int index=abs(nums[i])-1;
         if(nums[index]>0)
            nums[index]=-nums[index];
      }
      for(int j=0;j<nums.size();j++){
         if(nums[j]>0)
            res.push_back(j+1);
      }
      return res;
   }
};

int main() {
   vector<int> a = {4, 3, 2, 7, 8, 2, 3, 1};
   Solution s;
   vector<int> res = s.findDisappearedNumbers(a);
   for (int i:res)
      cout << i << " ";
   cout << endl;
   return 0;
}
```

Reverse a singly linked list.

Example:
```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?
# C/C++解法
分为迭代解法和递归解法。迭代解法即另创建一个链表头节点，然后用头插法将head链表首个节点不断插入到新创建的链表头部；递归法思路有点复杂，请仔细看下面代码注释中的详细说明。