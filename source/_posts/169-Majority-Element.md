---
title: 169-Majority Element
date: 2019-04-09 09:51:11
tags:
- Leetcode
categories:
- Leetcode
---

# 题目

Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

Example 1:
```
Input: [3,2,3]
Output: 3
```
Example 2:
```
Input: [2,2,1,1,1,2,2]
Output: 2
```
# C/C++解法
```cpp
# include <string>
# include <iostream>
# include <vector>
# include <map>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   //一种高效做法,只需遍历一次,先初始化majority为首个数,flag为0,遍历数组,遇到和majority相同数则flag加1,否则减1
   //如果flag<0了,说明遍历到第i个元素为止时,前i个元素的子数组中majority的数不是majorityElement,则令majority为子数组最后一个元素,flag=1重新开始计数
   //majorityElement的元素数量必须大于nums.size()/2
   //因此这种做法如果在前i个元素时majorityElement的数的数目未达到i/2时
   //后面nums.size()-i个元素的子数组中majorityElement的数目一定超过(nums.size()-i)/2个,故最后一定能取到majorityElement的元素
   int majorityElement(vector<int>& nums){
      int majority = nums[0];
      int flag = 0;
      for(int num : nums){
         if(num == majority)
            flag++;
         else flag--;
         if(flag < 0){
            majority = num;
            flag = 1;
         }
      }
      return majority;
   }
};

//class Solution {
//public:
// //常规做法,用hashmap统计每个值出现的次数,再遍历hashmap的所有键的值,找到最大值对应的键
// int majorityElement(vector<int> &nums) {
//    map<int, int> count_dict;
//    for (auto i:nums) {
//       if (!count_dict.count(i)) {
//          count_dict[i] = 1;
//       } else
//          count_dict[i] += 1;
//    }
//    map<int, int>::iterator it;
//    int majority = 0, max = 0;
//    for (it = count_dict.begin(); it != count_dict.end(); it++) {
//       if (it->second > max) {
//          max = it->second;
//          majority = it->first;
//       }
//       //值如果大于nums.size()/2直接可以确定是majorityElement
//       if(it->second>nums.size()/2)
//          break;
//    }
//    return majority;
// }
//};

int main() {
   vector<int> a = {2,2,1,1,1,3,3,2,2,2};
   Solution s;
   int majority = s.majorityElement(a);
   cout << majority << endl;
   return 0;
}
```