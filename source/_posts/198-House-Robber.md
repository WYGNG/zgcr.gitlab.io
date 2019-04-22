---
title: 198-House Robber
date: 2019-04-22 13:09:58
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:
```
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
```
Example 2:
```
Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.
```
# C/C++解法
一道典型的动态规划题目。做动态规划编程题时因为我们不需要给出严密的递推公式证明，我们可以从最简单的情况开始考虑，推断出递推公式。该题要求n个数的数组的连续不相邻元素的最大值。先考虑数组中只有一个元素，那最大值就是该元素；如果有两个元素，则最大值是两个元素中较大的值；如果有三个元素，就有两种可能，即要么是第二个元素，要么是第一个元素加第三个元素的和。于是我们就可以知道其递推公式是这样:f(3)=max(f(2),f(1)+nums[2])。故推导到第n项的通项公式为:f(n) = max(f(n-1), f(n-2)+A[n-1])。
```cpp
# include <iostream>
# include <vector>
# include <queue>

using namespace std;

//若想要提交到leetcode，只需提交class Solution
class Solution {
public:
   //对n个数的数组，如果要求得其连续不相邻元素的最大值，我们只需求得n-1个数的最大值，以及求得n-2个数的最大值
   //这样就形成了求解该问题的子问题的最大值问题
   //这种问题推测递推公式可以从最简单的情形开始推起:
   //只有1个房屋nums[0]，最大收益为dp[0] = nums[0];
   //有2个房屋nums[0], nums[1], 不能同时取，最大收益为dp[1] = max(nums[0], nums[1]);
   //有3个房屋，有两种取法，取nums[1],或者取nums[0]和nums[2].即 dp[2] = max(nums[1], nums[0] + nums[2]);
   //故递推公式为f(n) = max{f(n-1), f(n-2)+A[n-1]}
   //f(n-1)为n-1个元素的最大值，f(n-2)+Arr[n-1]为n-2个元素的最大值加上数组第n个元素的值，因为要求元素不能相邻，所以会跳过第n-1个元素
   int rob(vector<int>& nums) {
      if(nums.empty())
         return 0;
      if(nums.size()==1)
         return nums[0];
      if(nums.size()==2)
         return max(nums[0],nums[1]);
      int sum1=nums[0],sum2=max(nums[0],nums[1]);
      int temp;
      for(int i=2;i<nums.size();i++){
         if(sum2<sum1+nums[i]){
            temp=sum2;
            sum2=sum1+nums[i];
            sum1=temp;
         }
         else
            sum1=sum2;
      }
      return sum2;
   }
};


int main() {
   vector<int> a1 = {1,2,3,1};
   vector<int> a2 = {2,7,9,3,1};
   Solution s;
   int count=s.rob(a1);
   cout<<count;
   return 0;
}
```