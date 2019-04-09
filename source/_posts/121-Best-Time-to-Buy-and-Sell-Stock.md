---
title: 121-Best Time to Buy and Sell Stock
date: 2019-04-09 13:01:54
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:
```
Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
```

Example 2:
```
Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
```
# C/C++解法
```cpp
# include <iostream>
# include <vector>
# include <queue>
# include <stack>

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
   int maxProfit(vector<int> &prices) {
      int buy = INT32_MAX, max_profit = 0;
      for (int price : prices) {
         //每次迭代可找到最小的buy
         if (price < buy)
            buy = price;
         //如果本次迭代更新了buy,则price-buy为0(找到了更小的buy时)
         //如果没有更新buy,则说明本次迭代的price必定大于buy,这样就保证了buy在卖出价格之前的顺序性
         if (max_profit < price - buy)
            max_profit = price - buy;
      }
      return max_profit;
   }
};

int main() {
   // 建立的数组必须是二叉搜索树,满足左孩子<根<右孩子
   vector<int> a = {7, 1, 5, 3, 6, 4};
   Solution s;
   int max_profit = s.maxProfit(a);
   cout << max_profit << endl;
   return 0;
}
```