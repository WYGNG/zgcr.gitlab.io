---
title: 53-Maximum Subarray
date: 2019-04-12 10:38:21
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:
```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```

Follow up:

If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
# C/C++解法
给出一个数组，找出数组中元素和最大的子数组。本题有三种思路:
* Kadane算法。kadane算法利用了数学归纳法思想。对于整个数组求其最大和的子数组，我们可以从首个元素开始扩充子数组，始终对其最大子数组和保持跟踪，就可以求出整个数组的最大子数组和。即从元素只有一个的子数组开始，往一个长度为i的数组后面插入第i+1个元素，这时，数组的最大子数组只有两种情况，要么包括第i+1个元素，要么不包括第i+1个元素。换个角度来说，最大子数组要么是以第i个数结尾的子列作为前缀，要么它不以之作为前缀(即只有第i+1个元素)。这样我们就可以一步一步求出整个数组的最大子数组和。
* 动态规划法。循环中每次比较当前位置元素值与前一步的局部最优加当前位置的值之和，如果前一步局部最优<0，那么取当前位置值为temp(新子数组开头第一个元素)。思路上和Kadane算法比较相似。
* 分治法。将数组均分为两个部分，那么最大子数组会存在于:左半子数组中的最大子数组、右半子数组中的最大子数组、包含mid元素同时包括左半子数组中一部分元素和右半子数组中一部分元素的最大子数组。假设数组下标有效范围是left到right，将数组分为左半部分下标为left，mid-1和右半部分下标为mid+1，right以及中间元素下标为mid。接下来递归求出左半部分的最大子序和和右半部分最大子序和，再求包含mid元素同时包括左半子数组中一部分元素和右半子数组中一部分元素的最大子数组:先从mid向左遍历,找到左半子数组中的最大子数组的部分,再从mid向右遍历,加上右半子数组中的最大子数组部分。最后对上面三种情况的值进行比较，取最大值返回。
```cpp
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   int maxSubArray(vector<int> &nums) {
      //Kadane算法
        //最大子片段中不可能包含求和值为负的前缀,如-2,1,4必然不是最大子数列, 因为去掉值为负的前缀-2,1之后，可以得到一个更大的子数列4
        //在遍历过程中，每当累加结果成为一个非正值时，就应当将下一个元素作为潜在最大子数列的起始元素，重新开始累加
      //累加过程中出现过的最大值都会被记录，且每一个可能成为最大子数列起始元素的位置都会导致新一轮累加
      // 这样就保证了搜索过程的完备性和正确性
      if (nums.empty())
         return 0;
      //temp记录目前的子数组累加和,max记录已经找到的最大子数组累加和,temp初始化为0
      int max = nums[0],temp =0;
      //循环里就是每轮判断最大子数组是前面的子数组+第i个元素还是抛弃前面的子数组,只保留第i个元素
      for (int num : nums) {
         temp = temp + num;
         max = max > temp ? max : temp;
         temp = temp > 0 ? temp : 0;
      }
      return max;
   }
};

// 若要提交到leetcode只需提交class Solution
//class Solution {
//public:
// //动态规划法
// //动态规划法与kadane算法的区别就是直接比较每一步的temp和max哪个大，每一步的temp都存起来
// int maxSubArray(vector<int> &nums) {
//    // temp是局部最优，max_value是全局最优
//    int temp = nums[0], max_value = nums[0];
//    for (int i = 1; i < nums.size(); i++) {
//       //每次比较当前位置元素值与前一步的局部最优加当前位置的值之和，如果前一步局部最优<0，那么取当前位置值为temp
//       //相当于temp重新取了一个子数组，该子数组的开头就是当前位置
//       //类比上面算法,其实本质一样,都是temp<0时就抛弃当前子数组换到下一个位置取一个新子数组
//       temp = max(nums[i], nums[i] + temp);
//       max_value = max(max_value, temp);
//    }
//    return max_value;
// }
//};

// 若要提交到leetcode只需提交class Solution
//class Solution {
//public:
// //分治法
// //将数组均分为两个部分，那么最大子数组会存在于:左半子数组中的最大子数组、右半子数组中的最大子数组
// //包含mid元素同时包括左半子数组中一部分元素和右半子数组中一部分元素的最大子数组
// //假设数组下标有效范围是l到r,将数组分为左半部分下标为l，mid-1和右半部分下标为mid+1，r以及中间元素下标为mid
// //接下来递归求出左半部分的最大子序和和右半部分最大子序和
// //再求包含mid元素同时包括左半子数组中一部分元素和右半子数组中一部分元素的最大子数组
// //先从mid向左遍历,找到左半子数组中的最大子数组的部分,再从mid向右遍历,加上右半子数组中的最大子数组部分
// int maxSubArray(vector<int> &nums) {
//    if (nums.empty())
//       return 0;
//    return divide_and_conquer(nums, 0, nums.size() - 1);
// }
//
// int divide_and_conquer(vector<int> &nums, int left, int right) {
//    //注意如果划分到后面成了空集时要返回INT32_MIN,不能返回0,因为右子集可能不为空,右子集结果可能为负数
//    if (left > right)
//       return INT32_MIN;
//    if (left == right)
//       return nums[left];
//    int mid = (left + right) / 2;
//    int left_max = divide_and_conquer(nums, left, mid - 1);
//    int right_max = divide_and_conquer(nums, mid + 1, right);
//    int temp = nums[mid], max_value = nums[mid];
//    for (int i = mid - 1; i >= left; i--) {
//       //temp如果加了一个负数,则max_value还是取上一轮的值,temp继续加前面的值,除非遇到前面有更大的正值加起来,否则max_value一直不更新
//       //可以用例子数组-2，5，-3，4，-1，2，1,mid元素为4,以左半子数组为例
//       temp += nums[i];
//       max_value = max(max_value, temp);
//    }
//    temp = max_value;
//    for (int j = mid + 1; j <= right; j++) {
//       temp += nums[j];
//       max_value = max(max_value, temp);
//    }
//    return max(max(left_max, right_max), max_value);
// }
//};

int main() {
   vector<int> a = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
   Solution s;
   int max_sub_sum = s.maxSubArray(a);
   printf("%d ", max_sub_sum);
   return 0;
}
```