---
title: 771-Jewels and Stones
date: 2019-04-07 15:42:16
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

Example 1:
```
Input: J = "aA", S = "aAAbbbb"
Output: 3
```
Example 2:
```
Input: J = "z", S = "ZZ"
Output: 0
```
Note:
* S and J will consist of letters and have length at most 50.
* The characters in J are distinct.
# C/C++解法
最快方法:使用C++的集合，J放入集合中去重，然后遍历S中每一个字符，若在J中则num加1。
```cpp
# include <string>
# include <set>
# include <iostream>

using namespace std;

// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   int numJewelsInStones(string J, string S) {
      //set集合所有元素都会根据元素的键值自动排序，set元素的键值就是实值。set不允许两个元素有相同的键值。
      set<char> setJ(J.begin(), J.end());
      int num = 0;
      for (char s : S) {
         //.count()记录s在setJ中出现一次,在集合中只可能出现1次或0次
         if (setJ.count(s))
            num++;
      }
      return num;
   }
};

int main() {
   string J_1 = "aA";
   string S_1 = "aAAbbbb";
   string J_2 = "z";
   string S_2 = "ZZ";
   Solution s;
   int num_1 = s.numJewelsInStones(J_1, S_1);
   int num_2 = s.numJewelsInStones(J_2, S_2);
   cout << num_1 << " " << num_2 << endl;
   return 0;
}
```