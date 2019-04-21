---
title: 437-Path Sum III
date: 2019-04-21 15:01:17
tags:
- Leetcode
categories:
- Leetcode
---


# 题目
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards
(traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:
```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```
# C/C++解法
三种做法:
* 深度递归遍历:先层层递归到叶节点，对叶节点左右孩子(空节点)进行深度遍历的结果为0，再对叶节点做深度遍历(没有孩子直接判断本节点值是否等于sum)，结果层层传递到上一层递归，直到到根节点为止，得到结果。
* 前序遍历:一层层向下递归到叶节点，对叶节点findpath，结果层层传递到上层递归，最后到根节点为止，得到结果。
* hashmap记录前缀路径和对应出现过的次数，m[curSum - sum]记录了从上往下遍历时curSum - sum差值的次数。res更新，要加上在root左右子树中findpath的结果，递归之下则一直递归到叶子节点。叶子节点返回的findPath(root->left, sum, curSum, m)和findPath(root->right, sum, curSum, m)均为0。res在左右子树找完curSum与sum差值的路径个人后，删掉这次curSum出现次数，避免在其他与本次寻找的子树不相关的子树中寻找时统计到了这次找到的curSum。
```cpp
# include <iostream>
# include <vector>
# include <queue>
# include <unordered_map>

# define null 0

using namespace std;


struct TreeNode {
   int val;
   TreeNode *left;
   TreeNode *right;

   TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

//若想要提交到leetcode，只需提交class Solution
class Solution {
public:
   //用哈希表来建立所有的前缀路径之和跟其出现次数之间的映射，然后看子路径之和有没有等于给定值的
   int pathSum(TreeNode *root, int sum) {
      unordered_map<int, int> m;
      //前缀路径之和为0的设为1
      m[0] = 1;
      int res = findPath(root, sum, 0, m);
      return res;
   }
   //以当前节点为终点的路径和满足target的路径有多少条
   //某节点到当前节点的路径和=根节点到当前节点的路径和-根节点到某节点的路径和
   int findPath(TreeNode *root, int sum, int curSum, unordered_map<int, int> &m) {
      if (!root)
         return 0;
      //curSum加上本节点之和
      curSum += root->val;
      //m[curSum - sum]即当前curSum-sum的差值(与sum还差多少的键值)出现过的次数赋给res
      //如果curSum - sum这个键之前没有则res为0
      int res = m[curSum - sum];
      //表明和curSum出现了一次,这里加1是为了在下面res+=递归中如果用到m[curSum]
      m[curSum]++;
      //res更新,要加上在root左右子树中findpath的结果,递归之下则一直递归到叶子节点
      //叶子节点返回的findPath(root->left, sum, curSum, m)和findPath(root->right, sum, curSum, m)均为0
      //结果返回上层,res加上左右子树的结果
      res += (findPath(root->left, sum, curSum, m) + findPath(root->right, sum, curSum, m));
      //res在左右子树找完curSum与sum差值的路径个人后,删掉这次curSum出现次数
      // 避免在其他与本次寻找的子树不相关的子树中寻找时统计到了这次找到的curSum
      m[curSum]--;
      return res;
   }
};


//若想要提交到leetcode，只需提交class Solution
//class Solution {
//public:
// //以每一个节点作为路径根节点进行前序遍历,查找每一条路径的权值和与sum是否相等
// int pathSum(TreeNode *root, int sum) {
//    if (!root)
//       return 0;
//    int res = findPath(root, 0, sum) + pathSum(root->left, sum) + pathSum(root->right, sum);
//    return res;
// }
//
// int findPath(TreeNode *root, int curSum, int sum) {
//    if (!root)
//       return 0;
//    curSum += root->val;
//    return (curSum == sum ? 1 : 0) + findPath(root->left, curSum, sum) + findPath(root->right, curSum, sum);
// }
//};


//若想要提交到leetcode，只需提交class Solution
//class Solution {
//public:
// //递归地使用深度优先遍历,树为空则结果是0,先一直向下递归,计算对叶子节点的pathSum(叶子节点地子树为空,结果为0)
// //然后一层一层向上返回结果,每一层节点的结果为本层dfs结果加左右子树的结果
// int pathSum(TreeNode *root, int sum) {
//    if (!root)
//       return 0;
//    int res = pathSum(root->left, sum) + pathSum(root->right, sum);
//    return DFS(root, sum) + res;
// }
//
// int DFS(TreeNode *root, int sum) {
//    if (!root)
//       return 0;
//    //dfs递归时,每次sum减去当前节点的值,剩余值
//    sum -= root->val;
//    //如果sum=0,说明刚好凑成一条路径,共有1+左右子树dfs的结果的路径个数
//    // 如果sum不等于,说明这个节点向下dfs没有路径,那么只将左右子树dfs的结果相加
//    return (sum == 0 ? 1 : 0) + DFS(root->left, sum) + DFS(root->right, sum);
// }
//};


TreeNode *build_tree_hierarchical_traversal(vector<int> &a) {
   if (a.empty())
      return nullptr;
   auto *root = new TreeNode(a[0]);
   queue<TreeNode *> s;
   s.push(root);
   int i = 1;
   while (!s.empty() && i < a.size()) {
      auto *temp = s.front();
      s.pop();
      if (a[i] != null) {
         temp->left = new TreeNode(a[i]);
         s.push(temp->left);
      }
      i++;
      if (a[i] != null && i < a.size()) {
         temp->right = new TreeNode(a[i]);
         s.push(temp->right);
      }
      i++;
   }
   return root;
}

void print_tree_hierarchical_traversal(TreeNode *root) {
   if (!root) {
      cout << "二叉树是空树" << endl;
      return;
   }
   queue<TreeNode *> s;
   s.push(root);
   while (!s.empty()) {
      auto *temp = s.front();
      s.pop();
      if (temp) {
         cout << temp->val << " ";
         if (temp->left || temp->right) {
            s.push(temp->left);
            s.push(temp->right);
         }
      } else {
         cout << "null" << " ";
      }
   }
   cout << endl;
}


int main() {
   vector<int> a = {10, 5, -3, 3, 2, null, 11, 3, -2, null, 1};
   //结果为3
   int sum = 8;
   TreeNode *root = build_tree_hierarchical_traversal(a);
   print_tree_hierarchical_traversal(root);
   Solution s;
   int count = s.pathSum(root, sum);
   cout << count;
   return 0;
}
```