---
title: 104-Maximum Depth of Binary Tree
date: 2019-04-07 21:22:35
tags:
- Leetcode
categories:
- Leetcode
---

# 题目

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
return its depth = 3.
# C/C++解法
注意递归算法非常耗时，所以请不要直接用三元运算符进行下面类似操作:
```cpp
return (1+maxDepth(head->left))>(1+maxDepth(head->left))?1+maxDepth(head->left):1+maxDepth(head->right)
```
上面的语句会进行四次递归运算，会大大增加计算时间，采用下面的代码先将1+maxDepth(head->left)和1+maxDepth(head->right)的结果保存下来，再用三元运算符比较只需要进行两次递归运算。
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
   int maxDepth(TreeNode *head) {
      //还是用递归来找深度
      if (head == nullptr)
         return 0;
      else{
         int leftDepth=1+maxDepth(head->left);
         int rightDepth=1+maxDepth(head->right);
         return leftDepth>rightDepth?leftDepth:rightDepth;
      }

    }
};

// 层次遍历建立树
TreeNode *buildTrees_per_level(vector<int> a) {
   TreeNode *root = nullptr;
   queue<TreeNode *> s;
   if (a.empty()) {
      return root;
   }
   root = new TreeNode(a[0]);
   s.push(root);
   int i = 1;
   // 因为vector数组中用0来占位空节点了，所以左孩子编号%2==1,右孩子编号%2==0,编号0开始(根节点)
   while (!s.empty() && i < a.size()) {
      auto *temp = s.front();
      s.pop();
      //vector数组中为0的值表明对应下标位置是空节点
      if (a[i] != 0 && i % 2 == 1)
         temp->left = new TreeNode(a[i]);
      i++;
      //因为每个循环里i指针要移动两位,如果i移动一位就没有元素了，我们就跳出循环
      if (i >= a.size())
         break;
      if (a[i] != 0 && i % 2 == 0)
         temp->right = new TreeNode(a[i]);
      i++;
      // 因为是队列，所以左孩子先进队列，有孩子后进队列，出队列也是这个顺序
      if (temp->left != nullptr)
         s.push(temp->left);
      if (temp->right != nullptr)
         s.push(temp->right);
   }
   return root;
}

//层次遍历打印树中每个节点
void print_Trees_per_level(TreeNode *root) {
   if (root == nullptr) {
      cout << "二叉树为空树!" << endl;
      return;
   }
   queue<TreeNode *> s;
   s.push(root);
   // 完全二叉树左孩子编号%2==1,右孩子编号%2==0,编号0开始(根节点)
   while (!s.empty()) {
      auto *temp = s.front();
      s.pop();
      cout << temp->val << " ";
      // 因为是队列，所以左孩子先进队列，有孩子后进队列，出队列也是这个顺序
      if (temp->left != nullptr)
         s.push(temp->left);
      if (temp->right != nullptr)
         s.push(temp->right);
   }
   cout << endl;
}

int main() {
   // vector数组中按完全二叉树层次遍历的顺序来排列元素,中间没有元素的节点令其元素为0,建立树时这些节点不会建立
   vector<int> a = {3, 9, 20, 0, 0, 15, 7};
   TreeNode *root = buildTrees_per_level(a);
   print_Trees_per_level(root);
   Solution s;
   int depth = s.maxDepth(root);
   cout << depth << endl;
   return 0;
}
```