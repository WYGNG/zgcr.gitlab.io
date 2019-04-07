---
title: 617. Merge Two Binary Trees
date: 2019-04-07 20:55:08
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:
```
Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
```
Note: The merging process must start from the root nodes of both trees.
# C/C++解法
用递归处理最方便。如果要处理的相同位置上的两个结点都不存在的话，直接返回；如果t1存在，t2不存在，那么我们就以t1的结点值建立一个新结点，然后分别对t1的左右子结点和空结点调用递归函数；如果t1不存在，t2存在，那么我们就以t2的结点值建立一个新结点，然后分别对t2的左右子结点和空结点调用递归函数；如果t1和t2都存在，那么我们就以t1和t2的结点值之和建立一个新结点，然后分别对t1的左右子结点和t2的左右子结点递归调用函数。
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
   // 这里用递归最方便
   // 如果要处理的相同位置上的两个结点都不存在的话，直接返回;
   // 如果t1存在，t2不存在，那么我们就以t1的结点值建立一个新结点，然后分别对t1的左右子结点和空结点调用递归函数
   // 如果t1不存在，t2存在，那么我们就以t2的结点值建立一个新结点，然后分别对t2的左右子结点和空结点调用递归函数。
   // 如果t1和t2都存在，那么我们就以t1和t2的结点值之和建立一个新结点，然后分别对t1的左右子结点和t2的左右子结点调用递归函数
   TreeNode *mergeTrees(TreeNode *t1, TreeNode *t2) {
      if (t1 == nullptr)
         return t2;
      if (t2 == nullptr)
         return t1;
      auto *newTree = new TreeNode(t1->val + t2->val);
      newTree->left = mergeTrees(t1->left, t2->left);
      newTree->right = mergeTrees(t1->right, t2->right);
      return newTree;
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
   vector<int> a = {1, 3, 2, 5};
   vector<int> b = {2, 1, 3, 0, 4, 0, 7};
   TreeNode *t1 = buildTrees_per_level(a);
   TreeNode *t2 = buildTrees_per_level(b);
   print_Trees_per_level(t1);
   print_Trees_per_level(t2);
   Solution s;
   TreeNode *mergeTree = s.mergeTrees(t1, t2);
   print_Trees_per_level(mergeTree);
   return 0;
}
```