---
title: 572-Subtree of Another Tree
date: 2019-04-22 12:18:06
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.


Example 1:


Given tree s:
```
     3
    / \
   4   5
  / \
 1   2
```
Given tree t:
```
   4 
  / \
 1   2
```
Return true, because t has the same structure and node values with a subtree of s.


Example 2:


Given tree s:
```
     3
    / \
   4   5
  / \
 1   2
    /
   0
```
Given tree t:
```
   4
  / \
 1   2
```
Return false.
# C/C++解法
前序遍历思想，先判断s从根节点开始的树是否包含树t，如果不是，使用前序遍历思想，再判断s的左子树和右子树中是否包含树t。is_same_tree函数中也是一个前序遍历，即分别判断s和t中根、左孩子、右孩子位置上的节点值是否相同，只要有一个节点不相同即返回false。
```cpp
# include <iostream>
# include <vector>
# include <queue>

# define null -1

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
   bool isSubtree(TreeNode *s, TreeNode *t) {
      if (!s)
         return false;
      if (is_same_tree(s, t))
         return true;
      return isSubtree(s->left, t) || isSubtree(s->right, t);
   }

   bool is_same_tree(TreeNode *s, TreeNode *t) {
      if (!s && !t)
         return true;
      if (!s || !t)
         return false;
      if (s->val != t->val)
         return false;
      return is_same_tree(s->left, t->left) && is_same_tree(s->right, t->right);
   }
};


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
         s.push(temp->left);
         s.push(temp->right);
      } else {
         cout << "null" << " ";
      }
   }
   cout << endl;
}


int main() {
   vector<int> a1 = {3, 4, 5, 1, 2};
   vector<int> sub = {4, 1, 2};
   vector<int> b1 = {3, 4, 5, 1, 2, null, null, null, null, 0, null};
   TreeNode *root_a1 = build_tree_hierarchical_traversal(a1);
   TreeNode *root_a2 = build_tree_hierarchical_traversal(b1);
   TreeNode *root_sub = build_tree_hierarchical_traversal(sub);
   print_tree_hierarchical_traversal(root_a1);
   print_tree_hierarchical_traversal(root_a2);
   print_tree_hierarchical_traversal(root_sub);
   Solution s;
   bool flag1 = s.isSubtree(root_a1, root_sub);
   bool flag2 = s.isSubtree(root_a2, root_sub);
   cout << boolalpha << flag1 << " " << flag2;
   return 0;
}
```