---
title: 226-Invert Binary Tree
date: 2019-04-08 18:23:01
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Invert a binary tree.

Example:

Input:
```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```
Output:
```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```
Trivia:

This problem was inspired by this original tweet by Max Howell:
```
Google: 90% of our engineers use the software you wrote (Homebrew), but you can’t invert a binary tree on a whiteboard so f*** off.
```
# C/C++解法
还是层次遍历，根节点先存入队列，然后出队，交换左右孩子指针，如果左右孩子不为空再入队，然后再出队进行上述循环直到队空。
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
   TreeNode *invertTree(TreeNode *root) {
      if (root == nullptr) {
         return root;
      }
      queue<TreeNode *> s;
      s.push(root);
      while (!s.empty()) {
         auto *temp = s.front();
         s.pop();
         auto *temp_1=temp->left;
         temp->left = temp->right;
         temp->right = temp_1;
         if(temp->left!= nullptr)
            s.push(temp->left);
         if(temp->right!= nullptr)
            s.push(temp->right);
      }
      return root;
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
   // 建立的数组必须是完全二叉树,且最后一个节点必须是父节点的右孩子
   vector<int> a = {4, 2, 7, 1, 3, 6, 9};
   TreeNode *head = buildTrees_per_level(a);
   print_Trees_per_level(head);
   Solution s;
   TreeNode *invertTree = s.invertTree(head);
   print_Trees_per_level(invertTree);
   return 0;
}
```