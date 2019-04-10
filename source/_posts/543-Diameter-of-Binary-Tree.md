---
title: 543-Diameter of Binary Tree
date: 2019-04-10 13:57:36
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.



Example:

Given a binary tree 
```
          1
         / \
        2   3
       / \     
      4   5    
```


Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].


Note:
The length of path between two nodes is represented by the number of edges between them.
# C/C++解法
注释中的四句cout输出是为了更直观的看到递归过程。这里要计算最长路径，最长路径分为两种情况：
* 最长路径经过根节点(根节点两个子树都有)，那么根节点的左子树的深度和右子树的深度就是我们的结果;
* 最长路径没有经过根节点(即根节点只有一个子树)，这个问题就分为两个子问题，分别设置新的根节点为其左子节点和右子节点。
   使用递归函数计算树最大深度，底层叶节点左右子树深度为0，maxlength总是与左右子树深度之和比较取大值，返回时返回左右子树深度中大的值加1，就是本节点的深度。
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
   //两种情况:
   //最长路径经过根节点,那么根节点的左子树的深度和右子树的深度就是我们的结果
   //最长路径没有经过根节点，这个问题就分为两个子问题，分别设置新的根节点为其左子节点和右子节点
   int diameterOfBinaryTree(TreeNode *root) {
      if (!root)
         return 0;
      int maxLength = 0;
      findmaxlength(root, maxLength);
      return maxLength;
   }

   int findmaxlength(TreeNode *root, int &maxLength) {
      if (!root)
         return 0;
      int left = findmaxlength(root->left, maxLength);
      int right = findmaxlength(root->right, maxLength);
      //递归时先不断调用递归函数将参数入栈,然后出栈(先计算最底层的递归函数)
      //cout << "本节点"<<root->val << " " <<"初始最大路径长度"<<maxLength << endl;
      //cout << "左子树深度"<<left << " " <<"右子树深度"<<right << endl;
      //分别求左右子树的maxlength
      //比较现在的maxlength和我们现在递归的某个子树的左子子树深度+右子子树深度和哪个大
      maxLength = (left + right > maxLength) ? left + right : maxLength;
      //cout << "本轮结束最大路径长度"<<maxLength << " " <<"本节点树深度"<<((left > right) ? (left + 1) : (right + 1)) << endl;
      //cout<<endl;
      //返回值是子树最大深度+1,即上一级子树的最大深度
      return (left > right) ? left + 1 : right + 1;
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
   // 数组可取1, 2, 3, 4, 5, 0, 0, 8, 9和1, 2, 0, 4, 5, 0, 0, 8, 9来对应上面的两种情况
   vector<int> a = {1, 2, 0, 4, 5, 0, 0, 8, 9};
   TreeNode *root = buildTrees_per_level(a);
   print_Trees_per_level(root);
   Solution s;
   int path_length = s.diameterOfBinaryTree(root);
   cout << path_length << endl;
   return 0;
}
```