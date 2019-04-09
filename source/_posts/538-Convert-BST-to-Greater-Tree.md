---
title: 538-Convert BST to Greater Tree
date: 2019-04-09 12:35:00
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.


Example:
```
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13	
```
# C/C++解法
二叉搜索树的特性是左孩子值<根节点值<右孩子值。这里对值的操作是每个节点的值要加上所有大于该值的节点的值。那么我们就可以考虑用中序遍历将每个节点的指针存在一个vector数组中，中序遍历的顺序就是左根右，我们只要倒序遍历一次vector数组，从倒数第二个数开始，加上后面一个节点的值即可。
中序遍历有非递归和递归两种，非递归速度要快的多。
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
   //二叉搜索树的特性:左孩子<根<有孩子,因此我们可以用中序遍历(左根右)把树转成数组,就得到了一个从小到大排序的数组,然后从后往前累加
   //因为我们的vector数组存储的是指针,我们这里直接修改了指针指向的节点的值,而其left、right域值不变,所以直接返回root即可
   //这是非递归中序遍历,速度较快
   TreeNode *convertBST(TreeNode *root) {
      vector<TreeNode *> node_list;
      //先用中序遍历按左根右的顺序把各个节点指针存入vector;
      non_recursive_inorder_traversal(root, node_list);
      //按vector数组存储顺序修改各个指针指向的节点的值,指针值无需变化
      for (int i = node_list.size() - 2; i >= 0; i--)
         node_list[i]->val += node_list[i + 1]->val;
      return root;
   }

   void non_recursive_inorder_traversal(TreeNode *root, vector<TreeNode *> &a) {
      //中序遍历顺序是左根右
      stack<TreeNode *> s;
      while (root != nullptr || !s.empty()) {
         //先把所有左孩子都入栈,并且找到最左下角的左孩子
         while (root != nullptr) {
            s.push(root);
            root = root->left;
         }
         //这个时候入栈的都是左子树一系列左孩子,最后一个左孩子是最左下角的左孩子,出栈
         //先保存这个左孩子指针到a,然后其右孩子入栈,继续下一个循环,如果右孩子不存在,则从栈中再取上一个左孩子
         // 如果右孩子存在,找这个有孩子为根节点的子树中的最左下角孩子
         if (!s.empty()) {
            root = s.top();
            a.push_back(root);
            s.pop();
            root = root->right;
         }
      }
   }
};

//class Solution {
//public:
// //二叉搜索树的特性:左孩子<根<有孩子,因此我们可以用中序遍历(左根右)把树转成数组,就得到了一个值从小到大排序的数组,然后从后往前累加
//  //因为我们的vector数组存储的是指针,我们这里直接修改了指针指向的节点的值,而其left、right域值不变,所以直接返回root即可
// //这是递归中序遍历,速度较慢
// TreeNode *convertBST(TreeNode *root) {
//    vector<TreeNode *> node_list;
//    //先用中序遍历按左根右的顺序把各个节点指针存入vector;
//    InOrder(root, node_list);
//    //按vector数组存储顺序修改各个指针指向的节点的值,指针值无需变化
//    for (int i = node_list.size() - 2; i >= 0; i--)
//       node_list[i]->val += node_list[i + 1]->val;
//    return root;
// }
//
// void InOrder(TreeNode *root, vector<TreeNode *> &s) {
//    if (!root)
//       return;
//    else {
//       InOrder(root->left, s);
//       s.push_back(root);
//       InOrder(root->right, s);
//    }
// }
//};

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
   // 建立的数组必须是二叉搜索树,满足左孩子<根<右孩子
   vector<int> a = {9, 7, 11, 0, 8, 10, 12};
   TreeNode *root = buildTrees_per_level(a);
   print_Trees_per_level(root);
   Solution s;
   TreeNode *convert_root = s.convertBST(root);
   print_Trees_per_level(convert_root);
   return 0;
}
```