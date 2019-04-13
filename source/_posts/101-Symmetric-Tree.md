---
title: 101-Symmetric Tree
date: 2019-04-13 13:48:36
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).


For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```


But the following [1,2,2,null,3,null,3]  is not:
```
    1
   / \
  2   2
   \   \
   3    3
```



Note:

Bonus points if you could solve it both recursively and iteratively.
# C/C++解法
本题是要判断一个树是否是对称的。有迭代解法和递归解法，迭代解法较快。两种解法的思路基本相同，先判断根节点的左孩子和右孩子，如果只有根节点，那么也是对称的；如果只有左孩子或右孩子，立刻可以判断不对成；如果左孩子和右孩子值相同，我们就要继续向下一层判断左孩子的左孩子与右孩子的右孩子值是否相同，且左孩子的右孩子与右孩子的左孩子是否相同。迭代和递归的区别就是迭代是从根节点还是向下一层一层判断，而递归是先递归到最底层判断，然后逐层向上一层一层判断。
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
   //迭代方法,中序遍历
   bool isSymmetric(TreeNode *root) {
      if (!root)
         return true;
      //顺着左右子树分别向下找最左下的左孩子和最右下的右孩子
      stack<TreeNode *> left, right;
      left.push(root);
      right.push(root);
      TreeNode *lcur = root, *rcur = root;
      while (true) {
         //每轮循环只有左下的左孩子和右下的右孩子都存在时才各自入栈
         if (lcur && rcur) {
            left.push(lcur);
            lcur = lcur->left;
            right.push(rcur);
            rcur = rcur->right;
            //如果已经找到了最左下的左孩子和最右下的右孩子且都已经入栈时
         } else if (!lcur && !rcur) {
            //若两栈为空,说明一路出栈比较的结果都是正确的,终止循环
            if (left.empty() && right.empty())
               break;
               //如果两栈都不为空,分别取出一个左下左孩子和右下右孩子比较
               //对于只有根节点的树就是根节点和根节点比较
            else if (!left.empty() && !right.empty()) {
               TreeNode *temp1 = left.top(), *temp2 = right.top();
               left.pop();
               right.pop();
               if (temp1->val != temp2->val)
                  return false;
               //左下左孩子和右下右孩子正确时,再比较左下左孩子的右兄弟与右下右孩子的左兄弟
               lcur = temp1->right;
               rcur = temp2->left;
               //若两栈一个空一个不为空,肯定返回false
            } else
               return false;
            //如果一开始找最左下左孩子和最右下右孩子时就发现其深度不一样
         } else
            return false;
      }
      return true;
   }
};

//class Solution {
//public:
// bool isSymmetric(TreeNode *root) {
//    if (!root)
//       return true;
//    return check(root->left, root->right);
// }
//
// bool check(TreeNode *left, TreeNode *right) {
//    if (!left&& !right)
//       return true;
//    else if (left&& right) {
//       //递归时检查左右孩子值是否相等,还要检查左孩子的左孩子与右孩子的右孩子是否相等
//       //还要检查左孩子的右孩子与右孩子的左孩子是否相等
//       return left->val == right->val &&check(left->left, right->right) &&check(left->right, right->left);
//    }
//    return false;
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
   vector<int> a = {1, 2, 2, 3, 4, 4, 3};
   vector<int> b = {1, 2, 2, 0, 3, 0, 3};
   TreeNode *root1 = buildTrees_per_level(a);
   TreeNode *root2 = buildTrees_per_level(b);
   print_Trees_per_level(root1);
   print_Trees_per_level(root2);
   Solution s;
   bool flag1 = s.isSymmetric(root1);
   bool flag2 = s.isSymmetric(root2);
   cout << boolalpha << flag1 << flag2 << endl;
   return 0;
}
```