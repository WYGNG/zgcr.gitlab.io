---
title: 21-Merge Two Sorted Lists
date: 2019-04-09 20:12:12
tags:
- Leetcode
categories:
- Leetcode
---

# 题目
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:
```
Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```
# C/C++解法
当l1和l2均不为空时，逐一将值节点从小到大连接起来。注意对首个节点的单独特殊处理(leetcode输入的链表没有头节点，首个节点有值)。当只剩下l1残余或l2残余部分时直接连到前面的链表尾部。
```cpp
# include <iostream>
# include <vector>
# include <queue>
# include <stack>

using namespace std;


struct ListNode {
   int val;
   ListNode *next;

   ListNode(int x) : val(x), next(nullptr) {}
};


// 若要提交到leetcode只需提交class Solution
class Solution {
public:
   ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
      //如果l1为空或l2为空直接返回另一个链表
      if (!l1)
         return l2;
      if (!l2)
         return l1;
      //如果l1和l2都不为空，先单独对首个节点(有值)处理
      ListNode *l, *mergehead;
      if (l1->val <= l2->val) {
         l = l1;
         l1 = l1->next;
      } else {
         l = l2;
         l2 = l2->next;
      }
      //mergehead指向首个节点,作为最后返回的表头,l用来指向表的最后一个节点,这样可以不断在后面添加节点
      mergehead = l;
      //若l2和l2都不为空,循环为l后不断添加值更小的节点
      while (l1 && l2) {
         if (l1->val <= l2->val) {
            l->next = l1;
            l1 = l1->next;
         } else {
            l->next = l2;
            l2 = l2->next;
         }
         l = l->next;
      }
      //若l1或l2还未遍历到表尾,则直接把剩余部分连到l之后
      if (l1)
         l->next = l1;
      if (l2)
         l->next = l2;
      return mergehead;
   }
};

//新建单链表
ListNode *bulid_linklist(vector<int> &a) {
   ListNode *head;
   if (a.empty()) {
      head = nullptr;
      return head;
   }
   head = new ListNode(a[0]);
   auto *p = head;
   for (int i = 1; i < a.size(); i++) {
      auto *temp = new ListNode(a[i]);
      p->next = temp;
      p = p->next;
   }
   return head;
}

//打印单链表
void print_linklist(ListNode *head) {
   if (head == nullptr)
      return;
   while (head != nullptr) {
      cout << head->val << " ";
      head = head->next;
   }
   cout << endl;
}


int main() {
   vector<int> a = {5};
   vector<int> b = {1, 2, 4};
   ListNode *t1 = bulid_linklist(a);
   ListNode *t2 = bulid_linklist(b);
   print_linklist(t1);
   print_linklist(t2);
   Solution s;
   ListNode *mergehead = s.mergeTwoLists(t1, t2);
   print_linklist(mergehead);
   return 0;
}
```