---
title: 206-Reverse Linked List
date: 2019-04-08 21:46:37
tags:
- Leetcode
categories:
- Leetcode
---

# 题目

Reverse a singly linked list.

Example:
```
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

Follow up:

A linked list can be reversed either iteratively or recursively. Could you implement both?
# C/C++解法
分为迭代解法和递归解法。迭代解法即另创建一个链表头节点，然后用头插法将head链表首个节点不断插入到新创建的链表头部；递归法思路有点复杂，请仔细看下面代码注释中的详细说明。
```cpp
# include <string>
# include <iostream>
# include <vector>
# include <queue>

using namespace std;


struct ListNode {
   int val;
   ListNode *next;

   ListNode(int x) : val(x), next(NULL) {}
};


// 若要提交到leetcode只需提交class Solution
// 迭代做法
class Solution {
public:
   //迭代方法就是新创建一个指向另一个空链表头的指针,使用头插法将head中的首个元素不断地插入到另一个空链表中
   ListNode* reverseList(ListNode* head) {
      if(head== nullptr)
         return head;
      ListNode *p=head;
      head=head->next;
      p->next= nullptr;
      while(head!= nullptr){
         auto *temp=head;
         head=head->next;
         temp->next=p;
         p=temp;
      }
      return p;
   }
};

// 若要提交到leetcode只需提交class Solution
// 递归做法
//class Solution {
//public:
// ListNode *reverseList(ListNode *head) {
//    //这里用值代表位置
//    //不断进入递归函数，直到head指向倒数第一个节点即reverseList(5)，此时head->next为空,reverseList(5)返回head,即newHead指向的位置为5
//    //对reverseList(4),newhead指向5,5->next指向4,4->next指向nullptr,同时1->2->3->4,注意这里指向的4和前面5指向的4是同一个节点,返回newhead指向5
//    //即递归形式如下:
//    //r(1)
//    //  newhead=r(2)
//    //  r(2)
//    //      newhead=r(3) 上面各级操作与类似r(3)中操作
//    //      r(3)
//    //         newhead=r(4),head指向3,3->next->next原来是nullptr,现在把3节点移过去,然后3->next令为nullptr
//    //          r(4)
//    //              newhead=r(5),直接得到newhead指向5节点,然后r(4)中head指向4,把5->next指向4,4->next指向nullptr,同时1->2->3->4
//    //          r(4)中5->4->nullptr与1->2->3->4中的4是同一个节点。
//    // 最终返回的newHead指向5节点
//    if (head == nullptr || head->next == nullptr)
//       return head;
//    //每次调用输入的指针变量都是复制值进去,对于调用的主函数,head指针没有改变
//    ListNode *newHead = reverseList(head->next);
//    head->next->next = head;
//    head->next = nullptr;
//    return newHead;
// }
//};

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
   vector<int> a = {1, 2, 3, 4, 5};
   ListNode *head = bulid_linklist(a);
   print_linklist(head);
   Solution s;
   ListNode *newHead = s.reverseList(head);
   print_linklist(newHead);
   return 0;
}
```