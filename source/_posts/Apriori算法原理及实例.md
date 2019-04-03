---
title: Apriori算法原理及实例
date: 2019-04-03 23:36:29
tags:
- 机器学习原理推导
- 机器学习算法实践
categories:
- 机器学习原理推导
- 机器学习算法实践
mathjax: true
---


# 关联分析:频繁项集和关联规则
从大规模的数据中发现物品间隐含关系的方法被称为关联分析。关联分析是一种在大规模数据集中寻找有趣关系的任务。
**这些任务有两种形式:频繁项集和关联规则。**
频繁项集是经常出现在一块的物品的集合；
关联规则暗示的是两种物品之间可能存在很强的关系。

比如著名的“啤酒与尿布”的例子中，（啤酒，尿布）就是频繁项集中的一个例子。

**常用的频繁项集的评估标准有支持度、置信度、提升度:**
* 支持度就是几个关联的数据在数据集中出现的次数占总数据集的比重。一般来说，支持度高的数据不一定构成频繁项集，但是支持度太低的数据肯定不构成频繁项集。如果我们有两个想分析关联性的数据X和Y，则对应的支持度为:
$$
(X, Y)=P(X Y)=\frac{\text {number }(X Y)}{\text {num(AllSamples)}}
$$
* 置信度体现了一个数据出现后，另一个数据出现的概率，或者说数据的条件概率。如果我们有两个想分析关联性的数据X和Y，X对Y的置信度为:
$$
(X \Leftarrow Y)=P(X | Y)=\frac{P(X Y)}{P(Y)}
$$
* 提升度即在含有Y的条件下，同时含有X的概率，与X总体发生的概率之比。提升度>1且越高表明正相关性越高；提升度<1且越低表明负相关性越高；提升度=1表明没有相关性。
$$
Lift(X \Leftarrow Y)=\frac{P(X | Y)}{P(X)}=\frac{\text { Confidence }(X \Leftarrow Y)}{ P(X)}
$$

**举例:**
10000个超市订单（10000个事务），其中购买A牛奶（A事务）6000个，购买B牛奶（B事务）7500个，4000个同时包含两者。
那么A牛奶（A事务）和B牛奶（B事务）的支持度为:
P（A&B）=4000/10000=0.4
A牛奶（A事务）对B牛奶（B事务）的置信度为包含A的事务中同时包含B的占包含A的事务比例:
4000/6000=0.67
说明在购买A牛奶后，有0.67的用户去购买B牛奶。
伊利牛奶（B事务）B牛奶对A牛奶（A事务）的置信度为包含B的事务中同时包含A的占包含B的事务比例:
4000/7500=0.53
说明在购买A牛奶后，有0.53的用户去购买B牛奶。
这里有一点要注意，就是没有任何条件下时，B事务的出现的比例是0.75，而出现A事务，且同时出现B事务的比例是0.67，也就是说设置了A事务出现这个条件，B事务出现的比例反而降低了。这说明A事务和B事务是排斥的。
我们计算A事务对B事务的提升度:
P（B|A）/P（B）=0.67/0.75
显然提升度<1，说明A和B是负相关的。

商品列表中，可能存在单一商品组成的频繁项集，也可能存在两个以及两个以上的商品组成的频繁项集。在计算一个频繁项集的支持度时，通常需要遍历所有的商品列表求得，但当列表数目成千上万时，计算量过大，这种方法显然不适用。这时候我们就要用Apriori算法了。
# Apriori算法原理
Apriori算法的原理就是如果某个项集是频繁的，那么它的所有子集也是频繁的。同时它的逆否命题也成立:如果一个项集是非频繁集，那么它的所有超集也是非频繁的。
Apriori算法的两个输入参数分别是最小支持度和数据集，该算法首先会生成所有单个物品的项集列表，接着扫描交易记录来查看哪些项集满足最小支持度要求，那些不满足最小支持度的集合会被去掉。然后，对生下来的集合进行组合以生成包含两个元素的项集，接下来，再重新扫描交易记录，去掉不满足最小支持度的项集。该过程重复进行，直到所有项集都被去掉。
**举例:**
假设一个集合{A,B}是频繁项集，即A、B同时出现在一条记录的次数大于等于最小支持度min_support，则它的子集{A},{B}出现次数必定大于等于min_support，即它的子集都是频繁项集。
假设集合{A}不是频繁项集，即A出现的次数小于min_support，则它的任何超集如{A,B}出现的次数必定小于min_support，因此其超集必定也不是频繁项集。

**Apriori算法步骤:**
* 先计算1项集的支持度，筛选出频繁1项集；
* 然后排列组合出2项集，计算出2项集的支持度，筛选出频繁2项集；
* 然后通过连接和剪枝计算出3项集，计算出3项集的支持度，筛选出频繁3项集；
* 然后依次类推处理K项集，直到没有频繁集出现。 

**如何从K-1项集计算出K项集（K>=3，K=2时用组合公式C（2,n）即可）？**
连接:对K-1项集中的每个项集中的项排序，只有在前K-1项相同时才将这两项合并，形成候选K项集（因为必须形成K项集，所以只有在前K-1项相同，第K项不相同的情况下才合并。） 
剪枝:对于候选K项集，要验证所有项集的所有K-1子集是否频繁（是否在K-1项集中），去掉不满足的项集，就形成了K项集。
# Apriori算法Python实践
首先创建一个单项的候选项集合，然后通过支持度筛选得到单项的频繁项集合。while循环不断的用k-1项的频繁项集合使用apriori_generate（）函数生成k项的候选项集合，再使用scan_data_set（）函数生成k项的频繁项集合，加入l_all列表中，并把k项的候选项对应的支持度加入字典support_data。
```python
# 创建一个简单的数据集用来测试
def load_data_set():
   data_list = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
   return data_list


# 创建初始情况下只包含一个元素的候选项集集合
def create_one_candidate_set(data_set):
   one_candidate_set = []
   # 遍历数据集，并且遍历每一个集合中的每一项，创建只包含一个元素的候选项集集合
   for data_items in data_set:
      for item in data_items:
         if [item] not in one_candidate_set:
            one_candidate_set.append([item])
   # 对列表进行排序
   one_candidate_set.sort()
   # 固定列表one_candidate_set，使其不可变
   return list(map(frozenset, one_candidate_set))


# 找出满足支持度要求的候选项集合(即频繁项集)
def scan_data_set(data_set, ck, min_support):
   # 定义存储每个项集在消费记录中出现的次数的字典
   ss_count = {}
   # 遍历这个数据集，并且遍历候选项集集合，判断候选项是否是一条记录的子集，如果是则累加其出现的次数
   for data_items in data_set:
      # print(data_items)
      for scan in ck:
         # print(scan)
         # 判断scab是否是data_items的子集
         if scan.issubset(data_items):
            # 把scan存入ss_count计数
            if scan not in ss_count:
               ss_count[scan] = 1
            else:
               ss_count[scan] += 1
   # print(ss_count)
   data_item_num = float(len(data_set))
   # 定义满足最小支持度的候选项集列表
   suitable_list = []
   # 用于所有项集的支持度
   support_data = {}
   # 遍历整个字典
   for key in ss_count:
      # 计算当前项集的支持度
      support = ss_count[key] / data_item_num
      # 如果该项集支持度大于最小要求，则将其头插至L1列表中
      if support >= min_support:
         # 添加的是key，不是ss_count
         suitable_list.insert(0, key)
      # 记录每个项集的支持度
      support_data[key] = support

   return suitable_list, support_data


# Aprior算法,由k-1项的频繁项集生成k项的频繁项集和k项的候选集的支持度
def apriori_generate(ck, k):
   """
   由初始频繁项集的集合Lk生成新的生成更多项的频繁项集，k表示生成的新项集中所含有的元素个数
   :param ck: 初始的单项频繁项集的集合
   :param k: 生成的新项集中所含有的元素个数
   :return:
   """
   # 循环调用,一开始ck为初始单项的频繁项集,k=2
   # 存储新生成的k项的频繁项集
   suitable_list = []
   # 获取ck长度
   ck_len = len(ck)
   # 两两遍历候选项集中的集合
   for i in range(ck_len):
      for j in range(i + 1, ck_len):
         # 因为列表元素为集合，所以在比较前需要先将其转换为list,选择集合中前k-2个元素进行比较，如果相等，则对两个集合进行并操作
         # 这里可保证集合元素比合并前增加一个
         l1 = list(ck[i])[:k - 2]
         l2 = list(ck[j])[:k - 2]
         # 对转化后的列表进行排序，便于比较
         l1.sort()
         l2.sort()
         # 一开始k=2时l1和l2都取的是空集,这样两两单项频繁项集就组合成两项的候选项集
         # 后面由于frozenset会自动将小的数排在前面,我们每次组成k项候选集时会看前k-2项是否相等,相等则组成k项候选集
         if l1 == l2:
            # 对两个集合进行并操作
            suitable_list.append(frozenset(ck[i]) | frozenset(ck[j]))

   return suitable_list


# 生成所有的频繁项集
def apriori(data_set, min_support):
   # 创建初始情况下只包含一个元素的候选项集集合
   c1 = create_one_candidate_set(data_set)
   # 对数据集进行转换
   d = list(map(set, data_set))
   # 找出满足支持度要求的候选项集合(即单项的频繁项集集合),得到所有单项候选项集的支持度
   l1, support_data = scan_data_set(d, c1, min_support)
   # 定义存储所有频繁项集的列表
   l_all = [l1]
   k = 2
   # 生成所有满足条件的频繁项集(每次迭代项集元素个数加1)。迭代停止条件为当频繁项集中包含了所有单个项集元素后停止
   while len(l_all[k - 2]) > 0:
      # 每一轮循环都取了k-1项的频繁项集,然后生成了k项的频繁项集和k项的候选集支持度,分别存在l_all和support_data中
      ck = apriori_generate(l_all[k - 2], k)
      lk, support_data_k = scan_data_set(d, ck, min_support)
      # 将新产生的k项长的频繁项集和所有k项长的候选集支持度更新到l_all和support_data
      l_all.append(lk)
      support_data.update(support_data_k)
      k += 1

   for item in l_all:
      if len(item) == 0:
         l_all.remove(item)

   return l_all, support_data


if __name__ == '__main__':
   my_data_set = load_data_set()
   # # 构建第一个候选项集列表C1
   # C1 = create_one_candidate_set(my_data_set)
   # # 构建集合表示的数据集D
   # D = list(map(set, my_data_set))
   # # 选择支持度不小于0.5的项集作为频繁项集
   # L, suppData = scan_data_set(D, C1, 0.6)
   # print("单项的频繁项集L:", L)
   # print("所有单项的候选项集的支持度:", suppData)
   L_all, suppData_all = apriori(my_data_set, 0.5)
   print("所有项数的频繁项集L:", L_all)
   print("所有项数的候选项集的支持度:", suppData_all)
```
# 从频繁项集中找到满足最小置信度的事件对（又称为规则）
我们遍历所有频繁项集，只有一个元素的频繁项集直接跳过；恰好两个元素的频繁项直接计算其置信度，保留满足最小置信度的事件对；多于两个元素的频繁项使用rules_from_conseq（）函数，将每个频繁项先拆成单个元素，然后用apriori_generate（）函数重新生成两两事件对，这些事件对显然满足最小支持度，我们再计算其置信度，保留满足最小置信度的事件对。所有满足最小置信度的事件对我们称之为规则。
```python
# 创建一个简单的数据集用来测试
def load_data_set():
   data_list = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
   return data_list


# 创建初始情况下只包含一个元素的候选项集集合
def create_one_candidate_set(data_set):
   one_candidate_set = []
   # 遍历数据集,并且遍历每一个集合中的每一项,创建只包含一个元素的候选项集集合
   for data_items in data_set:
      for item in data_items:
         if [item] not in one_candidate_set:
            one_candidate_set.append([item])
   # 对列表进行排序
   one_candidate_set.sort()
   # 固定列表one_candidate_set,使其不可变
   # frozenset类型是指被"冰冻"的集合,就是说它们是不可变的,用户不能修改它们
   # 这里必须使用frozenset类型而不是set类型,因为之后必须要将这些集合作为字典键值使用,使用frozenset可以实现这一点,而set却做不到。
   return list(map(frozenset, one_candidate_set))


# 找出满足支持度要求的候选项集合(即频繁项集)
def scan_data_set(data_set, ck, min_support):
   # 定义存储每个项集在消费记录中出现的次数的字典
   ss_count = {}
   # 遍历这个数据集,并且遍历候选项集集合,判断候选项是否是一条记录的子集,如果是则累加其出现的次数
   for data_items in data_set:
      # print(data_items)
      for scan in ck:
         # print(scan)
         # 判断scab是否是data_items的子集
         if scan.issubset(data_items):
            # 把scan存入ss_count计数
            if scan not in ss_count:
               ss_count[scan] = 1
            else:
               ss_count[scan] += 1
   # print(ss_count)
   data_item_num = float(len(data_set))
   # 定义满足最小支持度的候选项集列表
   suitable_list = []
   # 用于所有项集的支持度
   support_data = {}
   # 遍历整个字典
   for key in ss_count:
      # 计算当前项集的支持度
      support = ss_count[key] / data_item_num
      # 如果该项集支持度大于最小要求,则将其头插至L1列表中
      if support >= min_support:
         # 添加的是key,不是ss_count
         suitable_list.insert(0, key)
      # 记录每个项集的支持度
      support_data[key] = support

   return suitable_list, support_data


# Aprior算法,由k-1项的频繁项集生成k项的频繁项集和k项的候选集的支持度
def apriori_generate(ck, k):
   """
   由初始频繁项集的集合Lk生成新的生成更多项的频繁项集,k表示生成的新项集中所含有的元素个数
   :param ck: 初始的单项频繁项集的集合
   :param k: 生成的新项集中所含有的元素个数
   :return:
   """
   # 循环调用,一开始ck为初始单项的频繁项集,k=2
   # 存储新生成的k项的频繁项集
   suitable_list = []
   # 获取ck长度
   ck_len = len(ck)
   # 两两遍历候选项集中的集合
   for i in range(ck_len):
      for j in range(i + 1, ck_len):
         # 因为列表元素为集合,所以在比较前需要先将其转换为list,选择集合中前k-2个元素进行比较,如果相等,则对两个集合进行并操作
         # 这里可保证集合元素比合并前增加一个
         l1 = list(ck[i])[:k - 2]
         l2 = list(ck[j])[:k - 2]
         # 对转化后的列表进行排序,便于比较
         l1.sort()
         l2.sort()
         # 一开始k=2时l1和l2都取的是空集,这样两两单项频繁项集就组合成两项的候选项集
         # 后面由于frozenset会自动将小的数排在前面,我们每次组成k项候选集时会看前k-2项是否相等,相等则组成k项候选集
         if l1 == l2:
            # 对两个集合进行并操作
            suitable_list.append(frozenset(ck[i]) | frozenset(ck[j]))

   return suitable_list


# 生成所有的频繁项集
def apriori(data_set, min_support):
   # 创建初始情况下只包含一个元素的候选项集集合
   c1 = create_one_candidate_set(data_set)
   # 对数据集进行转换
   d = list(map(set, data_set))
   # 找出满足支持度要求的候选项集合(即单项的频繁项集集合),得到所有单项候选项集的支持度
   l1, support_data = scan_data_set(d, c1, min_support)
   # 定义存储所有频繁项集的列表
   l_all = [l1]
   k = 2
   # 生成所有满足条件的频繁项集(每次迭代项集元素个数加1)。迭代停止条件为当频繁项集中包含了所有单个项集元素后停止
   while len(l_all[k - 2]) > 0:
      # 每一轮循环都取了k-1项的频繁项集,然后生成了k项的频繁项集和k项的候选集支持度,分别存在l_all和support_data中
      ck = apriori_generate(l_all[k - 2], k)
      lk, support_data_k = scan_data_set(d, ck, min_support)
      # 将新产生的k项长的频繁项集和所有k项长的候选集支持度更新到l_all和support_data
      l_all.append(lk)
      support_data.update(support_data_k)
      k += 1
   # 遍历删除频繁项集中的空列表
   for item in l_all:
      if len(item) == 0:
         l_all.remove(item)

   return l_all, support_data


# 计算每对事件的置信度,与最小置信度比较,返回满足最小置信度的后件
def calc_conf(freq_set, h, support_data, brl, min_conf):
   """
   计算规则的置信度,返回满足最小置信度的规则
   :param freq_set: 频繁项集
   :param h: 频繁项集中所有的单个元素
   :param support_data: 候选项集中所有元素的支持度
   :param brl: 满足置信度条件的关联规则
   :param min_conf: 最小置信度
   :return:
   """
   # 用于存储满足置信度要求的规则后件集合列表
   pruned_h = []
   # h表示规则后件的元素集合列表,遍历后件集合列表
   for con_seq in h:
      # 通过遍历每个频繁项元素都有机会作为后件
      # 计算置信度,对于每一个频繁集对应一组规则,对于每一个规则的每一个后件con_seq来说,其前件故为freqSet-con_seq
      # 所以此可能规则的置信度为以下计算公式
      # freq_set={2,3},con_seq={2},则freq_set - con_seq={3}
      conf = support_data[freq_set] / support_data[freq_set - con_seq]
      # 判断该规则置信度是否达到要求
      if conf >= min_conf:
         # 前件就是作为条件的事件,后件是条件前面的事件
         # print("前件", freq_set - con_seq, "-->", "后件", con_seq, "置信度:", conf)
         # 将满足置信度要求的规则（前件,后件,置信度）元组添加至规则列表中
         brl.append((freq_set - con_seq, con_seq, conf))
         # 存储满足置信度要求的后件
         pruned_h.append(con_seq)

   return pruned_h


# 对规则后件进行合并,以此生成后件有两元素的规则,有三元素的规则
def rules_from_conseq(freq_set, h, support_data, brl, min_conf):
   """
   对频繁项集中元素超过2的项集进行合并
   :param freq_set: 频繁项集
   :param h: 频繁项集中的所有元素,即可以出现在规则右部的元素
   :param support_data: 所有候选项集的支持度信息
   :param brl: 生成的规则
   :param min_conf: 最小置信度
   :return:
   """
   # 被调用时输入的是频繁项{2,3,5}
   # 获取后件元素的个数
   m = len(h[0])
   # 如果频繁集元素个数大于规则后件个数,也就是后件元素为1个,其他元素至少2个时
   if len(freq_set) > m + 1:
      # h是分解成单个项的频繁项集中的元素,通过apriori_generate函数变为两两的组合,这样就可以计算组合对的置信度了
      # 生成的组合对都满足最小支持度的限制
      hmp1 = apriori_generate(h, m + 1)
      # 通过calc_conf()获得满足置信度要求的后件元素列表
      hmp1 = calc_conf(freq_set, hmp1, support_data, brl, min_conf)
      # 判断是否还会有后件元素,如果有,则继续合并,并计算新生成规则
      if len(hmp1) > 1:
         rules_from_conseq(freq_set, hmp1, support_data, brl, min_conf)


# 这个函数调用rules_from_conseq()和calc_conf(),生成关联规则列表[(前件,后件,置信度)]
# 前件指条件概率中作为条件的事件,如P(A|B)中的B,后件指A
def generate_rules(l, support_data, min_conf):
   """
   根据频繁项集和最小置信度生成规则,生成的规则就是所有置信度大于最小置信度的事件对及它们的置信度
   :param l: 存储频繁项集,是一个双层列表,每个元素是有1、2、3...项元素的频繁项集
   :param support_data: 存储着所有候选项集的支持度
   :param min_conf: 最小置信度
   :return:
   """
   # 规则列表
   big_rule_list = []
   # 遍历整个频繁集列表
   for i in range(1, len(l)):
      # 遍历频繁集的每个元素,每个元素是有2、3...项元素的频繁项集(只有一个元素的频繁项集不遍历,因为不能计算置信度)
      for freq_set in l[i]:
         # 遍历每一种长度的频繁项集中的每一项
         h1 = [frozenset([item]) for item in freq_set]
         # 如果频繁项的元素个数大于2,则将h1合并,i>1即i从2开始,则频繁项至少有3个元素
         if i > 1:
            # 此函数对规则后件进行合并,创建出多条候选规则,在此函数中计算规则置信度,并将满足置信度的规则加入列表中
            # 递归创建可能出现的规则
            rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
         else:
            # 如果频繁集元素只有两个则直接计算置信度
            calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)

   return big_rule_list


if __name__ == "__main__":
   my_data_set = load_data_set()
   # # 构建第一个候选项集列表C1
   # C1 = create_one_candidate_set(my_data_set)
   # # 构建集合表示的数据集D
   # D = list(map(set, my_data_set))
   # # 选择支持度不小于0.5的项集作为频繁项集
   # L, suppData = scan_data_set(D, C1, 0.6)
   # print("单项的频繁项集L:", L)
   # print("所有单项的候选项集的支持度:", suppData)
   L_all, suppData_all = apriori(my_data_set, 0.5)
   # print("所有项数的频繁项集L:", L_all)
   # print("所有项数的候选项集的支持度:", suppData_all)
   # 置信度可调节
   rules_1 = generate_rules(L_all, suppData_all, 0.7)
   print("规则_1:", rules_1)
   # rules_2 = generate_rules(L_all, suppData_all, 0.5)
   # print("规则_2:", rules_2)
```

# Apriori算法进行关联分析案例:发现毒蘑菇的相似特征
数据集下载地址:https://pan.baidu.com/s/1Xt6egpTM-D4mIGW5Po6SlA  。
提取码：dwd2 
```python
import apriori as ap
mushDatSet = [line.split() for line in open("mushroom.dat").readlines()]
# print(mushDatSet[0])
# 第一个特征项表示有毒或者可食用,如果某样本有毒,则值为2。如果可食用,则值为1
# 计算0.3支持度下的频繁项集和候选项集以及对应的支持度
L, support_data = ap.apriori(mushDatSet, 0.3)
# print(L)
# 运行Apriori算法来寻找包含特征值为2的频繁项集,找到特征为毒蘑菇的最关联特征(1个)
for item in L[1]:
   # 在两项的频繁项中查找包含2的特征值
   if item.intersection("2"):
      print(item)
for item in L[3]:
   # 在四项的频繁项中查找包含2的特征值
   if item.intersection("2"):
      print(item)
```