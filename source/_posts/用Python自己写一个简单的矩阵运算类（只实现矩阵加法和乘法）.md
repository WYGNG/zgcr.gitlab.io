---
title: 用Python自己写一个简单的矩阵运算类（只实现矩阵加法和乘法）
date: 2019-07-22 19:02:18
tags:
- Python
categories:
- Python
---

# 介绍
这是一道面试时的编程题。我们要在Python中实现一个简单的二维矩阵运算类，该类可以将双层列表初始化为二维矩阵，并可以进行矩阵加法和矩阵乘法。
我们使用assert断言来判断初始化时输入的是否是列表，在进行矩阵加法和矩阵乘法时也使用断言来判断两个矩阵的维度是否满足要求。需要注意的是，为了使矩阵运算更具有通用性，矩阵加法和乘法的结果也应初始化为我们写的矩阵类，这样可以连续进行多次加法和乘法。
# Python代码实现
```python
class Matrix(object):
	def __init__(self, list_a):
		assert isinstance(list_a, list), "输入格式不是列表"

		self.matrix = list_a
		self.shape = (len(list_a), len(list_a[0]))
		self.row = self.shape[0]
		self.column = self.shape[1]

	def build_zero_value_matrix(self, shape):
		"""
		建立零值矩阵用来保存矩阵加法和乘法的结果
		:param shape: 
		:return: 
		"""
		assert isinstance(shape, tuple), "shape格式不是元组"

		zero_value_mat = []
		for i in range(shape[0]):
			zero_value_mat.append([])
			for j in range(shape[1]):
				zero_value_mat[i].append(0)

		zero_value_matrix = Matrix(zero_value_mat)

		return zero_value_matrix

	def matrix_addition(self, the_second_mat):
		"""
		矩阵加法
		:param the_second_mat: 
		:return: 
		"""
		assert isinstance(the_second_mat, Matrix), "输入的第二个矩阵不是矩阵类"
		assert the_second_mat.shape == self.shape, "两个矩阵维度不匹配,不能相加"

		result_mat = self.build_zero_value_matrix(self.shape)

		for i in range(self.row):
			for j in range(self.column):
				result_mat.matrix[i][j] = self.matrix[i][j] + the_second_mat.matrix[i][j]

		return result_mat

	def matrix_multiplication(self, the_second_mat):
		"""
		矩阵乘法
		:param the_second_mat: 
		:return: 
		"""
		assert isinstance(the_second_mat, Matrix), "输入的不是矩阵类"
		assert self.shape[1] == the_second_mat.shape[0], "第一个矩阵的列数与第二个矩阵的行数不匹配，不能相乘"

		shape = (self.shape[0], the_second_mat.shape[1])
		result_mat = self.build_zero_value_matrix(shape)

		for i in range(self.shape[0]):
			for j in range(the_second_mat.shape[1]):
				number = 0
				for k in range(self.shape[1]):
					number += self.matrix[i][k] * the_second_mat.matrix[k][j]
				result_mat.matrix[i][j] = number

		return result_mat


list_1 = [[1, 1], [2, 2]]
list_2 = [[2, 2], [3, 3]]
list_3 = [[1, 1, 1], [2, 2, 2]]

mat1 = Matrix(list_1)
mat2 = Matrix(list_2)
print(mat1.matrix, mat2.matrix)
mat4 = mat1.matrix_addition(mat2)
print(mat4.matrix)

mat3 = Matrix(list_3)
print(mat3.matrix)
mat5 = mat1.matrix_multiplication(mat3)
print(mat5.matrix)
```