#coding=utf-8

import math

'''
activation function for neural node
'''

def forward(func_type, x):
	if func_type == 0:
		return x
	elif func_type == 1:
		return sigmoid(x)
	else:
		return tanh(x)

def backward(func_type, y, x):
	if func_type == 0:
		return x
	elif func_type == 1:
		return dr_sigmoid(y, x)
	else:
		return dr_tanh(y, x)

def sigmoid(x):
	return (1 / (1 + math.exp(-x)))

def dr_sigmoid(y, x):
	return (y * (1 - y) * x)

def tanh(x):
	return ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))

def dr_tanh(y, x):
	return ((1 - math.pow(y, 2)) * x)
