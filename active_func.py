#coding=utf-8

import math

def forward(func_type, x):
	if func_type == 0:
		return x
	elif func_type == 1:
		return log_sigmoid(x)
	else:
		return tan_sigmoid(x)

def backward(func_type, y, x):
	if func_type == 0:
		return x
	elif func_type == 1:
		return dr_log_sigmoid(y, x)
	else:
		return dr_tan_sigmoid(y, x)

def log_sigmoid(x):
	return (1 / (1 + math.exp(-x)))

def dr_log_sigmoid(y, x):
	return (y * (1 - y) * x)

def tan_sigmoid(x):
	return ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))

def dr_tan_sigmoid(y, x):
	return ((1 - math.pow(y, 2)) * x)
