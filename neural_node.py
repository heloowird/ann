#coding=utf-8

import activation_func as af

'''
neural node
'''
class Node:
	def __init__(self, activation_func_type):
		self.af_type = activation_func_type
		self.forward_input = 0.0
		self.forward_output = 0.0
		self.backward_input = 0.0
		self.backward_output = 0.0
	
	# forward
	def forward(self, x):
		self.forward_input = x
		self.forward_output = af.forward(self.af_type, self.forward_input) 
	
	# backward
	def backward(self, x):
		self.backward_input = x
		self.backward_output = af.backward(self.af_type, self.forward_output, self.backward_input)
