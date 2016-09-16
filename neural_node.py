#coding=utf-8

import active_func as af

'''
neural node
'''
class Node:
	def __init__(self, active_func_type):
		self.active_func_type = active_func_type
		self.forward_input = 0.0
		self.forward_output = 0.0
		self.backward_input = 0.0
		self.backward_output = 0.0
	
	# forward
	def forward(self, x):
		self.forward_input = x
		self.forward_output = af.forward(self.active_func_type, self.forward_input) 
	
	# backward
	def backward(self, x):
		self.backward_input = x
		self.backward_output = af.backward(self.active_func_type, self.forward_output, self.backward_input)
