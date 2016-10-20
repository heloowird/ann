#coding=utf-8

import numpy as np

'''
neural input layer
'''
class InputLayer:
	def __init__(self):
		self.forword_output = None
		self.backward_output = None	
	
	def set_forward_output(self, f_output):
		self.forword_output = f_output

	def set_backward_output(self, b_output):
		self.backward_output = b_output

	# forward
	def forward(self, x):
		self.forward_output =  x
	
	# backward
	def backward(self, x):
		self.backward_output =  self.forward_output
