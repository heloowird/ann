#coding=utf-8

import neural_node
import random

'''
The simple neural net only contains:
	one input layer
	one hidden layer
	one output layer
'''
class Net:
	# initalize neural net
	def __init__(self, input_node_cnt, hidden_node_cnt, output_node_cnt, active_func_type):
		self.input_node_cnt = input_node_cnt
		self.hidden_node_cnt = hidden_node_cnt
		self.output_node_cnt = output_node_cnt
		self.active_func_type = active_func_type

		# initial all neural node that every layer contains
		# WARNING:
			# MUST NOT use [neural_node.Node(self.active_func_type)] * count
			# In this way, the nodes will be same, because deep copy
		self.input_layer = []
		for i in xrange(self.input_node_cnt):
			self.input_layer.append(neural_node.Node(0))
		self.hidden_layer = []
		for i in xrange(self.hidden_node_cnt):
			self.hidden_layer.append(neural_node.Node(self.active_func_type))
		self.output_layer = []
		for i in xrange(self.output_node_cnt):
			self.output_layer.append(neural_node.Node(self.active_func_type))

		# initail all weight, connecting layers
		self.input_hidden_wgt = []
		for i in xrange(self.input_node_cnt):
			w_j = []
			for j in xrange(self.hidden_node_cnt):
				w_j.append(random.random() * 0.1)
			self.input_hidden_wgt.append(w_j)
			
		self.hidden_output_wgt = []
		for i in xrange(self.hidden_node_cnt):
			w_j = []
			for j in xrange(self.output_node_cnt):
				w_j.append(random.random() * 0.1)
			self.hidden_output_wgt.append(w_j)
	
	# forward
	def forward(self, data):
		for i in xrange(self.input_node_cnt):
			self.input_layer[i].forward(data[i])

		# compute forward outputs of hidden layer
		for j in xrange(self.hidden_node_cnt):
			sum_ = 0.0
			for i in xrange(self.input_node_cnt):
				sum_ += self.input_layer[i].forward_output * self.input_hidden_wgt[i][j]
			self.hidden_layer[j].forward(sum_) 

		# compute forward outputs of output layer
		for j in xrange(self.output_node_cnt):
			sum_ = 0.0
			for i in xrange(self.hidden_node_cnt):
				sum_ += self.hidden_layer[i].forward_output * self.hidden_output_wgt[i][j]
			self.output_layer[j].forward(sum_)

	# backward
	def backward(self, label):
		# compute backward outputs of output layer
		for i in xrange(self.output_node_cnt):
			# WARNING: 
				# 1. different active function with different result
				# 2. if log sigmoid function with result -1, may be not converge
			if self.active_func_type == 1:
				result = 0
			elif self.active_func_type == 2:
				result = -1

			if i == label:
				result = 1
			self.output_layer[i].backward(self.output_layer[i].forward_output - result)

		# compute backward outputs of hidden layer
		for i in xrange(self.hidden_node_cnt):
			sum_ = 0.0
			for j in xrange(self.output_node_cnt):
				sum_ += self.output_layer[j].backward_output * self.hidden_output_wgt[i][j]
			self.hidden_layer[i].backward(sum_)

	# update all weights
	def update(self, eta):
		for i in xrange(self.input_node_cnt):
			for j in xrange(self.hidden_node_cnt):
				self.input_hidden_wgt[i][j] -= eta * self.input_layer[i].forward_output * self.hidden_layer[j].backward_output

		for i in xrange(self.hidden_node_cnt):
			for j in xrange(self.output_node_cnt):
				self.hidden_output_wgt[i][j] -= eta * self.hidden_layer[i].forward_output * self.output_layer[j].backward_output

	def train(self, data, label, eta):
		self.forward(data)
		self.backward(label)
		self.update(eta)
	
	def test(self, data, label):
		self.forward(data)
		max_index = -1
		max_value = 0.0
		for i in xrange(self.output_node_cnt):
			if self.output_layer[i].forward_output > max_value:
				max_index = i
				max_value = self.output_layer[i].forward_output
		return 1 if label == max_index else 0
