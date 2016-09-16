#coding=utf-8

import node
import random

class Net:
	def __init__(self, input_node_cnt, hidden_node_cnt, output_node_cnt, active_func_type):
		self.input_node_cnt	 = input_node_cnt
		self.hidden_node_cnt = hidden_node_cnt
		self.output_node_cnt = output_node_cnt
		self.active_func_type = active_func_type

		self.input_layer  = []
		for i in xrange(self.input_node_cnt):
			self.input_layer.append(node.Node(0))
		self.hidden_layer = []
		for i in xrange(self.hidden_node_cnt):
			self.hidden_layer.append(node.Node(self.active_func_type))
		self.output_layer = []
		for i in xrange(self.output_node_cnt):
			self.output_layer.append(node.Node(self.active_func_type))

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
	
	def forward(self, data):
		for i in range(self.input_node_cnt):
			self.input_layer[i].forward(data[i])

		for j in range(self.hidden_node_cnt):
			sum_ = 0.0
			for i in range(self.input_node_cnt):
				sum_ += self.input_layer[i].forward_output * self.input_hidden_wgt[i][j]
			self.hidden_layer[j].forward(sum_) 

		for j in range(self.output_node_cnt):
			sum_ = 0.0
			for i in range(self.hidden_node_cnt):
				sum_ += self.hidden_layer[i].forward_output * self.hidden_output_wgt[i][j]
			self.output_layer[j].forward(sum_)

	def backward(self, label):
		for i in xrange(self.output_node_cnt):
			if self.active_func_type == 1:
				result = 0
			elif self.active_func_type == 2:
				result = -1

			if i == label:
				result = 1
			self.output_layer[i].backward(self.output_layer[i].forward_output - result)

		for i in xrange(self.hidden_node_cnt):
			sum_ = 0.0
			for j in xrange(self.output_node_cnt):
				sum_ += self.output_layer[j].backward_output * self.hidden_output_wgt[i][j]
			self.hidden_layer[i].backward(sum_)

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
	
	def log_weight(self):
		print >>sys.stderr, ",".join([str(e) for e in self.input_hidden_wgt[1]])
