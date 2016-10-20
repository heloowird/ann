#coding=utf-8

import sys
from neural_input_layer import InputLayer
from neural_sigmoid_layer import SigmoidLayer
import numpy as np
import time

'''
The simple neural net only contains:
	one input layer
	one hidden layer
	one output layer
'''
class Net:
	# initalize neural net
	def __init__(self, batch_size, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		# initailize weights
		self.input_hidden_wgt = np.random.random((self.input_dim, self.hidden_dim)) * 0.01
		#print self.input_hidden_wgt[0]
		#print self.input_hidden_wgt[1]
		self.hidden_output_wgt = np.random.random((self.hidden_dim, self.output_dim)) * 0.01

		# initialize layers
		self.input_layer = InputLayer()
		self.hidden_layer = SigmoidLayer()
		self.output_layer = SigmoidLayer()

		# initialize data of layers
		self.reset(batch_size)
	
	def reset(self, batch_size):
		self.batch_size = batch_size
		#self.input_layer.set_forward_output(np.zeros((batch_size, self.input_dim)))
		#self.hidden_layer.set_forward_output(np.zeros((batch_size, self.hidden_dim)))
		#self.hidden_layer.set_backward_output(np.zeros((batch_size, self.hidden_dim)))
		#self.output_layer.set_forward_output(np.zeros((batch_size, self.output_dim)))
		#self.output_layer.set_backward_output(np.zeros((batch_size, self.output_dim)))

	# forward
	def forward(self, data):
		if data.shape[0] != self.batch_size:
			#print >>sys.stderr, "bad batch: input_num=%d, batch_size=%d" % (data.shape[0], self.batch_size)
			exit()
		self.input_layer.forward(data)
		#print "input_forward_zero_cnt: %d" % np.sum(self.input_layer.forward_output == 0.0)
		#print self.input_layer.forward_output[-1]
		#print "sum_input_before_activation:"
		#print np.dot(self.input_layer.forward_output, self.input_hidden_wgt)[-1]
		self.hidden_layer.forward(np.dot(self.input_layer.forward_output, self.input_hidden_wgt))
		#print "hidden_forward_zero_cnt: %d" % np.sum(self.hidden_layer.forward_output == 0.0)
		#print self.hidden_layer.forward_output[-1]
		self.output_layer.forward(np.dot(self.hidden_layer.forward_output, self.hidden_output_wgt))
		#print "output_forward_zero_cnt: %d" % np.sum(self.output_layer.forward_output == 0.0)
		#print self.output_layer.forward_output[-1]

	# backward
	def backward(self, label):
		if label.shape[0] != self.batch_size:
			#print >>sys.stderr, "bad batch: input_num=%d, batch_size=%d" % (label.shape[0], self.batch_size)
			exit()
		label.shape = (self.batch_size, 1)
		result = np.zeros((self.batch_size, self.output_dim))
		for i in range(self.batch_size):
			result[i][int(label[i][0])] = 1.0
		#print "reuslt_zero_cnt: %d" % np.sum(result == 0.0)
		#print self.output_layer.forward_output[-1]
		#print "erro_zero_cnt: %d" % np.sum((self.output_layer.forward_output - result) == 0.0)
		#print (self.output_layer.forward_output - result)[-1]
		self.output_layer.backward(self.output_layer.forward_output - result)
		#print "output_backward_zero_cnt: %d" % np.sum(self.output_layer.backward_output == 0.0)
		#print self.output_layer.backward_output[-1]
		self.hidden_layer.backward(np.dot(self.output_layer.backward_output, np.transpose(self.hidden_output_wgt)))
		#print "hidden_backward_zero_cnt: %d" % np.sum(self.hidden_layer.backward_output == 0.0)
		#print self.hidden_layer.backward_output[-1]

	# update
	def update(self, eta):
		sum_deta_input_hidden = np.zeros((self.input_dim, self.hidden_dim))
		for i in range(self.batch_size):
			sum_deta_input_hidden += np.dot(np.transpose(self.input_layer.forward_output[i:i+1, :]), self.hidden_layer.backward_output[i:i+1, :])
		#print "++++++++++ start ++++++++++"
		#print "before update:"
		#print  self.input_hidden_wgt[-1]
		#print "zero_cnt: %d" % np.sum(sum_deta_input_hidden == 0.0)
		self.input_hidden_wgt -= (eta / self.batch_size) * sum_deta_input_hidden
		#print "after update:"
		#print self.input_hidden_wgt[-1]
		#print "++++++++++ end ++++++++++++"

		sum_deta_hidden_output = np.zeros((self.hidden_dim, self.output_dim))
		for i in range(self.batch_size):
			sum_deta_hidden_output += np.dot(np.transpose(self.hidden_layer.forward_output[i:i+1, :]), self.output_layer.backward_output[i:i+1, :])
		self.hidden_output_wgt -= (eta / self.batch_size) * sum_deta_hidden_output

	def train(self, data, label, eta):
		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])
		#t1 = time.time()
		self.forward(data)
		#t2 = time.time()
		self.backward(label)
		#t3 = time.time()
		self.update(eta)
		#t4 = time.time()
		#print "forward cost time:  %d ms" % (int((t2-t1) * 1000))
		#print "backward cost time:  %d ms" % (int((t3-t2) * 1000))
		#print "update cost time:  %d ms" % (int((t4-t3) * 1000))
	
	def test(self, data, label):
		if data.shape[0] < 1:
			return 0.0

		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])

		self.forward(data)

		label.shape = (data.shape[0], 1) 
		predict_label = np.argmax(self.output_layer.forward_output, axis=1)
		predict_label.shape = (data.shape[0], 1)
		right_cnt = np.sum((predict_label - label - 0.0) == 0.0)
		return 1.0 * right_cnt / data.shape[0]

	def predict(self, data):
		if data.shape[0] < 1:
			return None

		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])

		self.forward(data)
		predict_label = np.argmax(self.output_layer.forward_output, axis=1)
		predict_label.shape = (data.shape[0], 1)
		predict_label = predict_label.astype(int)
		return predict_label
