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
	def __init__(self, batch_size, input_dim, hidden_dim, output_dim, input_hidden_w=None, input_hidden_b=None, hidden_output_w=None, hidden_output_b=None):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		# initailize weights
		self.init_weight(input_hidden_w, input_hidden_b, hidden_output_w, hidden_output_b)

		# initialize layers
		self.input_layer = InputLayer()
		self.hidden_layer = SigmoidLayer()
		self.output_layer = SigmoidLayer()

		# initialize data of layers
		self.reset(batch_size)
	
	def init_weight(self, input_hidden_w, input_hidden_b, hidden_output_w, hidden_output_b):
		# initailize weights
		if input_hidden_w is None:
			self.input_hidden_w = np.random.random((self.input_dim, self.hidden_dim)) * 0.01
		else:
			self.input_hidden_w = input_hidden_w

		if input_hidden_b is None:
			self.input_hidden_b = np.zeros((1, self.hidden_dim))
		else:
			self.input_hidden_b = input_hidden_b

		if hidden_output_w is None:
			self.hidden_output_w = np.random.random((self.hidden_dim, self.output_dim)) * 0.01
		else:
			self.hidden_output_w = hidden_output_w

		if hidden_output_b is None:
			self.hidden_output_b = np.zeros((1, self.output_dim))
		else:
			self.hidden_output_b = hidden_output_b

	def reset(self, batch_size):
		self.batch_size = batch_size

	# forward
	def forward(self, data):
		if data.shape[0] != self.batch_size:
			print >>sys.stderr, "bad batch: input_num=%d, batch_size=%d" % (data.shape[0], self.batch_size)
			exit()
		self.input_layer.forward(data)
		self.hidden_layer.forward(np.dot(self.input_layer.forward_output, self.input_hidden_w) + self.input_hidden_b)
		self.output_layer.forward(np.dot(self.hidden_layer.forward_output, self.hidden_output_w) + self.hidden_output_b)

	# backward
	def backward(self, label):
		if label.shape[0] != self.batch_size:
			print >>sys.stderr, "bad batch: input_num=%d, batch_size=%d" % (label.shape[0], self.batch_size)
			exit()
		self.output_layer.backward(self.output_layer.forward_output - label)
		self.hidden_layer.backward(np.dot(self.output_layer.backward_output, np.transpose(self.hidden_output_w)))

	# update
	def update(self, eta):
		sum_deta_input_hidden = np.zeros((self.input_dim, self.hidden_dim))
		for i in range(self.batch_size):
			sum_deta_input_hidden += np.dot(np.transpose(self.input_layer.forward_output[i:i+1, :]), self.hidden_layer.backward_output[i:i+1, :])
		self.input_hidden_w -= (eta / self.batch_size) * sum_deta_input_hidden
		self.input_hidden_b -= (eta / self.batch_size) * (np.cumsum(self.hidden_layer.backward_output)[-1])

		sum_deta_hidden_output = np.zeros((self.hidden_dim, self.output_dim))
		for i in range(self.batch_size):
			sum_deta_hidden_output += np.dot(np.transpose(self.hidden_layer.forward_output[i:i+1, :]), self.output_layer.backward_output[i:i+1, :])
		self.hidden_output_w -= (eta / self.batch_size) * sum_deta_hidden_output
		self.hidden_output_b -= (eta / self.batch_size) * (np.cumsum(self.output_layer.backward_output)[-1])

	def train(self, data, label, eta):
		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])
		self.forward(data)
		self.backward(label)
		self.update(eta)
	
	def get_accuracy(self, data, label):
		if data.shape[0] < 1:
			return 0.0

		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])

		self.forward(data)
		predict_label = np.argmax(self.output_layer.forward_output, axis=1)
		true_label = np.argmax(label, axis=1)
		right_cnt = np.sum((predict_label - true_label) == 0)
		return 1.0 * right_cnt / data.shape[0]

	def predict(self, data):
		if data.shape[0] < 1:
			return None

		if data.shape[0] != self.batch_size:
			self.reset(data.shape[0])

		self.forward(data)
		predict_label = np.argmax(self.output_layer.forward_output, axis=1)
		predict_label = predict_label.astype(int)
		return predict_label
