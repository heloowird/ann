#coding=utf-8

import os
import numpy as np
import pandas as pd
import neural_net
import time

def load_pre_train_weight(filename, shape):
	if os.path.isfile(filename):
		weight = np.fromfile(filename, dtype=np.float)
		weight.shape = shape
		return weight
	else:
		return None

def load_one_hot_data(filename):
	raw_train_data = pd.read_csv(filename)
	train_data = raw_train_data.values[:]
	return train_data

def labels_to_one_hot(raw_labels):
	labels = np.zeros((raw_labels.shape[0], 10))
	labels[np.arange(raw_labels.shape[0]), raw_labels] = 1
	labels = labels.astype(np.float32)
	return labels

def gen_batch_data(train_data, batch_size):
	batch_index = np.random.randint(0, train_data.shape[0], batch_size)
	batch_train_data = train_data[batch_index]
	return (batch_train_data[:, 1:] / 255.0, labels_to_one_hot(batch_train_data[:, 0]))

def train_ann_classifier(step_num, eta):
	raw_train_data = load_one_hot_data("./data/train.csv")

	feature_num = raw_train_data.shape[1] - 1
	hidden_node_num = 13
	label_num = 10
	batch_size = 50

	print "feature_num: %d" % feature_num
	print "hidden_node_num: %d" % hidden_node_num
	print "label_num: %d" % label_num
	print "batch_size: %d" % batch_size

	# load pre-trained weight and bias
	input_hidden_w = load_pre_train_weight("./model/input_hidden_w.bin", (feature_num, hidden_node_num))
	input_hidden_b = load_pre_train_weight("./model/input_hidden_b.bin", (1, hidden_node_num))
	hidden_output_w = load_pre_train_weight("./model/hidden_output_w.bin", (hidden_node_num, label_num))
	hidden_output_b = load_pre_train_weight("./model/hidden_output_b.bin", (1, label_num))

	# initialize simple ann
	ann_net = neural_net.Net(batch_size, feature_num, hidden_node_num, label_num, input_hidden_w, input_hidden_b, hidden_output_w, hidden_output_b)

	# train
	np.random.shuffle(raw_train_data)
	train_len = int(raw_train_data.shape[0] * 0.8)
	train_data, valid_data = raw_train_data[:train_len], raw_train_data[train_len:]
	valid_features, valid_labels = valid_data[:, 1:] / 255.0, labels_to_one_hot(valid_data[:, 0])
	
	for i in xrange(step_num):
		features_data, labels_data = gen_batch_data(train_data, batch_size)
		#start_t = time.time()
		ann_net.train(features_data, labels_data, eta)
		#print "%d batch cost time:  %d ms" % (i, int((time.time() - start_t)*1000))

		# print accuracy of every one epoch
		if (i+1) == 1 or (i+1) % 840 == 0 or (i+1) == step_num:
			print "accuracy of %d batch (%d epoch): %f" % ((i+1), (i+1)/840, ann_net.get_accuracy(valid_features, valid_labels))

	ann_net.input_hidden_w.tofile("./model/input_hidden_w.bin")
	ann_net.input_hidden_b.tofile("./model/input_hidden_b.bin")
	ann_net.hidden_output_w.tofile("./model/hidden_output_w.bin")
	ann_net.hidden_output_b.tofile("./model/hidden_output_b.bin")

	# test
	test_data = load_one_hot_data("./data/test.csv")
	predict_label = ann_net.predict(test_data / 255.0)
	predict_label = [np.arange(1, 1+len(predict_label)), predict_label]
	predict_label = np.transpose(predict_label)
	np.savetxt('submission.csv', predict_label, fmt='%i,%i', header='ImageId,Label', comments='')

if __name__ == "__main__":
	train_ann_classifier(20*1000, 0.01)
