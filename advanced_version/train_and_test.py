#coding=utf-8

import numpy as np
import pandas as pd
import neural_net
import random
import time

def train_ann_classifier(step_num, eta):
	raw_train_data = pd.read_csv("./data/train.csv")

	feature_num = len(raw_train_data.columns) - 1
	hidden_node_num = int(feature_num * 1.5)
	label_num = 10
	batch_size = 100

	print "feature_num: %d" % feature_num
	print "hidden_node_num: %d" % hidden_node_num
	print "label_num: %d" % label_num
	print "batch_size: %d" % batch_size
	

	# initialize simple ann
	ann_net = neural_net.Net(batch_size, feature_num, hidden_node_num, label_num)

	# normalization
	raw_train_data = raw_train_data.values[:]
	train_data = (raw_train_data - np.min(raw_train_data)) / (np.max(raw_train_data) * 1.0 - np.min(raw_train_data))
	
	# train
	for i in xrange(step_num):
		np.random.shuffle(train_data)
		for j in xrange(train_data.shape[0]/batch_size):
			train_batch = train_data[j*batch_size : (j+1)*batch_size]
			start_t = time.time()
			ann_net.train(train_batch[:, 1:], train_batch[:, 0], eta)
			print "%d epoch and %d batch cost time:  %d ms" % (i, j, int((time.time() - start_t)*1000))

		# print accuracy of every 100 iterion
		#if i % 100 == 0:
		if True:
			start = random.randint(0, raw_train_data.shape[0] - batch_size)
			test_batch = raw_train_data[start : start+batch_size]
			print "accuracy of %d batch: %f" % (i, ann_net.test(test_batch[:, 1:], test_batch[:, 0]))

	ann_net.input_hidden_wgt.tofile("input_hidden_wgt.bin")
	ann_net.hidden_output_wgt.tofile("hidden_output_wgt.bin")

	# test
	test_data = pd.read_csv("./data/test.csv")
	predict_label = ann_net.predict(test_data[0:].values[:])
	predict_label.shape = (predict_label.shape[0], 1)
	np.savetxt("submission.csv", predict_label, fmt="%d")

if __name__ == "__main__":
	train_ann_classifier(100, 0.1)
