#coding=utf-8

import sys
import random
import neural_net

# load all labels from local file
def load_label(filename):
	data_lst = []
	with open(filename) as f:
		for line in f:
			line = line.strip("\r\n")
			data_lst.append(line)
	return data_lst

# load train or test data from local file
def load_data(filename):
	data_lst = []
	with open(filename) as f:
		for line in f:
			line = line.strip("\r\n")
			fs = line.split(",")
			if len(fs) != 5:
				print >>sys.stderr, "bad format: %s" % line
				continue
			feature_lst = [float(ele) for ele in fs[:-1]]
			feature_lst.append(fs[-1])
			data_lst.append(feature_lst)
	return data_lst

# split all data into two parts: one for training, another for testing
def split_train_and_test(data_lst, train_lst, test_lst):
	for ele in data_lst:
		if random.random() < .15:
			test_lst.append(ele)
		else:
			train_lst.append(ele)
	
# save train and test data separately
def save(data_lst, filename):
	saved_file = open(filename, "w")
	for ele in data_lst:
		saved_file.write("%s\n" % ",".join([str(e) for e in ele]))
	saved_file.close()

# the function should run at the first time
#	1.	load all data for splitting
#	2.	split data into two parts
#	3.	save two parts respectly
def preprocess():
	train_data = []
	test_data = []

	total_data = load_data("./data/iris.data")
	split_train_and_test(total_data, train_data, test_data)
	save(train_data, "./data/iris_train.data")
	save(test_data, "./data/iris_test.data")

# train ann classifer and test
def train_ann_classifier(step_num, eta, active_func_type):
	#preprocess() # run at the first time to split the data

	train_data = load_data("./data/iris_train.data")
	labels = load_label("./data/iris_label.data")

	feature_num = len(train_data[0]) - 1
	hidden_node_num = feature_num + 8
	label_num = len(labels)

	'''
	print "feature_num: %d" % feature_num
	print "hidden_node_num: %d" % hidden_node_num
	print "label_num: %d" % label_num
	'''

	# initialize simple ann
	ann_net = neural_net.Net(feature_num, hidden_node_num, label_num, active_func_type)

	# train
	for i in xrange(step_num):
		for ele in train_data:
			ann_net.train(ele[:-1], labels.index(ele[-1]), eta)

		# print accuracy of every 100 iterion
		if i % 100 == 0:
			m_count = 0
			for ele in train_data:
				m_count += ann_net.test(ele[:-1], labels.index(ele[-1]))
			print "accuracy of %d steps: %f" % (i, 1.0 * m_count / len(train_data))

	# test
	test_data = load_data("./data/iris_train.data")
	m_count = 0
	for ele in test_data:
		m_count += ann_net.test(ele[:-1], labels.index(ele[-1]))
	print "final accuracy: %f" % (1.0 * m_count / len(test_data))

if __name__ == "__main__":
	train_ann_classifier(1000, 0.01, 2)

