# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-04-01 13:41:25
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-04-30 12:11:36

import numpy as np
import json
import os
import tensorflow as tf
import csv
from sklearn.utils import shuffle



# id ~ '0' - '4084'
# however, this feature is not neccassary to be encoded (can save 24 digits)
def mix_nodes_id(id1, id2):

	# 1> one-hot encoding -- dimension is too high
	'''
	array = np.zeros(4085)
	array[int(id1)] = 1
	array[int(id2)] = 1
	return array
	'''

	# 2> invert the id into binary code
	# max id is 4084 -> 12 digits bin
	bin_id1, bin_id2 = bin(int(id1))[2:], bin(int(id2))[2:]
	bin_id1 = '0' * (12 - len(bin_id1)) + bin_id1
	bin_id2 = '0' * (12 - len(bin_id2)) + bin_id2

	# shape (24,)
	return np.array(list(bin_id1+bin_id2)).astype(np.int32)


# encode each node in nodes.json
def encode_node(node):
	# "keyword_x" : x -> 0 ~ 52
	# "venue_x" : x -> 0 ~ 347
	keyword = np.zeros(53)
	venue = np.zeros(348)
	for key,val in node.items():
		if key == "id":
			nid = str(val)
		if key == "first":
			first = val
		if key == "last":
			last = val
		if key == "num_papers":
			num_paper = val
		if key.startswith("keyword"):
			keyword[int(key.split("_")[-1])] = 1
		if key.startswith("venue"):
			venue[int(key.split("_")[-1])] = 1

	info1 = np.array([first, last, num_paper])
	info2 = np.concatenate((keyword, venue), axis=0)
	info = np.concatenate((info1, info2), axis=0).astype(np.int32)

	return nid, info


def extract_node_info():
	# {'id':[first,last,keyword,venue], ...}
	nodes_vec = {}
	with open("nodes.json", "r") as f:
		nodes = json.load(f)
		for node in nodes:
			nid, info = encode_node(node)
			nodes_vec[nid] = info

	return nodes_vec


def fetch_train_data():
	# total 53872 edges in train.txt
	# but 26937 unique edges
	if os.path.exists("method2_train_data.npy"):
		# shape (26937, 832)
		return np.load("method2_train_data.npy")

	final_data = []
	edge_set = {}
	node_info = extract_node_info()
	with open("train.txt", "r") as f:
		for i in f.readlines():
			array = i.split()
			for ele in array[1:]:
				small, great = (array[0], ele) if (int(array[0]) < int(ele)) else (ele, array[0])
				tmp = edge_set.get(small, [])
				# duplicate edge
				if great in tmp: continue
				else:
					edge_set[small] = tmp + [great]
					print(small)
					# shape (24,)
					id_vec = mix_nodes_id(small, great)
					# shape (404,)
					info1 = node_info[small] if small in node_info else np.zeros(404)
					# shape (404,)
					info2 = node_info[great] if great in node_info else np.zeros(404)
					# shape (832,)
					complete_info = np.concatenate((id_vec, info1, info2), axis=0)
					if len(final_data) != 0:
						final_data = np.vstack((final_data, complete_info))
					else:
						final_data = complete_info
					
	np.save("method2_train_data.npy", final_data)

	# shape (26937, 832)
	return final_data


def fetch_test_data():

	if os.path.exists("method2_test_data.npy"):
		# shape (2000, 832)
		return np.load("method2_test_data.npy")

	node_info = extract_node_info()
	final_data = []
	with open("test-public.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			small, great = (line[0], line[1]) if (int(line[0]) < int(line[1])) else (line[1], line[0])
			# shape (24,)
			id_vec = mix_nodes_id(small, great)
			# shape (404,)
			info1 = node_info[small] if small in node_info else np.zeros(404)
			# shape (404,)
			info2 = node_info[great] if great in node_info else np.zeros(404)
			# shape (832,)
			complete_info = np.concatenate((id_vec, info1, info2), axis=0)
			if len(final_data) != 0:
				final_data = np.vstack((final_data, complete_info))
			else:
				final_data = complete_info

	np.save("method2_test_data.npy", final_data)
	print(final_data.shape)
	# shape (2000, 832)
	return final_data


def train():

	# model construction
	x = tf.placeholder(tf.float32, [None, 832])
	W = tf.Variable(tf.zeros([832, 2]))
	b = tf.Variable(tf.zeros([2]))

	# softmax layer
	y_pred = tf.nn.softmax(tf.matmul(x, W) + b)

	# real label
	y_real = tf.placeholder("float", [None, 2])

	# cross entropy loss
	cross_entropy = -tf.reduce_sum(y_real * tf.log(y_pred))
	# gradient descent algorithm (with lr 0.01)
	train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

	# variables intialization
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	# starts model
	with tf.Session() as sess:

		sess.run(init)
		train_data = fetch_train_data()

		# model training
		batch_size = 200
		for _ in range(train_data.shape[0] // batch_size):
			order = shuffle(np.arange(train_data.shape[0]))[:batch_size]
			batch_x = train_data[order]
			# all label is 'connected' - [1 0]   ps. 'not connected' - [0 1]
			batch_y = np.tile(np.array([1,0]), (batch_size, 1))

			sess.run(train_op, feed_dict={x: batch_x, y_real: batch_y})

		'''
		# 2000 test samples on training set
		testorder = shuffle(np.arange(train_data.shape[0]))[:20*batch_size]
		test_x = train_data[testorder]
		test_y = np.tile(np.array([1,0]), (20*batch_size, 1))
		correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print("For 5000 samples from training set, accuracy is:")
		print(sess.run(y_pred, feed_dict={x: test_x, y_real: test_y}))
		'''
		
		test_x = fetch_test_data()
		prediction = sess.run(y_pred, feed_dict={x: test_x})

		# write prediction
		res = []
		for i in range(len(prediction)):
			res.append([i+1, prediction[i][0]])
		with open("pred2.csv", "w", newline='') as f:
			writer = csv.writer(f)
			writer.writerow(["Id", "Predicted"])
			writer.writerows(res)

		# saver.save(sess, "method2/model.ckpt")


train()

