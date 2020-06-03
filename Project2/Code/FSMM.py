# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-05-19 13:26:19
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-30 16:08:19


import preprocess as pp
import numpy as np
import random
import tensorflow as tf
from argparse import ArgumentParser


# learning the mapping from target domain to source domain
def mapping_model(target, source, epoch=100):

	target, source = np.array(target), np.array(source)
	X_target, Y_target = target[:, :-1], target[:, -1]
	X_source, Y_source = source[:, :-1], source[:, -1]

	output_len = X_source.shape[1]

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(8, activation='relu'))
	model.add(tf.keras.layers.Dense(output_len))

	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	#model.compile(loss='mse', optimizer='sgd')
	 
	print('Training -----------')
	for step in range(epoch):
		np.random.shuffle(X_target)
		np.random.shuffle(X_source)
		# batch size
		target_batch = X_target[:200]
		source_batch = X_source[:200]
		model.fit(target_batch, source_batch, verbose=0)

	return model


# train a regression model on source domain samples (can be seen as SrcOnly model)
def predict_model(train, epoch=100):
	 
	train = np.array(train)
	X_train, Y_train = train[:, :-1], train[:, -1]
	
	# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
	
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(8, activation='relu'))
	model.add(tf.keras.layers.Dense(4, activation='relu'))
	model.add(tf.keras.layers.Dense(1))

	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	#model.compile(loss='mse', optimizer='sgd')
	 
	print('Training -----------')
	for step in range(epoch):
		model.fit(X_train, Y_train, verbose=0)

	return model


def FSMM(target):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:FSMM -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	# 100 subsamples from target domain
	sub_tra = random.sample(tra, 100)

	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t

	random.shuffle(tra)
	random.shuffle(train)

	# train the model on source samples
	pred_mol = predict_model(train)

	# train the model to learn mapping
	map_mol = mapping_model(tra, train)

	dev = np.array(dev)
	X_dev, Y_dev = dev[:, :-1], dev[:, -1]

	# mapping target samples to source samples
	mapping_res = map_mol.predict(X_dev)
	# prediction process
	pred = pred_mol.predict(mapping_res)

	# evaluation
	zero, two, five, ten = 0, 0, 0, 0
	for i in range(len(Y_dev)):
		if abs(Y_dev[i] - pred[i]) <= 0.5:
			zero += 1
		elif abs(Y_dev[i] - pred[i]) <= 2:
			two += 1
		elif abs(Y_dev[i] - pred[i]) <= 5:
			five += 1
		elif abs(Y_dev[i] - pred[i]) <= 10:
			ten += 1
	print("Using FSMM:")
	print("[Mark difference : Sample number] pairs between prediction and true mark:")
	print("Zero:{} Two:{} Five:{} Ten:{}".format(zero, two, five, ten))


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-d", help="domain: MALE/FAMALE/MIXED", dest="domain", default="MALE")
	args = parser.parse_args()
	FSMM(args.domain)

	