# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-05-19 13:26:19
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-30 16:11:06


import preprocess as pp
import numpy as np
import random
import tensorflow as tf
from sklearn import linear_model
from argparse import ArgumentParser


def NN(train, test, epoch=100, show=True):
	 
	train, test = np.array(train), np.array(test)
	X_train, Y_train = train[:, :-1], train[:, -1]
	X_test, Y_test = test[:, :-1], test[:, -1]
	
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

	'''
	print('Testing ------------')
	cost = model.evaluate(X_test, Y_test, batch_size=40)
	print('test cost:', cost)
	W, b = model.layers[1].get_weights()
	print('Weights=', W, 'biases=', b)
	'''

	Y_pred = model.predict(X_test)
	# print(Y_pred, Y_test,"\n=====================================")

	if show:
		zero, two, five, ten = 0, 0, 0, 0
		for i in range(len(Y_test)):
			if abs(Y_test[i] - Y_pred[i]) <= 0.5:
				zero += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 2:
				two += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 5:
				five += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 10:
				ten += 1
		print("Using Nerual Network:")
		print("[Mark difference : Sample number] pairs between prediction and true mark:")
		print("Zero:{} Two:{} Five:{} Ten:{}".format(zero, two, five, ten))
	
	return Y_pred


def BayesianRidge(train, test, show=True):
	train, test = np.array(train), np.array(test)
	X_train, Y_train = train[:, :-1], train[:, -1]
	X_test, Y_test = test[:, :-1], test[:, -1]
	
	# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

	clf = linear_model.BayesianRidge()
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)

	if show:
		zero, two, five, ten = 0, 0, 0, 0
		for i in range(len(Y_test)):
			if abs(Y_test[i] - Y_pred[i]) <= 0.5:
				zero += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 2:
				two += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 5:
				five += 1
			elif abs(Y_test[i] - Y_pred[i]) <= 10:
				ten += 1
		print("Using Machine Learning Method -- BayesianRidge:")
		print("[Mark difference : Sample number] pairs between prediction and true mark:")
		print("Zero:{} Two:{} Five:{} Ten:{}".format(zero, two, five, ten))
	
	return Y_pred


# feature augmentation
# Complete feature space <Common, MALE, FEMALE, MIXED>
def feature_aug(data, domain):
	if domain == "MALE":
		# each row
		for i in range(len(data)):
			tmp = []
			feature = data[i][:-1]
			mark = data[i][-1]
			tmp = feature + feature + [0]*len(feature) + [0]*len(feature)
			data[i] = tmp + [mark]
		return data
	elif domain == "FEMALE":
		# each row
		for i in range(len(data)):
			tmp = []
			feature = data[i][:-1]
			mark = data[i][-1]
			tmp = feature + [0]*len(feature) + feature + [0]*len(feature)
			data[i] = tmp + [mark]
		return data
	elif domain == "MIXED":
		# each row
		for i in range(len(data)):
			tmp = []
			feature = data[i][:-1]
			mark = data[i][-1]
			tmp = feature + [0]*len(feature) + [0]*len(feature) + feature
			data[i] = tmp + [mark]
		return data
	else:
		print("Wrong target")
		return


def FEDA(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:FEDA -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	# 100 subsamples from target domain
	sub_tra = random.sample(tra, 100)

	# feature augmentation
	sub_tra = feature_aug(sub_tra, target)
	dev = feature_aug(dev, target)
	
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		t = feature_aug(t, s)
		train += t

	train += sub_tra
	random.shuffle(train)

	if model == 'NN':
		NN(train, dev, 100)
	elif model == 'ML':
		BayesianRidge(train, dev)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-d", help="domain: MALE/FAMALE/MIXED", dest="domain", default="MALE")
	parser.add_argument("-a",  help="algorithm: NN/ML", dest="al", default="NN")
	args = parser.parse_args()
	FEDA(args.domain, args.al)

	