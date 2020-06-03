# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-05-19 13:26:19
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-25 15:03:07


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


def SrcOnly(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:SrcOnly -- Domain:{}".format(target))
	src = []
	for i in domains:
		if i != target:
			src.append(i)

	_, dev, _ = pp.get_splited_data("{}.csv".format(target))
	
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t

	random.shuffle(train)

	if model == 'NN':
		NN(train, dev, 100)
	elif model == 'ML':
		BayesianRidge(train, dev)


def TgtOnly(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:TgtOnly -- Domain:{}".format(target))

	train, dev, _ = pp.get_splited_data("{}.csv".format(target))

	random.shuffle(train)

	if model == 'NN':
		NN(train, dev, 100)
	elif model == 'ML':
		BayesianRidge(train, dev)


def All(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:All -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t

	train += random.sample(tra, 100)
	random.shuffle(train)

	if model == 'NN':
		NN(train, dev, 100)
	elif model == 'ML':
		BayesianRidge(train, dev)


def Weighted(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:Weighted -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t

	# weight proportion constant
	# Actually, w = 1 / (len(train) / 100)
	# where len(train) is the number of source samples and 100 is the number of target sub-samples
	w_constant = 100 / len(train)

	# weighted process
	for i in range(len(train)):
		# Don't change the label value
		for j in range(len(train[0])-1):
			train[i][j] = round(w_constant*train[i][j], 2)

	# add target samples without weighting
	train += random.sample(tra, 100)
	random.shuffle(train)

	if model == 'NN':
		NN(train, dev, 100)
	elif model == 'ML':
		BayesianRidge(train, dev)


def Pred(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:Pred -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	target = tra + dev

	# trained on source domain
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t
	random.shuffle(train)

	if model == 'NN':
		pred = NN(train, target, 100, False)

		# add predicted feature
		for i in range(len(target)):
			target[i] = [round(pred[i][0], 2)] + target[i]

		# split updated target domain samples (last 100 samples are dev samples)
		new_tra, new_dev = target[:-100], target[-100:]

		random.shuffle(new_tra)
		# trained on new target domain
		NN(new_tra, new_dev, 100)
	elif model == 'ML':
		pred = BayesianRidge(train, target, False)
		print(pred)
		# add predicted feature
		for i in range(len(target)):
			target[i] = [round(pred[i], 2)] + target[i]

		# split updated target domain samples (last 100 samples are dev samples)
		new_tra, new_dev = target[:-100], target[-100:]

		random.shuffle(new_tra)
		# trained on new target domain
		BayesianRidge(new_tra, new_dev)

def LinInt(target, model='ML'):

	# three domains
	domains = ['MALE', 'FEMALE', 'MIXED']

	if target not in domains:
		print("Wrong target !")
		return

	print("Model:LinInt -- Domain:{}".format(target))

	src = []
	for i in domains:
		if i != target:
			src.append(i)

	tra, dev, _ = pp.get_splited_data("{}.csv".format(target))
	
	train = []
	for s in src:
		t, _, _ = pp.get_splited_data("{}.csv".format(s))
		train += t

	random.shuffle(train)

	if model == 'NN':
		# SrcOnly pred
		pred1 = NN(train, dev, 100, False)
		# TgtOnly pred
		pred2 = NN(tra, dev, 100, False)

		two_pred = np.hstack((pred1, pred2))
		#print(pred1, pred2)
		#print(two_pred, two_pred.shape)

		# final pred = w1*pred1 + w2*pred2 + bias
		# train a new model to get W and b
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(1))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		dev_label = np.array(dev)[:, -1]
		# print(dev_label,dev_label.shape)
		for _ in range(500):
			model.fit(two_pred, dev_label, verbose=0)

		W, b = model.layers[0].get_weights()

		# use W and b to interpolate SrcOnly and TgtOnly predictions
		new_pred1 = NN(train, dev, 100, False)
		new_pred2 = NN(tra, dev, 100, False)
		new_two_pred = np.hstack((new_pred1, new_pred2))
		# print(W, b)
		final_pred = np.dot(new_two_pred, W) + b

		zero, two, five, ten = 0, 0, 0, 0
		for i in range(len(final_pred)):
			if abs(dev_label[i] - final_pred[i]) <= 0.5:
				zero += 1
			elif abs(dev_label[i] - final_pred[i]) <= 2:
				two += 1
			elif abs(dev_label[i] - final_pred[i]) <= 5:
				five += 1
			elif abs(dev_label[i] - final_pred[i]) <= 10:
				ten += 1
		print("Using Nerual Network:")
		print("[Mark difference : Sample number] pairs between prediction and true mark:")
		print("Zero:{} Two:{} Five:{} Ten:{}".format(zero, two, five, ten))
	if model == 'ML':
		# SrcOnly pred
		pred1 = BayesianRidge(train, dev, False)
		# TgtOnly pred
		pred2 = BayesianRidge(tra, dev, False)
		pred1 = pred1.reshape(100,1)
		pred2 = pred2.reshape(100,1)
		two_pred = np.hstack((pred1, pred2))

		dev_label = np.array(dev)[:, -1]
		clf = linear_model.BayesianRidge()
		# clf to learn to combine the pred of SrcOny and TgtOnly
		clf.fit(two_pred, dev_label)

		new_pred1 = BayesianRidge(train, dev, False)
		new_pred2 = BayesianRidge(tra, dev, False)
		new_pred1 = new_pred1.reshape(100,1)
		new_pred2 = new_pred2.reshape(100,1)
		new_two_pred = np.hstack((new_pred1, new_pred2))
		final_pred = clf.predict(new_two_pred)

		zero, two, five, ten = 0, 0, 0, 0
		for i in range(len(final_pred)):
			if abs(dev_label[i] - final_pred[i]) <= 0.5:
				zero += 1
			elif abs(dev_label[i] - final_pred[i]) <= 2:
				two += 1
			elif abs(dev_label[i] - final_pred[i]) <= 5:
				five += 1
			elif abs(dev_label[i] - final_pred[i]) <= 10:
				ten += 1
		print("Using Machine Learning Method -- BayesianRidge:")
		print("[Mark difference : Sample number] pairs between prediction and true mark:")
		print("Zero:{} Two:{} Five:{} Ten:{}".format(zero, two, five, ten))


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("-b",  help="baseline: 1(SrcOnly)\
											2(TgtOnly)\
											3(All)\
											4(Weighted)\
											5(Pred)\
											6(LinInt)", dest="bl", default="1")
	parser.add_argument("-d", help="domain: MALE/FAMALE/MIXED", dest="domain", default="MALE")
	parser.add_argument("-a",  help="algorithm: NN/ML", dest="al", default="ML")
	args = parser.parse_args()
	if args.bl == '1':
		SrcOnly(args.domain, args.al)
	elif args.bl == '2':
		TgtOnly(args.domain, args.al)
	elif args.bl == '3':
		All(args.domain, args.al)
	elif args.bl == '4':
		Weighted(args.domain, args.al)
	elif args.bl == '5':
		Weighted(args.domain, args.al)
	elif args.bl == '6':
		LinInt(args.domain, args.al)
	else:
		print("Wrong Parameters!")