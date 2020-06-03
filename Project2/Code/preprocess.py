# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-05-15 18:53:30
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-19 14:25:06

import csv
import random


# filename: 'MALE.csv' / 'FEMALE.csv' / 'MIXED.csv'
def get_raw_data(filename):
	res = []
	with open(filename, "r") as f:
		reader = csv.reader(f)
		for item in reader:
			# skip the 1st row
			# ['Year', 'FSM', 'VR1 Band', 'VR Band of Student', 
			# 'Ethnic group of student', 'School denomination', 'Exam Score']
			if reader.line_num == 1:
				continue
			item = encode_vector(list(map(int, item)))
			res.append(item)
	return res


# filename: 'MALE.csv' / 'FEMALE.csv' / 'MIXED.csv'
def get_splited_data(filename):

	res = get_raw_data(filename)

	# shuffle
	random.shuffle(res)

	# normally splited by 60(train):20(dev):20(test)
	# but we only need 100 samples for dev
	num_train = int(len(res)*0.7)
	train = res[:num_train]
	dev = res[num_train:num_train+100]
	test = res[num_train+100:]

	return train, dev, test


# find details(such as range) of each feature in raw data
# feature: 0~6 ['Year', 'FSM', 'VR1 Band', 'VR Band of Student',
#				'Ethnic group of student', 'School denomination', 'Exam Score']
def find_details(filename, feature):
	fea = ['Year', 'FSM', 'VR1 Band', 'VR Band of Student', 
			'Ethnic group of student', 'School denomination', 'Exam Score']
	res = get_raw_data(filename)
	val = set()
	for i in res:
		val.add(int(i[feature]))

	print("As for feature \"{}\", the value distributed in: ".format(fea[feature]), sorted(list(val)))


def encode_vector(vec):
	# vec shape: (7,)
	res = []
	feature_Year = [0] * 3
	feature_VR_Band_of_S = [0] * 4
	feature_Ethnic = [0] * 11
	feature_Schoole_de = [0] * 3
	for i in range(len(vec)):
		if i == 0:
			feature_Year[vec[i]-1] = 1
			res += feature_Year
		elif i == 3:
			feature_VR_Band_of_S[vec[i]] = 1
			res += feature_VR_Band_of_S
		elif i == 4:
			feature_Ethnic[vec[i]-1] = 1
			res += feature_Ethnic
		elif i == 5:
			feature_Schoole_de[vec[i]-1] = 1
			res += feature_Schoole_de
		elif i == 6:
			res += [vec[i]]
		# i = 1/2
		else:
			res += [0.01*vec[i]]
	# res shape: (24,) / the last digit is score
	return res

