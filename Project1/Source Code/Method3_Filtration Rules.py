# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-04-13 12:10:40
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-04-22 10:16:28


import numpy as np
import json
import os
import random
import tensorflow as tf
import csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def get_raw_data():
	dic = {}
	with open("train.txt", "r") as f:
		for i in f.readlines():
			array = i.split()
			if array:
				dic[array[0]] = array[1:]
	return dic


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


def get_relation(n1, n2, node_info):
	
	if n1 not in node_info or n2 not in node_info:
		print("some node not in node_info")
		return None
	n1f, n2f = node_info[n1], node_info[n2]
	kw_overlap, kw_diff = 0, 0
	vn_overlap, vn_diff = 0, 0
	for i in range(3, 57):
		# both are 1
		if n1f[i] + n2f[i] == 2:
			kw_overlap += 1
		# one of them is 1
		elif n1f[i] + n2f[i] == 1:
			kw_diff += 1
	for i in range(57, len(n1f)):
		# both are 1
		if n1f[i] + n2f[i] == 2:
			vn_overlap += 1
		# one of them is 1
		elif n1f[i] + n2f[i] == 1:
			vn_diff += 1
	return [n1f[:3], n2f[:3], kw_overlap, kw_diff, vn_overlap, vn_diff]


def visualization_raw_data():
	raw_graph = get_raw_data()
	node_info = extract_node_info()

	x, overlap, diff = [], [], []
	for key,val in raw_graph.items():
		for j in val:
			res = get_relation(key, j, node_info)
			if res[-2] < 10:
				print(key,j,res)
			if res[-1] > 40:
				print(key,j,res)


def get_sim(n1, n2, node_info):
	rela = get_relation(n1, n2, node_info)
	overlap, diff = rela[-2], rela[-1]
	if overlap < 8:
		return random.uniform(0, 0.1)
	if diff > 3*overlap and diff > 50:
		return random.uniform(0, 0.1)
	return random.uniform(0.98, 1)


# self-defined similarity
def predict_by_sim_of_features():
	raw_graph = get_raw_data()
	node_info = extract_node_info()
	predicted = []
	with open("test-public.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			key, val = line[1], line[2]
			if key in raw_graph:
				if val in raw_graph[key]:
					prob = random.uniform(0.98, 1)
				else:
					prob = get_sim(key, val, node_info)
			elif val in raw_graph:
				if key in raw_graph[val]:
					prob = random.uniform(0.98, 1)
				else:
					prob = get_sim(key, val, node_info)
			else:
				prob = get_sim(key, val, node_info)

			predicted.append([line[0], round(prob, 6)])

	with open("pred3_sof.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["Id", "Predicted"])
		writer.writerows(predicted)


def depth_of_coauthor(n1, n2, raw_graph):
	depth0, depth1 = 0, 0
	if n1 not in raw_graph or n2 not in raw_graph:
		print("some node not in raw_graph")
		return None
	if n1 in raw_graph[n2] or n2 in raw_graph[n1]:
		depth0 = 1
	n1_co, n2_co = raw_graph[n1], raw_graph[n2]
	for i in n1_co:
		if i in n2_co:
			depth1 += 1
	return {"d0":depth0, "d1":depth1}


def test_data_relation():
	raw_graph = get_raw_data()
	node_info = extract_node_info()
	predicted = []
	with open("test-public.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			key, val = line[1], line[2]
			if key in raw_graph:
				if val in raw_graph[key]:
					print(line[0],key,val,get_relation(key,val,node_info),depth_of_coauthor(key,val,raw_graph))
				else:
					print(line[0],key,val,get_relation(key,val,node_info),depth_of_coauthor(key,val,raw_graph),"???")
			elif val in raw_graph:
				if key in raw_graph[val]:
					print(line[0],key,val,get_relation(key,val,node_info),depth_of_coauthor(key,val,raw_graph))
				else:
					print(line[0],key,val,get_relation(key,val,node_info),depth_of_coauthor(key,val,raw_graph),"???")
			else:
				print(line[0],key,val,get_relation(key,val,node_info),depth_of_coauthor(key,val,raw_graph),"???")


def pred_by_num_of_coauthor():
	raw_graph = get_raw_data()
	node_info = extract_node_info()
	predicted = []
	with open("test-public.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			key, val = line[1], line[2]
			if key in raw_graph:
				if val in raw_graph[key]:
					prob = random.uniform(0.98, 1)
				else:
					if depth_of_coauthor(key,val,raw_graph) and depth_of_coauthor(key,val,raw_graph)["d1"] > 0:
						prob = random.uniform(0.98, 1)
					else:
						prob = random.uniform(0, 0.01)
			elif val in raw_graph:
				if key in raw_graph[val]:
					prob = random.uniform(0.98, 1)
				else:
					if depth_of_coauthor(key,val,raw_graph) and depth_of_coauthor(key,val,raw_graph)["d1"] > 0:
						prob = random.uniform(0.98, 1)
					else:
						prob = random.uniform(0, 0.01)
			else:
				prob = random.uniform(0, 0.01)

			predicted.append([line[0], round(prob, 6)])

	with open("pred3_noa.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["Id", "Predicted"])
		writer.writerows(predicted)


def count_real_edges():
	cnt = 0
	with open("pred3-tune.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			if float(line[1]) > 0.5:
				cnt += 1
	print(cnt)


def pred_info():
	res = {}
	with open("pred3-tune.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			res[line[0]] = float(line[1])
	return res


def change_pred_prob():
	pif = pred_info()
	res = []
	for i in range(1, len(pif)+1):
		i = str(i)
		if pif[i] < 0.02:
			res.append([i, round(random.uniform(0.03, 0.12),6)])
		elif 0.98 < pif[i]:
			res.append([i, round(random.uniform(0.88, 0.98),6)])
		else:
			res.append([i, pif[i]])

	with open("pred3-tune.csv", "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(["Id", "Predicted"])
		writer.writerows(res)


# filtration function, help to find targeted edges
def filtration():
	raw_graph = get_raw_data()
	node_info = extract_node_info()
	predicted = []
	pif = pred_info()

	with open("test-public.csv", "r") as cf:
		cfread = csv.reader(cf)
		for line in cfread:
			if line[0] == "Id": continue
			key, val = line[1], line[2]

			rl = get_relation(key,val,node_info)
			dp = depth_of_coauthor(key,val,raw_graph)
			if key in raw_graph:
				if val in raw_graph[key]:
					prob = random.uniform(0.98, 1)
				else:
					cond1 = rl[0][1] > rl[1][0] or rl[0][0] < rl[1][1]
					if not cond1 and (dp and 3 > dp["d1"] > 0) and rl[3] > rl[2] and rl[4] == 1:
						if pif[line[0]] > 0.5:
							print(line[0], rl,dp,pif[line[0]])
			elif val in raw_graph:
				if key in raw_graph[val]:
					prob = random.uniform(0.98, 1)
				else:
					cond1 = rl[0][1] > rl[1][0] or rl[0][0] < rl[1][1]
					if not cond1 and (dp and 3 > dp["d1"] > 0) and rl[3] > rl[2] and rl[4] == 1:
						if pif[line[0]] > 0.5:
							print(line[0], rl,dp,pif[line[0]])
			else:
				continue


#filtration()
#count_real_edges()
#change_pred_prob()
pred_by_num_of_coauthor()

