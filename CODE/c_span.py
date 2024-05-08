# -*- coding: utf-8 -*-

import numpy as np
import pickle
from itertools import groupby
from collections import Counter

def euli(a, b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a-b)
    return distance

def my_fscore_determine(label_test, label_pred, flag='soft'):
    label_test, label_pred = label_test[1:-1], label_pred[1:-1]
    fun = lambda x: x[1] - x[0]
    listA, listB = [], []
    lstA = [i for i,v in enumerate(label_test) if v==1]
    lstB = [i for i,v in enumerate(label_pred) if v==1]
    for k, g in groupby(enumerate(lstA), fun):
        listA.append([v for i, v in g])
    for k, g in groupby(enumerate(lstB), fun):
        listB.append([v for i, v in g])
    ground_num = len(listA)
    pred_num = len(listB)
    if ground_num == 0 or pred_num == 0:
        return 0.0
    listA = sum(listA, [])
    listB = sum(listB, [])
    if flag == 'strict':
        correct_num = int(listA==listB)
    if flag == 'soft':
        correct_num = len(set(listA).intersection(set(listB)))/len(set(listA).union(set(listB)))
    precision = correct_num/pred_num
    recall = correct_num/ground_num
    if precision + recall != 0:
        F1 = 2*((precision*recall)/(precision+recall))
        return F1
    else:
        return 0.0

def skip_span(ref_traj, traj, threshold):
    feature_special = []
    feature_special.append(0)
    for c in range(1, len(traj)):
        distance = []
        for q in range(len(ref_traj)):
            distance.append(euli(traj[c],ref_traj[q]))

        if  min(distance)> threshold:
            feature_special.append(1)
        else:
            feature_special.append(0)

    return feature_special

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file


city = 'chengdu'

t, alpha, de_times = 0.4, 0.4, 2.5
para = "train"
slot = 24
sd = load_data("./data_roadnetwork/"+str(city)+"/SD_pair_data_"+str(slot))
val_data = load_data("./data_roadnetwork/"+str(city)+"/{}_data_{}_{}_n.pickle".format(para,np.round(alpha,1),np.round(de_times,1)))
val_label = load_data("./data_roadnetwork/"+str(city)+"/{}_label_{}_{}_n.pickle".format(para,np.round(alpha,1),np.round(de_times,1)))

val_data_position = []
val_feature = []
from collections import defaultdict
sd_ref = defaultdict(list)
for key,value in sd.items():

    value_ = []
    for item in value:
        if item !=[]:
            for iii in item:
                value_.append(iii)

    trajs = value_
    trajs_len = len(trajs)

    trajs_ex = [sub for item in trajs for sub in item]
    ex = Counter(trajs_ex)

    l1 = [(k, min(v/len(trajs), 1.0)) for k,v in ex.items()]

    for sub_tuple in l1:
        sd_ref[(key,sub_tuple[0])] = sub_tuple[1]

val_data_feature =[]
for sub,su in zip(val_data, val_label):
    fe =[]
    for item in sub:
        if ((sub[0], sub[-1]),item) in sd_ref.keys() and sd_ref[((sub[0], sub[-1]),item)]>t:
            fe.append(0)
        else:
            fe.append(1)
    val_data_feature.append(fe)

file=open(r"./data_roadnetwork/"+str(city)+"/{}_fea_{}_{}_n.pickle".format(para, np.round(alpha,1),np.round(de_times,1)),"wb")
pickle.dump(val_data_feature,file) #storing_list
file.close()
    
    
    
    
