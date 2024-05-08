# -*- coding: utf-8 -*-

import pickle
import math

thre = 0.4

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

#s =0
#per = 2
keep = 1
#for k in range(2,3,1):
k=2
import numpy as np
#for alpha in np.arange(0.2, 0.9, 0.1):
#    for de_times in np.arange(3.0, 3.5, 0.5):
#for m in range(1,5,1):
#        train = "./data_roadnetwork/chengdu/data_effectiveness/train_data_{}_{}.pickle".format(np.round(alpha,1), np.round(de_times,1))
#        train_dir = "./data_roadnetwork/chengdu/train_direction.pickle"
#        train_label = "./data_roadnetwork/chengdu/data_effectiveness/train_label_{}_{}.pickle".format(np.round(alpha,1), np.round(de_times,1))
#        train_feature = "./data_roadnetwork/chengdu/train_fea_{}.pickle".format(thre)
#        data_type = "train"
        #adding data

gamma = 0.2
alpha = 0.4
de_times = 2.5
grop_num = 2
train = "./data_roadnetwork/chengdu/classfier_data/train_data_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_dir = "./data_roadnetwork/chengdu/classfier_data/train_direction_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_label = "./data_roadnetwork/chengdu/classfier_data/train_label_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_feature = "./data_roadnetwork/chengdu/classfier_data/train_fea_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))


unsim_train_data = load_data(train)
unsim_train_direction = load_data(train_dir)
unsim_train_label = load_data(train_label)
unsim_train_fea = load_data(train_feature)

normal_data = []
normal_label=[]
normal_dir=[]
normal_fe =[]

abnormal_data = []
abnormal_label=[]
abnormal_dir=[]
abnormal_fe =[]

final_label_ab = []
final_label_normal = []

for sub_label, sub_data, sub_dir, sub_fe in zip(unsim_train_label, unsim_train_data, unsim_train_direction, unsim_train_fea):
    if sum(sub_label)>0:
        abnormal_data.append(sub_data)
        abnormal_label.append(sub_label)
        abnormal_dir.append(sub_dir)
        abnormal_fe.append(sub_fe)
    else:
        normal_data.append(sub_data)
        normal_label.append(sub_label)
        normal_dir.append(sub_dir)
        normal_fe.append(sub_fe)

import random

abnormal_data_part = abnormal_data

unsim_train_label = normal_label + abnormal_label
unsim_train = normal_data + abnormal_data
unsim_train_direction = normal_dir + abnormal_dir
unsim_train_fe = normal_fe + abnormal_fe

combine = []

for sub_label, sub_data, sub_dir, sub_fe in zip(unsim_train_label,unsim_train, unsim_train_direction, unsim_train_fe):
    combine.append([sub_data, sub_label,sub_dir, sub_fe])
random.shuffle(combine)
print(len(combine))
unsim_train_label = []
unsim_train = []
unsim_train_direction = []
unsim_train_fe = []

for item in combine:
    unsim_train.append(item[0])
    unsim_train_label.append(item[1])
    unsim_train_direction.append(item[2])
    unsim_train_fe.append(item[3])

whole_sim = []
whole_sim_label = []
whole_sim_index = []
whole_sim_fe = []
whole_sim_dir = []

for item,item_fe,item_dir, item_label in zip(unsim_train,unsim_train_fe, unsim_train_direction, unsim_train_label):
    sim_traj = []
    sim_index = []
    sim_label = []
    sim_fe = []
    sim_dir = []
    item_dir.append(0)
    item_dir.insert(0, 0)
    for i in range(len(item)):
        if i == 0 or i == len(item)-1:
            sim_traj.append(item[i])
            sim_label.append(item_label[i])
            sim_index.append(i)
            sim_fe.append(item_fe[i])
            continue
        if i%k < keep :
#            print("i:", i)
#            print("item[i]:",item[i])
#            print("len_item[i]:",len(item_dir))
            sim_traj.append(item[i])
            sim_label.append(item_label[i])
            sim_index.append(i)
            sim_fe.append(item_fe[i])
            if i<len(item_dir):
                sim_dir.append(item_dir[i])
        else:
            if item_label[i]==item_label[i-1] and item_label[i]==item_label[i+1]:
                pass
            else:
                sim_traj.append(item[i])
                sim_label.append(item_label[i])
                sim_index.append(i)
                sim_fe.append(item_fe[i])
                if i<len(item_dir):
                    sim_dir.append(item_dir[i])
                    
    whole_sim.append(sim_traj)
    whole_sim_label.append(sim_label)
    whole_sim_index.append(sim_index)
    whole_sim_fe.append(sim_fe)
    whole_sim_dir.append(sim_dir)
    

#untest_data_combine = unsim_train[:]
#untest_label_combine = unsim_train_label[:]
#untest_fe_combine = unsim_train_fe[:]
#untest_dir_combine = unsim_train_direction[:]


file=open(r"./data_roadnetwork/chengdu/classfier_data/simp_data","wb")
pickle.dump(whole_sim,file) #storing_list
file.close()

file=open(r"./data_roadnetwork/chengdu/classfier_data/simp_label","wb")
pickle.dump(whole_sim_label,file) #storing_list
file.close()

file=open(r"./data_roadnetwork/chengdu/classfier_data/simp_index","wb")
pickle.dump(whole_sim_index,file) #storing_list
file.close()

file=open(r"./data_roadnetwork/chengdu/classfier_data/simp_fe","wb")
pickle.dump(whole_sim_fe,file) #storing_list
file.close()

file=open(r"./data_roadnetwork/chengdu/classfier_data/simp_dir","wb")
pickle.dump(whole_sim_dir,file) #storing_list
file.close()



print("-------------------finish one---------------------")










