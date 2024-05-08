# -*- coding: utf-8 -*-

import pickle
import numpy as np
import networkx as nx
import random
random.seed(0)

city = 'chengdu'
traj_path = './data_roadnetwork/'+str(city)+'/'
slot = 24
G = nx.read_adjlist(traj_path+str(city)+".adjlist", create_using=nx.DiGraph, nodetype=int)
data_dict = pickle.load(open(traj_path+'data_dict_'+str(city)+'.pkl', 'rb'), encoding='bytes')
SD_pair_data = pickle.load(open(traj_path+'SD_pair_data_'+str(slot), 'rb'), encoding='bytes')
sample_table = pickle.load(open(traj_path+'SD_pair_'+str(slot)+'_sample_table', 'rb'), encoding='bytes')

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def SD_sampling():
    SD_rows, SD_cols = sample_table[0], sample_table[1]
    PAIR = list(SD_rows.keys())
    index = random_index(SD_rows.values())
    sample_pair = PAIR[index]
    sample_slot = random_index(SD_cols[sample_pair])
    return sample_pair, sample_slot

def get_normal_route():
    sample_pair, sample_slot = SD_sampling()
    normal_SD_set = SD_pair_data[sample_pair][sample_slot]
    #print('len', len(normal_SD_set))
    return normal_SD_set[random.randint(0, len(normal_SD_set)-1)]

def get_travel_distance(route):
    travel = 0
    for a in route:
        travel += data_dict['road_lengths'][a]
    return travel

def labelling(normal_route, detour_route, A, B):
    label = [0]*len(detour_route)
    normalS, normalD, detourS, detourD = -1, -1, -1, -1
    for i in range(A, len(detour_route)):
        if normal_route[i] != detour_route[i]:
            detourS = i
            normalS = i
            break
    for i in range(-len(detour_route)+B, -len(detour_route)-1, -1):
        if normal_route[i] != detour_route[i]:
            detourD = len(detour_route)+i
            normalD = len(normal_route)+i
            break
    label[detourS:detourD+1] = [1]*(detourD-detourS+1)
    return label, normalS, normalD, detourS, detourD

def get_detour_route(alpha=0.2, para = 2.5):
    #print('detour')
    normal_route = get_normal_route()
    detour_len = max(int(len(normal_route)*alpha), 1)
    start = random.randint(0, int(len(normal_route)*0.5))
    A, B = start, start+detour_len
    normal_seg = normal_route[A:B+1]
    #print(normal_route, A, B, normal_seg)
    if normal_seg[0] not in G or normal_seg[-1] not in G:
        return []
    paths = list(nx.all_simple_paths(G, normal_seg[0], normal_seg[-1], 10))
    #print('Done paths', paths)
    for path in paths:
        if path != normal_seg and get_travel_distance(path) > para*get_travel_distance(normal_seg):
            detour_route = normal_route[0:A] + path + normal_route[B+1:len(normal_route)]
            label, normalS, normalD, detourS, detourD = labelling(normal_route, detour_route, A, A+len(path)-1)
            if normalS <= normalD and detourS <= detourD:
                return [normal_route, detour_route, label, normalS, normalD, detourS, detourD]
    return []

def ger_detour_batch(alpha=0.2, batch=10, para = 2.5):
    count = batch
    detour_data_batch = []
    while count !=0 :
        tmp = get_detour_route(alpha, para)
        if tmp != []:
            detour_data_batch.append(tmp)
            count -= 1
    return detour_data_batch

if __name__ == '__main__':
    slot = 24
    SD_pair_data = pickle.load(open(traj_path+'SD_pair_data_'+str(slot), 'rb'), encoding='bytes')
    alpha, de_times = 0.4, 2.5
    detour_ = ger_detour_batch(np.round(alpha,1), 5000, np.round(de_times,1))

    detour_combine = []
    for ii in detour_:
        detour_combine.append([ii[1], ii[2]])    
    normal_data = []
    
#    for i in range(5000):
#        l1 = get_normal_route()
#        tmp_label = [0]*len(l1)
#        normal_data.append([l1,tmp_label])
        
    combine = normal_data + detour_combine
    
    np.random.shuffle(combine)
    train_data = []
    train_label =[]
    for item in combine:
        train_data.append(item[0])
        train_label.append(item[1])
        
    file=open(r"./data_roadnetwork/"+str(city)+"/train_data_{}_{}_n.pickle".format(np.round(alpha,1), np.round(de_times,1)),"wb")
    pickle.dump(train_data,file) #storing_list
    file.close()
    
    file=open(r"./data_roadnetwork/"+str(city)+"/train_label_{}_{}_n.pickle".format(np.round(alpha,1), np.round(de_times,1)),"wb")
    pickle.dump(train_label,file) #storing_list
    file.close()    
    
    
