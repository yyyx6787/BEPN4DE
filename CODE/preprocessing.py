import pickle
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, auc
from itertools import groupby
from collections import Counter
random.seed(0)

def get_signal(partial, references):
    for reference in references:
        for i in range(len(reference) - len(partial) + 1):
            if reference[i: i+len(partial)] == partial:
                return 0
    return 1

def sort_nomal_rep_SD(normal_representation):
    sorted_normal_representation = {}
    for key, value in normal_representation.items():
        sorted_normal_representation[key] = []
        for item in value:
            if len(item) == 0:
                sorted_normal_representation[key].append([])
                continue
            else:
                tmp = []
                for item_ in item:
                    tmp.append(tuple(item_))
                d2 = Counter(tmp)
                sorted_normal_representation[key].append(sorted(d2.items(), key=lambda x: x[1], reverse=True))
    return sorted_normal_representation

def get_dict_with_num(_labels):
    _dict = {}
    _num = 0
    fun = lambda x: x[1] - x[0]
    for index, label_test in enumerate(_labels):
        if sum(label_test) > 0:
            _dict[index] = []
            lst = [i for i,v in enumerate(label_test) if v==1]
            for k, g in groupby(enumerate(lst), fun):
                _dict[index].append([v for i, v in g])
                _num += len(_dict[index])
    return _dict, _num

def get_correct_num(ground_dict, pred_dict, flag):
    correct_num = 0
    for pd in pred_dict:
        if pd in ground_dict:
            if flag == 'strict':
                correct_num += int(pred_dict[pd]==ground_dict[pd])
            if flag == 'soft':
                listA, listB = sum(ground_dict[pd], []), sum(pred_dict[pd], [])
                correct_num += len(set(listA).intersection(set(listB)))/len(set(listA).union(set(listB)))
    return correct_num

def my_fscore_whole_determine(Label_Test, Label_Pred, flag='soft'):
    ground_num, pred_num, correct_num = 0, 0, 0
    for label_test, label_pred in zip(Label_Test, Label_Pred):
        #print(label_test, label_pred)
        if sum(label_test) == 0 and sum(label_pred) == 0:
            continue
        #print('label_test, label_pred', label_test, label_pred)
        label_test, label_pred = label_test[1:-1], label_pred[1:-1]
        fun = lambda x: x[1] - x[0]
        listA, listB = [], []
        lstA = [i for i,v in enumerate(label_test) if v==1]
        lstB = [i for i,v in enumerate(label_pred) if v==1]
        for k, g in groupby(enumerate(lstA), fun):
            listA.append([v for i, v in g])
        for k, g in groupby(enumerate(lstB), fun):
            listB.append([v for i, v in g])
        ground_num += len(listA)
        pred_num += len(listB)
        listA = sum(listA, [])
        listB = sum(listB, [])
        if flag == 'strict':
            correct_num += int(listA==listB)
        if flag == 'soft':
            if len(set(listA).union(set(listB))) == 0.0:
                correct_num += 0
            else:
                correct_num += len(set(listA).intersection(set(listB)))/len(set(listA).union(set(listB)))
    if pred_num == 0 or ground_num == 0:
        return 0.0
    else:
        #print('ground_num, pred_num, correct_num', ground_num, pred_num, correct_num)
        precision = correct_num/pred_num
        recall = correct_num/ground_num
        if precision+recall==0:
            return 0.0
        else:
            #print('correct_num, pred_num, ground_num', listA, listB, correct_num, pred_num, ground_num)
            F1 = 2*((precision*recall)/(precision+recall))
            #print(label_test, label_pred, F1)
            return F1

def SD_Data(slot=24, clean=5):
    c = 0
    for key, value in raw_data.items():
        c+=1
        if c%5000 == 0:
            print('process', c, '/', len(raw_data))
        if value['match'] == False or len(value['cpath'])<2:
            continue
        pair = (value['cpath'][0], value['cpath'][-1])
        hour = time.localtime(value['tms'][0]).tm_hour
        if pair in SD_pair_data:
            endidx = value['cpath'].index(pair[1])
            staidx = len(value['cpath'][0:endidx+1]) - 1 -  value['cpath'][0:endidx+1][::-1].index(pair[0])
            route = value['cpath'][staidx:endidx+1]
            route = sorted(set(route),key=route.index)
            if has_the_route_in_G(route) and len(route)>=2:
                SD_pair_data[pair][hour%slot].append(route)
                SD_pair_time[pair][hour%slot].append(value['tms'][-1] - value['tms'][0])
        else:
            SD_pair_data[pair] = [[] for i in range(slot)]
            SD_pair_time[pair] = [[] for i in range(slot)]
            endidx = value['cpath'].index(pair[1])
            staidx = len(value['cpath'][0:endidx+1]) - 1 -  value['cpath'][0:endidx+1][::-1].index(pair[0])     
            route = value['cpath'][staidx:endidx+1]
            route = sorted(set(route),key=route.index)
            if has_the_route_in_G(route) and len(route)>=2:
                SD_pair_data[pair][hour%slot].append(route)
                SD_pair_time[pair][hour%slot].append(value['tms'][-1] - value['tms'][0])
    print('Done SD pair data and start clean')
    collect_clean = []
    for key, value in SD_pair_data.items():
        c = 0 
        for val in value:
            c+=len(val)
        if c < clean:
            collect_clean.append(key)
    for key in collect_clean:
        del SD_pair_data[key]
        del SD_pair_time[key]

def SD_distribution():
    SD_rows, SD_cols = {}, {}
    for key, value in SD_pair_data.items():
        SD_cols[key] = []
        for val in value:
            SD_cols[key].append(len(val))
        SD_rows[key] = sum(SD_cols[key])
    print('Done sample table')
    return SD_rows, SD_cols

def random_index(rate):
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def get_normal_route():
    sample_pair, sample_slot = SD_sampling()
    normal_SD_set = SD_pair_data[sample_pair][sample_slot]
    #print('len', len(normal_SD_set))
    return normal_SD_set[random.randint(0, len(normal_SD_set)-1)]

def SD_sampling():
    SD_rows, SD_cols = sample_table[0], sample_table[1]
    PAIR = list(SD_rows.keys())
    index = random_index(SD_rows.values())
    sample_pair = PAIR[index]
    sample_slot = random_index(SD_cols[sample_pair])
    return sample_pair, sample_slot
    
def verify_distribution():
    A = {}
    for count in range(1000):
        sample_pair, sample_slot = SD_sampling()
        if sample_pair in A:
            A[sample_pair]+=1
        else:
            A[sample_pair]=1
    return A 

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
    
def get_detour_route(alpha=0.2):
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
        if path != normal_seg and get_travel_distance(path) > get_travel_distance(normal_seg):
            detour_route = normal_route[0:A] + path + normal_route[B+1:len(normal_route)]
            label, normalS, normalD, detourS, detourD = labelling(normal_route, detour_route, A, A+len(path)-1)
            if normalS <= normalD and detourS <= detourD:
                return [normal_route, detour_route, label, normalS, normalD, detourS, detourD]
    return []

def ger_detour_batch(alpha=0.2, batch=10):
    count = batch
    detour_data_batch = []
    while count !=0 :
        tmp = get_detour_route(alpha)
        if tmp != []:
            detour_data_batch.append(tmp)
            count -= 1
    return detour_data_batch

def get_average_len(detour_data_batch):
    detour_len= 0
    for c in detour_data_batch:
        detour_len += (c[4]-c[3]+1)
    print('average detour len', detour_len/len(detour_data_batch))
    return detour_len/len(detour_data_batch)

def _load_data(alpha, batch_size):
    #print('Creating %s data...' % batch_size)
    label, text, sentence_len = [], [], []
    detour_data_batch = ger_detour_batch(alpha, batch_size)    
    for ddb in detour_data_batch:
        label.extend(ddb[2])
        for idx in ddb[1]:
            text.append([idx])
        sentence_len.append(len(ddb[1]))
    return label, text, sentence_len

def has_the_route_in_G(route):
    for r in route:
        if r not in G:
            return False
    return True

def is_decision(pre, suf, last_label):
    #output: is_decision, the decision label
    #print('pre, suf, last_label', pre, suf, last_label)
    #one-to-more
    if G.out_degree(pre)>1 and G.in_degree(suf)==1:
        #print('case one-to-more')
        if last_label==1:
            return [False, last_label]
        else:
            return [True, None]
    #one-to-one
    if G.out_degree(pre)==1 and G.in_degree(suf)==1:
        #print('case one-to-one')
        return [False, last_label]
    #more-to-one
    if G.out_degree(pre)==1 and G.in_degree(suf)>1:
        #print('case more-to-one')
        if last_label==0:
            return [False, last_label]
        else:
            return [True, None]
    #more-to-more
    if G.out_degree(pre)>1 and G.in_degree(suf)>1:
        #print('case more-to-more')
        return [True, None]
    #print('case unknow')
    return [True, None]

def prepare_data(plt, S, D, slot, paths):
    save = traj_path+'manually/'+str(S)+'_'+str(D)+'_'+str(slot)+'_'+str(len(paths))+'_'+str(time.time())
    os.makedirs(save)
    plt.savefig(save+'/vis.jpg',dpi=300)
    plt.show()
    path_dict = {}
    for path in paths:
        if tuple(path) in path_dict:
            path_dict[tuple(path)]+=1
        else:
            path_dict[tuple(path)]=1
    #paths = [list(t) for t in set(tuple(_) for _ in paths)]
    c = 0
    is_manually = False
    threshold = 0.3
    if is_manually:
        for path in path_dict:
            f = open(save+'/'+str(c)+'.txt','w')
            f.write('The labelling path occurs {} times in total {} paths \n'.format(path_dict[path],len(paths)))
            for idx in range(len(path)):
                if idx == 0 or idx == len(path)-1:
                    f.write(str(path[idx])+',0'+'\n') #0 normal and 1 abnormal
                else:
                    f.write(str(path[idx])+','+'\n')
            f.close()
            c+=1
    else:
        path_list = sorted(path_dict.items(), key=lambda x: x[1], reverse=True)
        normal = list(path_list[0][0])
        f = open(save+'/'+str(c)+'.txt','w')
        f.write('The labelling path occurs {} times in total {} paths \n'.format(path_list[0][1],len(paths)))
        for idx in range(len(normal)):
            f.write(str(normal[idx])+',0'+'\n') #0 normal and 1 abnormal
        f.close()
        c+=1
        for idx in range(1, len(path_list)):
            route, fre = path_list[idx][0], path_list[idx][1]
            f = open(save+'/'+str(c)+'.txt','w')
            f.write('The labelling path occurs {} times in total {} paths \n'.format(fre,len(paths)))
            if fre/len(paths) > threshold:
                for idx_ in range(len(route)):
                    f.write(str(route[idx_])+',0'+'\n') #0 normal and 1 abnormal
            else:
                label = [0]*len(route)
                for i in range(len(route)):
                    if normal[i] != route[i]:
                        detourS = i
                        break
                for i in range(len(route)):
                    if normal[::-1][i] != route[::-1][i]:
                        detourD = len(route)-1-i
                        break
                label[detourS:detourD+1] = [1]*(detourD-detourS+1)
                for idx_ in range(len(route)):
                    f.write(str(route[idx_])+','+str(label[idx_])+'\n') #0 normal and 1 abnormal
            f.close()
            c+=1

def get_file_path(root_path,file_list,dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path,dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)
    return file_list, dir_list

def obtain_groundtruth():
    file_list = []
    dir_list = []
    groundtruth_table = {}
    file_list, dir_list = get_file_path(groundtruth_path, file_list, dir_list)
    NUM, NUM_detour = 0, 0
    abroads, roads = 0, 0
    for fl in file_list:
        tmp = fl.split(groundtruth_path)[1]
        if '.txt' not in tmp:
            continue
        tmp_ = tmp.split('_')
        S, D, slot, num = int(tmp_[0]), int(tmp_[1]), int(tmp_[2]), int(tmp_[3])
        NUM+=num
        f = open(fl)
        line_count = 0
        traj, label = [],[]
        line_tmp = ''
        for line in f:
            line_count+=1
            if line_count == 1:
                line_tmp = line
                continue
            temp = line.strip().split(',')
            traj.append(int(temp[0]))
            label.append(int(temp[1]))
        if sum(label)>0:
            NUM_detour+=int(line_tmp.split(' ')[4])
            abroads += (sum(label)*int(line_tmp.split(' ')[4]))
        roads += (len(traj)*num)
        f.close()
        groundtruth_table[(S,D,tuple(traj))] = label
    print('There are {} detours (with {} abnormal raods) in total {} paths (with {} roads)'.format(NUM_detour,abroads,NUM,roads))
    return groundtruth_table

def refine_groundtruth(groundtruth_table):
    re_groundtruth_table = {}
    record = []
    for key, label in groundtruth_table.items():
        route = key[2]
        if has_the_route_in_G(tuple(route)):
            relabel = [0]
            for i in range(1, len(route)-1):
                is_dec, tag = is_decision(route[i-1], route[i], relabel[i-1])
                if is_dec:
                    relabel.append(label[i])
                else:
                    relabel.append(tag)    
            relabel.append(0)
            re_groundtruth_table[key] = relabel
            if relabel != label:
                record.append([key, label, relabel])
        else:
            print('not in G')
            re_groundtruth_table[key] = label
    return re_groundtruth_table, record

if __name__ == '__main__':
    city = 'chengdu'
    traj_path = './data_roadnetwork/'+str(city)+'/'
    groundtruth_path = './'+str(city)+'_detour/'
    slot = 24
    
    G = nx.read_adjlist(traj_path+str(city)+".adjlist", create_using=nx.DiGraph, nodetype=int)
    data_dict = pickle.load(open(traj_path+'data_dict_'+str(city)+'.pkl', 'rb'), encoding='bytes')
    
    #note that it is very slow due to the big metadata
    '''
    SD_pair_data, SD_pair_time = {}, {}
    for name in ['20161101.pickle', '20161102.pickle', '20161103.pickle', '20161104.pickle', '20161105.pickle']:
        raw_data = pickle.load(open(traj_path+name, 'rb'), encoding='bytes')
        SD_Data(slot)
        print('len(SD_pair_data), len(SD_pair_time)', len(SD_pair_data), len(SD_pair_time))
    pickle.dump(SD_pair_data, open(traj_path+'SD_pair_data_'+str(slot), 'wb'), protocol=2)
    '''
    
    SD_pair_data = pickle.load(open(traj_path+'SD_pair_data_'+str(slot), 'rb'), encoding='bytes')
    
    SD_rows, SD_cols = SD_distribution()
    pickle.dump((SD_rows, SD_cols), open(traj_path+'SD_pair_'+str(slot)+'_sample_table', 'wb'), protocol=2)
    
    sample_table = pickle.load(open(traj_path+'SD_pair_'+str(slot)+'_sample_table', 'rb'), encoding='bytes')
    pickle.dump(sample_table, open(traj_path+'sample_table'+str(slot), 'wb'), protocol=2)
    
    #verify distribution sampling
    verifier = verify_distribution()
    
    detour_len= 0
    detour_data = get_detour_route(alpha=0.2)    
    alpha, batch = 0.3, 10
    detour_data_batch = ger_detour_batch(alpha, batch)
    for c in detour_data_batch:
        detour_len += (c[4]-c[3]+1)
    print('average detour len', detour_len/batch, 'with alpha', alpha)
    
    SD_reference = sort_nomal_rep_SD(SD_pair_data)
    pickle.dump(SD_reference, open(traj_path+'SD_reference_'+str(slot), 'wb'), protocol=2)
    
    groundtruth_table = obtain_groundtruth()
    groundtruth_table, record = refine_groundtruth(groundtruth_table)
    pickle.dump(groundtruth_table, open(traj_path+'groundtruth_table_'+str(slot), 'wb'), protocol=2)
        