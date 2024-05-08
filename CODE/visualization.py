# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:33:04 2021

@author: wang_zheng
"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import os

np.random.seed(0)

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

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
    for fl in file_list:
        tmp = fl.split(groundtruth_path)[1]
        if '.txt' not in tmp:
            continue
        tmp_ = tmp.split('_')
        S, D, slot, num = int(tmp_[0]), int(tmp_[1]), int(tmp_[2]), int(tmp_[3])
        #print('fl', fl)
        f = open(fl)
        line_count = 0
        token, traj, label, flag = [], [], [], False
        for line in f:
            line_count+=1
            if line_count == 1:
                pathnum = int(line.split(' ')[4])
                continue
            temp = line.strip().split(',')
            if int(temp[0]) in historical_average_coordinate:
                traj.append(historical_average_coordinate[int(temp[0])])
                token.append(int(temp[0]))
            else:
                #print('miss', int(temp[0]))
                flag = True
                break
            label.append(int(temp[1]))
        f.close()
        if flag:
            continue
        if (S,D) not in groundtruth_table:
            groundtruth_table[(S,D)] = [[token, traj, label, pathnum]]
        else:
            groundtruth_table[(S,D)].append([token, traj, label, pathnum])
    return groundtruth_table

def draw_block(name, draw_normal, best_item, control=1):
    plt.figure(figsize=(10.5/2,6.8/2))
    c = 0
    for items in draw_normal:
        #if c == control:
        #    break
        [token, traj, label, pathnum] = items
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,1], color="blue", linewidth = 4, label='normal route', alpha=0.4)
        c += 1
    plt.text(traj[0][0],traj[0][1], "S", ha="center", va="center", size=11,
    bbox=dict(boxstyle="square, pad=0.25", fc="w", lw=2, alpha=0.8))
    plt.text(traj[-1][0],traj[-1][1], "D", ha="center", va="center", size=11,
    bbox=dict(boxstyle="square, pad=0.25", fc="w", lw=2, alpha=0.8))
    
    [token, traj, label, pathnum] = best_item
    traj = np.array(traj)
    plt.plot(traj[:,0],traj[:,1], color="red", marker='o', linewidth = 2, linestyle='dashed', label='detour route', alpha=0.6)    
    detour = []
    flag = True
    for l in range(1, len(traj)):
        if label[l-1] != label[l]:
            detour.append(l-1)
        if len(detour) == 2: 
            #print(detour)
            s, d = detour[0], detour[1]+1
            if flag:
                plt.plot(traj[s:d+1,0],traj[s:d+1,1], color="green", linewidth = 9,  label='labeled detour', alpha=0.3)
                flag = False
            else:
                plt.plot(traj[s:d+1,0],traj[s:d+1,1], color="green", linewidth = 9,  alpha=0.3)
            detour = []
    plt.title('SD-Pair: ({}, {})'.format(S, D))
    #plt.axis('off')
    #plt.xticks([])
    #plt.yticks([])
    plt.legend(loc='best', prop = {'size': 12})
    plt.savefig(name, format='pdf', dpi=1000)
    #plt.savefig(name, format='png')

def draw(name, trajs):
    draw_normal, best, Best_Item = [], 0, []
    for items in trajs:
        if len(items) == 4:
            [token, traj, label, pathnum] = items
        if len(items) == 5:
            [token, traj, label_, pathnum, label] = items
        tmp = sum(label) 
        if tmp == 0 and pathnum > 10:
            if len(items) == 4:
                draw_normal.append(items)
            if len(items) == 5:
                draw_normal.append([token, traj, label, pathnum])
        elif tmp > 0:
            if len(items) == 4:
                Best_Item.append(items)
            if len(items) == 5:
                Best_Item.append([token, traj, label, pathnum])
    
    for i in range(len(Best_Item)):
        #print(label)
        draw_block(name+'-'+str(i)+'.pdf', draw_normal, Best_Item[i])


city = 'chengdu'
historical_average_coordinate = load_data("./data_roadnetwork/"+str(city)+"/historical_average_coordinate")
groundtruth_path = './'+str(city)+'_detour/'
traj_path = './data_roadnetwork/'+str(city)+'/'
vis_path = traj_path+'vis/'

groundtruth_table = obtain_groundtruth()
pickle.dump(groundtruth_table, open(traj_path+'case_study', 'wb'), protocol=2)

(S,D) = (209,2027)

draw(vis_path+'ground-'+str(S)+'-'+str(D), groundtruth_table[(S,D)])

#To test other methods, just to replace "label" in [token, traj, label, pathnum] with your predicted labels

    
    
    
    
    
    
    
     