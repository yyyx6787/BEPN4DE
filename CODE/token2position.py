# -*- coding: utf-8 -*-

import pickle
import math
#from math import sqrt
#from math import acos
import numpy as np

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

def get_traj_with_coor(route):
    if route[0] not in historical_average_coordinate:
        return []
    coorT = []
    for r in route:
        if r not in historical_average_coordinate:
            coorT.append(t_)
        else:
            t_ = historical_average_coordinate[r]
            coorT.append(t_)
    return coorT

def angle_of_vector(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def getting_direction(token_dict,anomaly_train,para,alpha,de_times,anomaly_label):
    s = 0
    anomaly_train_position=[]
    for sub_traj,sub_label in zip(anomaly_train, anomaly_label):

        sub_list = get_traj_with_coor(sub_traj)
        
        if sub_list!=[]:
            s +=1
            anomaly_train_position.append(sub_list)

    anomaly_train_angle=[]
    for sub_traj in anomaly_train_position:
        sub_angle=[]
        for i in range(1, len(sub_traj)-1):

            v1 = sub_traj[i-1]+ sub_traj[i]
            v2 = sub_traj[i]+ sub_traj[i+1]
            angle = angle_of_vector(v1, v2)

            sub_angle.append(angle)
        
        anomaly_train_angle.append(sub_angle)

    file=open(r"./data_roadnetwork/"+str(city)+"/train_direction_{}_{}_n.pickle".format(0.4,2.5),"wb")
    pickle.dump(anomaly_train_angle,file)   #storing_list
    file.close()

if __name__ == '__main__':
    para = "train"
    alpha = 0.4
    de_times = 2.5
    city = 'chengdu'
    
    historical_average_coordinate = load_data("./data_roadnetwork/"+str(city)+"/historical_average_coordinate")

    anomaly_train = load_data(r"./data_roadnetwork/"+str(city)+"/train_data_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1)))
    anomaly_label = load_data(r"./data_roadnetwork/"+str(city)+"/train_label_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1)))

    getting_direction(historical_average_coordinate,anomaly_train,para,np.round(alpha,1),np.round(de_times,1),anomaly_label)    
            
    







      