# -*- coding: utf-8 -*-


import pickle

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file


val_data = load_data("./data_roadnetwork/chengdu/val_data.pickle")
val_fea = load_data("./data_roadnetwork/chengdu/val_fea_0.4.pickle")
val_dir = load_data("./data_roadnetwork/chengdu/val_direction.pickle")
val_label = load_data("./data_roadnetwork/chengdu/val_label.pickle")




for data,fea,dirt, label in zip(val_data,val_fea,val_dir,val_label):
    if len(data)<15:
        g1_data.append(data)
        g1_fea.append(fea)
        g1_dir.append(dirt)
        g1_label.append(label)
    elif len(data)>=15 and len(data)<30:
        g2_data.append(data)
        g2_fea.append(fea)
        g2_dir.append(dirt)
        g2_label.append(label)
    elif len(data)>=30 and len(data)<45:
        g3_data.append(data)
        g3_fea.append(fea)
        g3_dir.append(dirt)
        g3_label.append(label)
    elif len(data)>=45 :
        g4_data.append(data)
        g4_fea.append(fea)
        g4_dir.append(dirt)
        g4_label.append(label)
#    elif len(data)>=60:
#        g5_data.append(data)
#        g5_fea.append(fea)
#        g5_dir.append(dirt)
#        g5_label.append(label)

print("sum_data:", len(g1_data)+len(g2_data)+len(g3_data)+len(g4_data))


file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g1_data.pickle","wb")
pickle.dump(g1_data,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g2_data.pickle","wb")
pickle.dump(g2_data,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g3_data.pickle","wb")
pickle.dump(g3_data,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g4_data.pickle","wb")
pickle.dump(g4_data,file) #storing_list
file.close()
#file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g5_data.pickle","wb")
#pickle.dump(g5_data,file) #storing_list
#file.close()


file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g1_fea.pickle","wb")
pickle.dump(g1_fea,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g2_fea.pickle","wb")
pickle.dump(g2_fea,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g3_fea.pickle","wb")
pickle.dump(g3_fea,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g4_fea.pickle","wb")
pickle.dump(g4_fea,file) #storing_list
file.close()
#file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g5_fea.pickle","wb")
#pickle.dump(g5_fea,file) #storing_list
#file.close()


file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g1_dir.pickle","wb")
pickle.dump(g1_dir,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g2_dir.pickle","wb")
pickle.dump(g2_dir,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g3_dir.pickle","wb")
pickle.dump(g3_dir,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g4_dir.pickle","wb")
pickle.dump(g4_dir,file) #storing_list
file.close()
#file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g5_dir.pickle","wb")
#pickle.dump(g5_dir,file) #storing_list
#file.close()


file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g1_label.pickle","wb")
pickle.dump(g1_label,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g2_label.pickle","wb")
pickle.dump(g2_label,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g3_label.pickle","wb")
pickle.dump(g3_label,file) #storing_list
file.close()
file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g4_label.pickle","wb")
pickle.dump(g4_label,file) #storing_list
file.close()
#file=open(r"./data_roadnetwork/chengdu/efficiency_group_data/g5_label.pickle","wb")
#pickle.dump(g5_label,file) #storing_list
#file.close()



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        