# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from itertools import groupby
from classifier_model import LSTMClassifier
from dataset import End2EndDataset, End2EndDataset_classifier,End2EndDataset_test
from utils.torch_util import calc_f1
from utils.path_util import from_project_root
from datetime import datetime
import numpy as np
import dataset
import heapq
import pickle
import os
from more_itertools import chunked

con_k = 2
val_unsim = "./data_roadnetwork/chengdu/test_data.pickle"
val_unsim_label = "./data_roadnetwork/chengdu/test_label.pickle"
val_sim_index = "./data_roadnetwork/chengdu/simplification/simp_index_test"
anomaly_val_pre = "./data_roadnetwork/chengdu/simplification/simp_data_test"

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file


dict_pre = {}
val_unsim = load_data(val_unsim)
val_unsim_label = load_data(val_unsim_label)
val_sim_index = load_data(val_sim_index)
anomaly_val_pre = load_data(anomaly_val_pre)


for i in range(len(anomaly_val_pre[:])):
    dict_pre[tuple(anomaly_val_pre[i])] = [val_unsim[i], val_unsim_label[i],val_sim_index[i]]
#    print("dict_pre:", dict_pre)
#    println()

overall_point =0
for item in val_unsim[:]:
    overall_point+=len(item)
print("overall_point:", overall_point)

#k=2


import time

def evaluate_e2e(model, val_data, val_dir, val_label, val_feature, BATCH_SIZE_Classfier, BATCH_SIZE, grop_num, bsl_model=None):
    """ evaluating end2end model on dataurl

    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
        bsl_model: trained binary sequence labeling model

    Returns:
        ret: dict of precision, recall, and f1

    """
    
#    print("\nevaluating model on:", data_url, "\n")
    import time
    start_time_c = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1  = LSTMClassifier(hidden_dim=128, n_embeddings_position = 6113,n_embeddings_direction = 182, embedding_dim=300,label_size=2, batch_size=128, use_gpu=True)
    model_1 = model_1.to(device)
    model_1.load_state_dict(torch.load('./models/classifier_simp.model'))
###    
    eval_data_1 = End2EndDataset_classifier(val_data, val_dir, val_label, val_feature , next(model_1.parameters()).device, evaluating=True)
    loader_1 = DataLoader(eval_data_1, batch_size=BATCH_SIZE_Classfier, collate_fn=eval_data_1.collate_func)
#    
    final_label = list()
    param = 1-0.8
    for sentence, setence_lengths, dirt, fe, sentence_labels in loader_1: 
        tags_lists =[]                                                                                                     
        pred = model_1(sentence, setence_lengths, dirt,  tags_lists, fe, records_list= None, epoch=0, mode='test')
        pred = pred.view(pred.shape[0]).float()
        pred_idx = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
#        pred_idx_1 = torch.where(pred<=0.5,torch.ones_like(pred),torch.zeros_like(pred))
        final_label += pred_idx.tolist()

    import random        
    from random import choice
    
    max_number = heapq.nlargest(int(len(final_label)*param), final_label)
    
#    max_number = heapq.nlargest(int(len(final_label)*param), final_label)
#    final_label = [item for item in range(50000)]
#    max_number = random.sample(final_label, int(len(final_label)*param))
    print("final_label:", len(final_label))
    print("len(max_num):", len(max_number))
    max_index = []
    for t in max_number:
        index = final_label.index(t)
        max_index.append(index)
        final_label[index] = -1
#        
    val_data = load_data(val_data)
    val_dir = load_data(val_dir)
    val_label = load_data(val_label)
    val_feature = load_data(val_feature)
    #checking the number of point
    
    ano_num =0
    for item in val_feature:
        if sum(item)>0:
            ano_num+=1
    print("ano_num:", ano_num, len(val_data))
    
    valid_data = []
    valid_dir = []
    valid_label = []
    valid_feature = []
#    valid_unsim = []
#    valid_unsim_label = []    
    for idx in max_index:
        valid_data.append(val_data[idx])
        valid_dir.append(val_dir[idx])
        valid_label.append(val_label[idx])
        valid_feature.append(val_feature[idx])
    sim_s = 0
    print("calsisifer have:", len(valid_data))
    for la in valid_label:
        if sum(la)>0:
            sim_s+=1
    print("sim_s:", sim_s)
#        
#        
##    println()
    overall_index = []
#    for i in range(len(val_data)):
#        overall_index.append(i)
    diffe_set = [i for i in overall_index if i not in max_index]
##    print("diffe_set:", len(diffe_set))
#    
#    diffe_set = overall_index.removeAll(max_index)
#    valid_undiff = []
#    for set_r in diffe_set:
#        valid_undiff.append(val_data[set_r])
    valid_undiff = [val_data[item] for item in diffe_set]
##    print("len(valid_undiff:)", len(valid_undiff))
##    print("len(valid_undiff:)", len(valid_data))
##    print("val:",len(val_data))
##    println()
    unsim_label_unpre = [dict_pre[tuple(ii)][1] for ii in valid_undiff]
##    print("unsim_label_unpre:",unsim_label_unpre[0])
    unsim_pre =  [[0]*len(yy) for yy in unsim_label_unpre]   




    flag  = True
    ##not adding classifier
    eval_data = End2EndDataset(valid_data, valid_dir, valid_label, valid_feature , flag, next(model.parameters()).device)
    loader = DataLoader(eval_data, batch_size=BATCH_SIZE, collate_fn=eval_data.collate_func)
    ##adding classifier
    sum_data = 0
#    eval_data = End2EndDataset(val_data[:20000], val_dir[:20000], val_label[:20000], val_feature[:20000] , flag, next(model.parameters()).device)
#    loader = DataLoader(eval_data, batch_size=BATCH_SIZE, collate_fn=eval_data.collate_func)
#    
    
    ret = {'precision': 0, 'recall': 0, 'f1': 0}
        
    # switch to eval mode
    model.eval()
    
    print("-----------------calculating time-----------------")
    
    
    region_true_list, region_pred_list = list(), list()
    with torch.no_grad():
#        sum_glod, sum_pre,sum_tp = 0,0,0
#        sum_pre_, sum_tp_ = 0,0
#        list1,list2 = [],[]
#        final_f1 = []
        ground_t = []
        pred =[]
        pred_unsim =[]
        ground_unsim = []
        grod_wrog =[]
        pred_wrog = []
        avg_length = 0
        avg_l_un = 0
        s_a = 0
        
        for sentence, setence_lengths, dirt, fe, sentence_labels, tags_lists, records_list in loader:
            sum_data+=1

            ind = 0
            if bsl_model:
                pred_sentence_labels = torch.argmax(bsl_model.forward(*sentence), dim=1)
                pred_region_output, _ = model.forward(*sentence, pred_sentence_labels)
            else:
                try:  
#                    tags_lists= []
#                    records_list =[]
                    probs_, frag_all,region_outputs, region_labels_expand, region_flag, sentence_outputs = model.forward(sentence, setence_lengths, dirt,  tags_lists, fe, sentence_labels, records_list,0, mode='test')
                   
                    sentence_labels_list = sentence_labels.tolist()
                    sentence_scala = sentence.tolist()

                    
                    import numpy as np
                    print("----------------------before entering frag----------------------")
                    

                    for sub_frag, grod,st in zip(frag_all, sentence_labels_list, setence_lengths):
                        if sub_frag ==[]:
                            grod_wrog.append([0]*len(grod[:st]))
                            grod_wrog.append(grod[:st])

                        if sub_frag!=[]:
                            s_a+=1

                            avg_length+=(st-1)
                            
                            ab_idex = [item for item in range(sub_frag[0][0], sub_frag[0][1]+1)]
                            
#                            print("grod_unsim:", len(grod_unsim))
                            
                            pre = [0]*(st)
                            for i in range(st):
                                if i in ab_idex:
                                    pre[i]=1
                            pred_unsim.append(pre)
                            ground_unsim.append(grod[:st])

                except RuntimeError:
                    continue

        combine_grod = ground_unsim + unsim_label_unpre+grod_wrog
        combine_pre = pred_unsim + unsim_pre+pred_wrog
        print(len(combine_pre))
        print("sum_point:", sum_data)
        end_time_c = time.time()
        period_time = end_time_c-start_time_c
        print("----------------------Average Time {}--------------------------".format(period_time))
        
        f1_final = my_fscore_whole_determine(ground_t, pred, flag='soft')
        f1_final_unsim = my_fscore_whole_determine(combine_grod, combine_pre, flag='soft')
        return f1_final,f1_final_unsim



def P_R_F1(sum_pre,sum_glod,sum_tp):
    precision, recall, F1 = 0.0, 0.0, 0.0
    if sum_pre != 0:
        precision = sum_tp * 1.0 / sum_pre
    if sum_glod != 0:
        recall = sum_tp * 1.0 / sum_glod
    if (precision + recall) != 0.0:
        F1 = 2 * precision * recall * 1.0 / (precision + recall)
    return precision, recall, F1
def my_fscore_unify(Y_true, Y_score, flag='soft'):
    thres = set(sum(Y_score, []))
    res = []
    #print('thres', thres)
    for the in thres:
        fscore = []
        for y_true, y_score in zip(Y_true, Y_score):
            tmp =  [1 if s >= the else 0 for s in y_score]
            #print(y_true, tmp)
            fscore.append(my_fscore_determine(y_true, tmp, flag))
        res.append(sum(fscore)/len(fscore))
    return max(res)
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
        #print('correct_num, pred_num, ground_num', listA, listB, correct_num, pred_num, ground_num)
        F1 = 2*((precision*recall)/(precision+recall))
        #print(label_test, label_pred, F1)
        return F1
    else:
        return 0.0

if __name__ == '__main__':
    con_k = 2
    BATCH_SIZE_Classfier, BATCH_SIZE, grop_num = 800, 16, 2
    
    test = "./data_roadnetwork/chengdu/simplification/simp_data_test"
    test_dir = "./data_roadnetwork/chengdu/simplification/simp_dir_test"
    test_label = "./data_roadnetwork/chengdu/simplification/simp_label_test"
    test_feature = "./data_roadnetwork/chengdu/simplification/simp_fe_test"
    
    #val_unsim = "./data/processed_porto_anomaly_val.pickle"
    #val_unsim_label = "./data/processed_val_label.pickle"
    #val_sim_index = "./data/processed_val_{}_index.pickle".format(con_k)
    #anomaly_val_pre = "./data/processed_porto_anomaly_val_{}.pickle".format(con_k)
    
    #best_model_url = "./models/end2end_model_epoch2_0.891295_0.200000_2.500000.pt"
    best_model_url = "./models/your_trained_model_simp.pt"# such as end2end_model_epoch2_0.891295_0.200000_2.500000.pt        
    best_model = torch.load(best_model_url)
    
    f1_test = evaluate_e2e(best_model, test, test_dir, test_label, test_feature, BATCH_SIZE_Classfier, BATCH_SIZE, grop_num)
    print("test_f1:", f1_test)
    #k=2