# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from itertools import groupby
from dataset import End2EndDataset,End2EndDataset_test
from utils.torch_util import calc_f1
from utils.path_util import from_project_root
from datetime import datetime
import dataset
import pickle

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
#    print("file_1:", file_1)
    data_load_file = pickle.load(file_1)
    return data_load_file

def evaluate_e2e(model, val_data, val_dir, val_label, val_feature, bsl_model=None):
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
    flag =  True
    val_data = load_data(val_data)
    val_dir = load_data(val_dir)
    val_label = load_data(val_label)
    val_feature = load_data(val_feature)
    data_sum = 0
    fake_in = 0
    eval_data = End2EndDataset(val_data, val_dir, val_label, val_feature , flag, next(model.parameters()).device, evaluating=True)
#    print("eval_data:", eval_data)

    loader = DataLoader(eval_data, batch_size=1, collate_fn=eval_data.collate_func)
    ret = {'precision': 0, 'recall': 0, 'f1': 0}
        
    # switch to eval mode
    model.eval()
    start_time =time.time()
    
    region_true_list, region_pred_list = list(), list()
    with torch.no_grad():
        sum_glod, sum_pre,sum_tp = 0,0,0
        sum_pre_, sum_tp_ = 0,0
        list1,list2 = [],[]
        final_f1 = []
        grod_sum = []
        pre_sum = [] 
        for sentence, setence_lengths, dirt, fe, sentence_labels,tags_lists, records_list in loader:
#            , tag_list, records
#            print("sentence:", sentence)
#            print("setence_lengths:",setence_lengths)
#            print("dirt:", dirt)
#            print("fe:", fe)
#            print("tags_lists:",tags_lists)
            data_sum+=1
            ind = 0
            if bsl_model:
                pred_sentence_labels = torch.argmax(bsl_model.forward(*sentence), dim=1)
                pred_region_output, _ = model.forward(*sentence, pred_sentence_labels)
            else:
                try:
                    tags_lists =[]
                    records_list = []
                    probs_, frag_all,region_outputs, region_labels_expand, region_flag, sentence_outputs = model.forward(sentence, setence_lengths, dirt,  tags_lists, fe, sentence_labels, records_list,0, mode='test')
#                    print("region_outputs:", region_outputs)
#                    print("region_labels_expand:",region_labels_expand)
#                    print("frag_all:", frag_all)
#                    print("tags_lists:", tags_lists)
#                    print("fe:", fe)
#                    print("sentence_labels:",sentence_labels)
#                    print(type(frag_all))
#                    print("sentence_labels:", sentence_labels)
#                    print(type(sentence_labels))
#                    print("setence_lengths:", setence_lengths)
#                    print(type(setence_lengths))
                    sentence_labels_list = sentence_labels.tolist()
#                    sum_f1 = []
                    import numpy as np
                    for sub_frag, grod,st in zip(frag_all, sentence_labels_list, setence_lengths):
                        if sub_frag ==[]:
                            fake_in+=1
                            pre_sum.append([0]*len(grod[:st-1]))
                            grod_sum.append(grod[:st-1])
                        if sub_frag!=[]:

                            ab_idex = [item for item in range(sub_frag[0][0], sub_frag[0][1]+1)]
                            pre = [0]*(st-1)
                            for i in range(st-1):
                                if i in ab_idex:
                                    pre[i]=1
#                            print("pre:", pre)
#                            print("grod:", grod)
                            
                            pre_sum.append(pre)
                            grod_sum.append(grod[:st])
                            

                except RuntimeError:
                    print("all 0 tags, no evaluating this epoch")
                    continue
    
    end_time =time.time()
    period_time = end_time - start_time
    
#    print("len_data:", data_sum)
#    print("fake_data:", fake_in)
#    print("every time:", period_time/data_sum)
#    print("pre_sum:", pre_sum[:3])
    
    f1_final = my_fscore_whole_determine(grod_sum, pre_sum, flag='soft')
   
#    print("final_f1:", np.mean(final_f1))
    return f1_final

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
    test = "./data_roadnetwork/chengdu/test_data.pickle"
    test_dir = "./data_roadnetwork/chengdu/test_direction.pickle"
    test_label = "./data_roadnetwork/chengdu/test_label.pickle"
    test_feature = "./data_roadnetwork/chengdu/test_fea_0.4.pickle"
    
    #best_model_url = "./models/end2end_model_epoch2_0.891295_0.200000_2.500000.pt"
    best_model_url = "./models/your_trained_model.pt"# such as end2end_model_epoch2_0.891295_0.200000_2.500000.pt        
    best_model = torch.load(best_model_url)
    
    f1_test = evaluate_e2e(best_model, test, test_dir, test_label, test_feature)
    print("test_f1:", f1_test)
            
    
