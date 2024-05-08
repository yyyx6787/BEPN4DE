# -*- coding: utf-8 -*-

import os,random
import torch
#import joblib
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.torch_util import get_device
from collections import defaultdict
#from gensim.models import KeyedVectors
import torch.nn as nn
#import utils.json_util as ju
#from utils.path_util import from_project_root, dirname
import pickle 


LABEL_IDS = {"O": 0, "Anomaly": 1}

LABEL_LIST = ["O","Anomaly"]

TAG_NUM = 4

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

class End2EndDataset(Dataset):
    def __init__(self, t_data, t_dir, t_label, t_feature, flag, device, evaluating=False):
        super().__init__()
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
        self.anomaly_train, self.anomaly_train_dir, self.train_feature, self.train_label, self.tag_list, self.records = load_raw_data(t_data, t_dir, t_label, t_feature, flag)
        self.device = device
        self.evaluating = evaluating

    def __getitem__(self, index):
        return self.anomaly_train[index], self.anomaly_train_dir[index], self.train_feature[index], self.train_label[index], self.tag_list[index], self.records[index]

    def __len__(self):
        return len(self.anomaly_train)

    def collate_func(self, data_list):

        data_list = sorted(data_list, key=lambda x: len(x[0]), reverse=True)
#        for item in data_list:
#            print("len_item:", len(item[0]))
#        
#        sentence_list, tags_list,sent_reverse_list,tag_boundary_reverse_list,records_list,alphaFeatures_list = zip(*data_list)
        traj_list, direction_list, feature_list, label_list, tag_list, records = zip(*data_list)

        # unzip
#        sentence_tensors = gen_sentence_tensors(sentence_list,self.device)
        traj_lengths = [len(traj) for traj in traj_list]

        traj_tensors = []
        dir_tensors = []
        fe_tensors=[]
#        label_ =[]
        for item in traj_list:
#            print("sub_traj:", item)
            traj_tensors.append(torch.LongTensor(item))
#        traj_tensors = torch.LongTensor(traj_list)
        for sub_item in direction_list:
            sub_item.insert(0, 181)
            sub_item.append(181)
            sub_item.append(181)
            dir_tensors.append(torch.LongTensor(sub_item))
        for sub_ in feature_list:
#            sub_.insert(0, 0)
            sub_.append(0)
#            sub_item.append(181)
            fe_tensors.append(torch.LongTensor(sub_))

        traj_pad = pad_sequence(traj_tensors, batch_first=True).to(self.device)
#        print("traj_pad:",traj_pad.size())        
        dir_pad =  pad_sequence(dir_tensors, batch_first=True).to(self.device)
        fe_pad =  pad_sequence(fe_tensors, batch_first=True).to(self.device)
#        print(dir_pad.size())
        
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices,alphas)
#        max_sent_len = traj_pad.size()[1]
        max_sent_len = max(traj_lengths)
      
        traj_labels = [sub_traj_label+([0]*(max_sent_len-len(sub_traj_label))) for sub_traj_label in label_list]

        labels_pad = torch.LongTensor(traj_labels).to(self.device)

        if self.evaluating:
#            return sentence_tensors, tags_list,sent_reverslabels_pade_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
            return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad, tag_list, records
#        return sentence_tensors, tags_list, sent_reverse_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
        return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad, tag_list, records

def lr_decay(optimizer, epoch, init_lr,decay_rate):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def load_raw_data(t_data, t_dir, t_label,t_feature,flag):
    print("----data loading----")
#    t_data_1 = load_data(t_data)
#    t_dir_1 = load_data(t_dir)
#    t_label_1 = load_data(t_label)
#    t_feature_1 = load_data(t_feature)
##    t_data = load_data(t_data)
##    t_dir = load_data(t_dir)
##    t_label = load_data(t_label)
##    t_fea = load_data(t_feature)
#    t_data =t_data_1[:]
#    t_dir =t_dir_1[:]
#    t_label =t_label_1[:]
#    t_fea =t_feature_1[:]
    if flag == True:
        t_data =t_data[:]
        t_dir =t_dir[:]
        t_label =t_label[:]
        t_fea =t_feature[:]
    else:
        print("----data loading----")
        t_data_1 = load_data(t_data)
        t_dir_1 = load_data(t_dir)
        t_label_1 = load_data(t_label)
        t_feature_1 = load_data(t_feature)
#        t_data = load_data(t_data)
#        t_dir = load_data(t_dir)
#        t_label = load_data(t_label)
#        t_fea = load_data(t_feature)
        t_data =t_data_1[:]
        t_dir =t_dir_1[:]
        t_label =t_label_1[:]
        t_fea =t_feature_1[:]
#    for data, dirt, label in zip(t_data_1, t_dir_1,t_label_1):
#        if len(data) ==len(dirt)+2 and len(dirt)+2==len(label):
#            pass
#        else:
#            print("%%%%%")
#            print(len(data))
#            print(len(dirt))
#            print(len(label))
#    
    
    t_data_= []
    tag_list = []
    t_label_ =[]
    t_dir_ =  []
    t_fea_ = []
    records = []
    all_data_num = 0
    for i in range(len(t_data)):
        item_pos = []
        item_dir = []
        tag_boundary = []
#        print(item)
        item_pos.extend(t_data[i])
#        item_pos.append('</s>')
        item_pos.append(6112)
        item_dir.extend(t_label[i])
        item_dir.append(0)
#        t_data_.append(item_pos)
#        t_label_.append(item_dir)
#        item_dir.append(t_label[i])
        if sum(item_dir)>0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
            for k, item in enumerate(item_dir[:-1]):
#                if k== len(item_dir)-1:
#                     tag_boundary.append(k)
                if item_dir[k]==0:
                    tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
                elif item_dir[k]==1:
                    sub_s = []
                    dict1={}
                    j= k-1
#                    print("j:", j)
                    existing_anomaly = [idx for idx, item in enumerate(item_dir[:]) if item==1]
#                    tag_boundary.extend(existing_anomaly)
                    sub_anomaly  = [idx for idx, item in enumerate(item_dir[k:]) if item==1][::-1] 
#                    sub_s.extend(sub_anomaly)
#                    print("sub_s:", sub_s)
#                    
                    ## [{(0,2):PER, (4,5):LOC}]
                    dict1[(existing_anomaly[0], existing_anomaly[-1])] = "Anomaly"
                    record.append(dict1)
                    records.append(record)
                    last_chance = [idx for idx, item in enumerate(item_dir[k:]) if item==1][-1]
#                    print("last_chance:", last_chance)
#                    tag_boundary.append(last_chance)
                    tag_boundary.extend(sub_anomaly)
#                    print("j+last_chance+1:", j+last_chance+1+1)
                    for kk, sub_item in enumerate(item_dir[j+last_chance+2:-1]):
#                        if kk == len(item_dir[j+last_chance+2+kk:])-1:
#                            print("****")
#                            print("final_end:", kk)
#                            tag_boundary.append(kk)
#                            break;
                        if sub_item == 0:
                            tag_boundary.append(len(item_dir[j+last_chance+2+kk:])-1)
                    tag_list.append(tag_boundary)
#                    if len(item_dir)-1 !=len(tag_boundary):
#                        print("item_dir:", item_dir)
#                        print("tag_boundary:", tag_boundary)
#                        
#                    print("******************************")
#                    print("item_dir:", item_dir)
#                    print("existing_anomaly:",existing_anomaly[0], existing_anomaly[-1])
#                    print("records:", records)
#                    if len(item_dir)==63:
#                        print("tag_boundary:", tag_boundary)
#                        print("item_dir:", len(item_dir))
##                    print("len_tag_boundary:", len(tag_boundary))
                    break;
                    
#        elif sum(item_dir) ==0:
#            record = []
#            t_data_.append(item_pos)
#            t_label_.append(item_dir)
#            t_dir_.append(t_dir[i])
#            t_fea_.append(t_fea[i])
#            for k, item in enumerate(item_dir[:-1]):
#                tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
#            tag_list.append(tag_boundary)
#            records.append(record)
#            print("-------------------------------------")
#            print("item_dir:", item_dir)
#            print("tag_boundary:", tag_boundary)
#            print("len_item:", len(item_dir))
#            if len(item_dir)-1 !=len(tag_boundary):
#                print("item_dir:", item_dir)
#                print("tag_boundary:", tag_boundary)
#                
                
#            print("len_tag_boundary:", len(tag_boundary))
#            print("record:", record)
    return t_data_, t_dir_, t_fea_, t_label_, tag_list, records

class End2EndDataset_test(Dataset):
    def __init__(self, t_data, t_dir, t_label, t_feature, device, evaluating=False):
        super().__init__()
#        self.data_url = data_url
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
#        , self.tag_list, self.records
#        self.sentences, self.tags,self.sent_reverse,self.tag_boundary_reverse,self.records,self.alphaFeatures = load_raw_data(data_url)
        self.anomaly_train, self.anomaly_train_dir, self.train_feature, self.train_label = load_raw_data_test(t_data, t_dir, t_label, t_feature)
#        self.anomaly_train_dir = load_data(t_dir)
#        self.train_label = load_data(t_label)
        
#        self.anomaly_val = load_data("./data/processed_porto_anomaly_val.pickle")
#        self.anomaly_val_dir = load_data("./data/val_anomaly_direction.pickle")
#        self.val_label = load_data("./data/processed_val_label.pickle")

        self.device = device
        self.evaluating = evaluating

    def __getitem__(self, index):
# =============================================================================
# #        , self.tag_list[index], self.records[index]
# =============================================================================
        return self.anomaly_train[index], self.anomaly_train_dir[index], self.train_feature[index], self.train_label[index]

    def __len__(self):
        return len(self.anomaly_train)

    def collate_func(self, data_list):
#        print("pay attention to bug")
#        print("data_list:", data_list)
#        print("ori:",data_list)
        data_list = sorted(data_list, key=lambda x: len(x[0]), reverse=True)
#        for item in data_list:
#            print("len_item:", len(item[0]))
#        
#        sentence_list, tags_list,sent_reverse_list,tag_boundary_reverse_list,records_list,alphaFeatures_list = zip(*data_list)
        traj_list, direction_list, feature_list, label_list = zip(*data_list)
#       sentence_tensors = gen_sentence_tensors(sentence_list,self.device)
        traj_lengths = [len(traj) for traj in traj_list]

        traj_tensors = []
        dir_tensors = []
        fe_tensors=[]
#        label_ =[]
        for item in traj_list:
#            print("sub_traj:", item)
            traj_tensors.append(torch.LongTensor(item))
#        traj_tensors = torch.LongTensor(traj_list)
        for sub_item in direction_list:
            sub_item.insert(0, 181)
            sub_item.append(181)
            sub_item.append(181)
            dir_tensors.append(torch.LongTensor(sub_item))
        for sub_ in feature_list:
#            sub_.insert(0, 0)
            sub_.append(0)
#            sub_item.append(181)
            fe_tensors.append(torch.LongTensor(sub_))

        traj_pad = pad_sequence(traj_tensors, batch_first=True).to(self.device)
#        print("traj_pad:",traj_pad.size())
#        
        dir_pad =  pad_sequence(dir_tensors, batch_first=True).to(self.device)
        fe_pad =  pad_sequence(fe_tensors, batch_first=True).to(self.device)

        max_sent_len = max(traj_lengths)
      
        traj_labels = [sub_traj_label+([0]*(max_sent_len-len(sub_traj_label))) for sub_traj_label in label_list]

        labels_pad = torch.LongTensor(traj_labels).to(self.device)

        if self.evaluating:
#            return sentence_tensors, tags_list,sent_reverslabels_pade_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
            return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad
#        return sentence_tensors, tags_list, sent_reverse_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
        return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad



def load_raw_data_test(t_data, t_dir, t_label,t_feature):
    print("----data loading----")
    t_data_1 = load_data(t_data)
    t_dir_1 = load_data(t_dir)
    t_label_1 = load_data(t_label)
    t_feature_1 = load_data(t_feature)
#    t_data = load_data(t_data)
#    t_dir = load_data(t_dir)
#    t_label = load_data(t_label)
#    t_fea = load_data(t_feature)
    t_data =t_data_1[:]
    t_dir =t_dir_1[:]
    t_label =t_label_1[:]
    t_fea =t_feature_1[:]

    
    t_data_= []
    tag_list = []
    t_label_ =[]
    t_dir_ =  []
    t_fea_ = []
    records = []
    all_data_num = 0
    for i in range(len(t_data)):
        item_pos = []
        item_dir = []
        tag_boundary = []
#        print(item)
        item_pos.extend(t_data[i])
#        item_pos.append('</s>')
        item_pos.append(6112)
        item_dir.extend(t_label[i])
        item_dir.append(0)
#        t_data_.append(item_pos)
#        t_label_.append(item_dir)
#        item_dir.append(t_label[i])
        if sum(t_fea[i])>0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])


                    
        elif sum(item_dir) ==0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
            for k, item in enumerate(item_dir[:-1]):
                tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
            tag_list.append(tag_boundary)
            records.append(record)

    return t_data_, t_dir_, t_fea_, t_label_

def compuLoss_entity(probs_, labels_):
    loss_ = 0
    entroy = nn.CrossEntropyLoss()
    #entroy = DiceLoss()
    num = 0
    for i in range(len(labels_)):
#        print("loop_i:", i)
        sents_loss_ = 0
        if len(probs_[i]) != 0:
            num += 1
            for k in range(len(probs_[i])):
#                sents_loss_ = sents_loss_ + entroy(probs_[i][k].view(1,-1).cuda(), torch.LongTensor([labels_[i][k]]).cuda())
                sents_loss_ = sents_loss_ + entroy(probs_[i][k].view(1,-1).cpu(), torch.LongTensor([labels_[i][k]]).cpu())
            sents_loss_ = sents_loss_*1.0 / len(probs_[i])
        loss_ = loss_ + sents_loss_
    if num != 0:
        loss_ = loss_*1.0 / num
    return loss_

class End2EndDataset_classifier(Dataset):
    def __init__(self, t_data, t_dir, t_label, t_feature, device, evaluating=False):
        super().__init__()
#        self.data_url = data_url
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
        
#        self.sentences, self.tags,self.sent_reverse,self.tag_boundary_reverse,self.records,self.alphaFeatures = load_raw_data(data_url)
        self.anomaly_train, self.anomaly_train_dir, self.train_feature, self.train_label = load_raw_data_1(t_data, t_dir, t_label, t_feature)
#        self.anomaly_train_dir = load_data(t_dir)
#        , self.tag_list, self.records
#        self.train_label = load_data(t_label)
        
#        self.anomaly_val = load_data("./data/processed_porto_anomaly_val.pickle")
#        self.anomaly_val_dir = load_data("./data/val_anomaly_direction.pickle")
#        self.val_label = load_data("./data/processed_val_label.pickle")
        
        self.device = device
        self.evaluating = evaluating

    def __getitem__(self, index):
#        , self.tag_list[index], self.records[index]
        return self.anomaly_train[index], self.anomaly_train_dir[index], self.train_feature[index], self.train_label[index]

    def __len__(self):
        return len(self.anomaly_train)

    def collate_func(self, data_list):
#        print("pay attention to bug")
#        print("data_list:", data_list)
#        print("----------------classifier----------------")
        data_list = sorted(data_list, key=lambda x: len(x[0]), reverse=True)
#        for item in data_list:
#            print("len_item:", len(item[0]))
#        
#        sentence_list, tags_list,sent_reverse_list,tag_boundary_reverse_list,records_list,alphaFeatures_list = zip(*data_list)
        traj_list, direction_list, feature_list, label_list = zip(*data_list)
#        sentence_tensors = gen_sentence_tensors(sentence_list,self.device)
        traj_lengths = [len(traj) for traj in traj_list]

        traj_tensors = []
        dir_tensors = []
        fe_tensors=[]
#        label_ =[]
        for item in traj_list:
#            print("sub_traj:", item)
            traj_tensors.append(torch.LongTensor(item))
#        traj_tensors = torch.LongTensor(traj_list)
        for sub_item in direction_list:
            sub_item.insert(0, 181)
            sub_item.append(181)
            sub_item.append(181)
            dir_tensors.append(torch.LongTensor(sub_item))
        for sub_ in feature_list:
#            sub_.insert(0, 0)
            sub_.append(0)
#            sub_item.append(181)
            fe_tensors.append(torch.LongTensor(sub_))

        traj_pad = pad_sequence(traj_tensors, batch_first=True).to(self.device)
#        print("traj_pad:",traj_pad.size())
#        
        dir_pad =  pad_sequence(dir_tensors, batch_first=True).to(self.device)
        fe_pad =  pad_sequence(fe_tensors, batch_first=True).to(self.device)
#        print(dir_pad.size())
        
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices,alphas)
#        max_sent_len = traj_pad.size()[1]
        max_sent_len = max(traj_lengths)
      
        traj_labels = [sub_traj_label+([0]*(max_sent_len-len(sub_traj_label))) for sub_traj_label in label_list]


        labels_pad = torch.LongTensor(traj_labels).to(self.device)

        return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad

def load_raw_data_1(t_data, t_dir, t_label,t_feature):   ##add flag to control whether to use classifer
    print("----data loading classifer----")
    t_data_1 = load_data(t_data)
    t_dir_1 = load_data(t_dir)
    t_label_1 = load_data(t_label)
    t_feature_1 = load_data(t_feature)
#        t_data = load_data(t_data)
#        t_dir = load_data(t_dir)
#        t_label = load_data(t_label)
#        t_fea = load_data(t_feature)
    t_data =t_data_1[:]
    t_dir =t_dir_1[:]
    t_label =t_label_1[:]
    t_fea =t_feature_1[:]

    t_data_= []
    tag_list = []
    t_label_ =[]
    t_dir_ =  []
    t_fea_ = []
    records = []
    all_data_num = 0
    for i in range(len(t_data)):
        item_pos = []
        item_dir = []
        tag_boundary = []
#        print(item)
        item_pos.extend(t_data[i])
#        item_pos.append('</s>')
        item_pos.append(6112)
        item_dir.extend(t_label[i])
        item_dir.append(0)
#        t_data_.append(item_pos)
#        t_label_.append(item_dir)
#        item_dir.append(t_label[i])
        if sum(t_fea[i])>0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
            for k, item in enumerate(item_dir[:-1]):
#                if k== len(item_dir)-1:
#                     tag_boundary.append(k)
                if item_dir[k]==0:
                    tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
                elif item_dir[k]==1:
                    sub_s = []
                    dict1={}
                    j= k-1
#                    print("j:", j)
                    existing_anomaly = [idx for idx, item in enumerate(item_dir[:]) if item==1]

                    sub_anomaly  = [idx for idx, item in enumerate(item_dir[k:]) if item==1][::-1] 

                    dict1[(existing_anomaly[0], existing_anomaly[-1])] = "Anomaly"
                    record.append(dict1)
                    records.append(record)
                    last_chance = [idx for idx, item in enumerate(item_dir[k:]) if item==1][-1]

                    tag_boundary.extend(sub_anomaly)
#                    print("j+last_chance+1:", j+last_chance+1+1)
                    for kk, sub_item in enumerate(item_dir[j+last_chance+2:-1]):

                        if sub_item == 0:
                            tag_boundary.append(len(item_dir[j+last_chance+2+kk:])-1)
                    tag_list.append(tag_boundary)

                    break;
                
        elif sum(item_dir) ==0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
            for k, item in enumerate(item_dir[:-1]):
                tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
            tag_list.append(tag_boundary)
            records.append(record)

    return t_data_, t_dir_, t_fea_, t_label_
           
class End2EndDataset_test(Dataset):
    def __init__(self, t_data, t_dir, t_label, t_feature, device, evaluating=False):
        super().__init__()
#        self.data_url = data_url
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
#        , self.tag_list, self.records
#        self.sentences, self.tags,self.sent_reverse,self.tag_boundary_reverse,self.records,self.alphaFeatures = load_raw_data(data_url)
        self.anomaly_train, self.anomaly_train_dir, self.train_feature, self.train_label = load_raw_data_test(t_data, t_dir, t_label, t_feature)


        self.device = device
        self.evaluating = evaluating

    def __getitem__(self, index):
# =============================================================================
# #        , self.tag_list[index], self.records[index]
# =============================================================================
        return self.anomaly_train[index], self.anomaly_train_dir[index], self.train_feature[index], self.train_label[index]

    def __len__(self):
        return len(self.anomaly_train)

    def collate_func(self, data_list):
#        print("pay attention to bug")
#        print("data_list:", data_list)
#        print("ori:",data_list)
        data_list = sorted(data_list, key=lambda x: len(x[0]), reverse=True)
#        for item in data_list:
#            print("len_item:", len(item[0]))
#        
#        sentence_list, tags_list,sent_reverse_list,tag_boundary_reverse_list,records_list,alphaFeatures_list = zip(*data_list)
        traj_list, direction_list, feature_list, label_list = zip(*data_list)
#, tag_list, records
        # un zip
#        sentence_tensors = gen_sentence_tensors(sentence_list,self.device)
        traj_lengths = [len(traj) for traj in traj_list]
#        print("traj_list:",traj_list)
#        print("traj_lengths:",traj_lengths)
#        print("tag_list:",tag_list)
#        exit()
#        print("data_list:", data_list[:3])
#        print(len(data_list))
#        print("original_length:", traj_lengths[:5])
        traj_tensors = []
        dir_tensors = []
        fe_tensors=[]
#        label_ =[]
        for item in traj_list:
#            print("sub_traj:", item)
            traj_tensors.append(torch.LongTensor(item))
#        traj_tensors = torch.LongTensor(traj_list)
        for sub_item in direction_list:
            sub_item.insert(0, 181)
            sub_item.append(181)
            sub_item.append(181)
            dir_tensors.append(torch.LongTensor(sub_item))
        for sub_ in feature_list:
#            sub_.insert(0, 0)
            sub_.append(0)
#            sub_item.append(181)
            fe_tensors.append(torch.LongTensor(sub_))
#        for item in label_list:
#            label_.append(torch.LongTensor(item))
#        for i, j, k in zip(traj_list, dir_tensors, label_list):
#            if len(i)==len(j) and len(j)==len(k):
#                pass
#            else:
#                print("traj:", len(i))
#                print("traj:", i)
#                print("dir:", len(j))
#                print("dir:", j)
#                print("label:", len(k))
#                print("label:", k)
        traj_pad = pad_sequence(traj_tensors, batch_first=True).to(self.device)
#        print("traj_pad:",traj_pad.size())
#        
        dir_pad =  pad_sequence(dir_tensors, batch_first=True).to(self.device)
        fe_pad =  pad_sequence(fe_tensors, batch_first=True).to(self.device)
#        print(dir_pad.size())
#        (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices,alphas)
#        max_sent_len = traj_pad.size()[1]
        max_sent_len = max(traj_lengths)
      
        traj_labels = [sub_traj_label+([0]*(max_sent_len-len(sub_traj_label))) for sub_traj_label in label_list]

        labels_pad = torch.LongTensor(traj_labels).to(self.device)
#        for item_1, item_2, item_3 in zip(traj_pad, dir_pad, labels_pad):
#            if len(item_1.data)== len(item_2.data) and len(item_2.data==len(item_3.data):
#                pass
#            else:
#                print("item_1:", item_1)
#                print("item_2:", item_2)
#                print("item_3:", item_3)
#        
#        sentence_reverse_labels = torch.LongTensor(sentence_reverse_labels).to(self.device)
        if self.evaluating:
#            return sentence_tensors, tags_list,sent_reverslabels_pade_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
            return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad
#        return sentence_tensors, tags_list, sent_reverse_list,tag_boundary_reverse_list,records_list,sentence_labels,sentence_reverse_labels
        return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad

def load_raw_data_test(t_data, t_dir, t_label,t_feature):
    print("----data loading----")
#    t_data_1 = load_data(t_data)
#    t_dir_1 = load_data(t_dir)
#    t_label_1 = load_data(t_label)
#    t_feature_1 = load_data(t_feature)
#    t_data = load_data(t_data)
#    t_dir = load_data(t_dir)
#    t_label = load_data(t_label)
#    t_fea = load_data(t_feature)
    
    t_data =t_data[:]
    t_dir =t_dir[:]
    t_label =t_label[:]
    t_fea =t_feature[:]
    
    t_data_= []
    tag_list = []
    t_label_ =[]
    t_dir_ =  []
    t_fea_ = []
    records = []
    all_data_num = 0
    for i in range(len(t_data)):
        item_pos = []
        item_dir = []
        tag_boundary = []
#        print(item)
        item_pos.extend(t_data[i])
#        item_pos.append('</s>')
        item_pos.append(6112)
        item_dir.extend(t_label[i])
        item_dir.append(0)
#        t_data_.append(item_pos)
#        t_label_.append(item_dir)
#        item_dir.append(t_label[i])
        if sum(t_fea[i])>0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
        elif sum(item_dir) ==0:
            record = []
            t_data_.append(item_pos)
            t_label_.append(item_dir)
            t_dir_.append(t_dir[i])
            t_fea_.append(t_fea[i])
            for k, item in enumerate(item_dir[:-1]):
                tag_boundary.append(len(item_dir[k:len(item_dir)])-1)
            tag_list.append(tag_boundary)
            records.append(record)

    return t_data_, t_dir_, t_fea_, t_label_
