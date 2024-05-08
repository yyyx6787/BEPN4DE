# -*- coding: utf-8 -*-



import os
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

#from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import End2EndDataset, compuLoss_entity,lr_decay
from model import End2EndModel
from eval import evaluate_e2e
import pickle
import pdb
import numpy as np
from torch.nn.utils.rnn import pad_sequence
#pdb.set_trace()
from itertools import groupby
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import sys
import argparse
import time
import random
import utils
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

#from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score


EARLY_STOP = 5
LR = 0.001
BATCH_SIZE = 800
MAX_GRAD_NORM = 5
N_TAGS = 6
FREEZE_WV = True
LOG_PER_BATCH = 1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#PRETRAINED_URL = from_project_root("data/embedding/glove.840B.300d_cased.txt")

#TRAIN_URL = from_project_root("data/CONLL2003/train.txt")
#DEV_URL = from_project_root("data/CONLL2003/valid.txt")
#TEST_URL = from_project_root("data/CONLL2003/test.txt")
#train = "./data/processed_porto_anomaly_train.pickle"
#train_dir = "./data/train_anomaly_direction.pickle"
#train_label = "./data/processed_train_label.pickle"
#train_feature = "./data/train_fea.pickle"
#
#val = "./data/processed_porto_anomaly_val.pickle"
#val_dir = "./data/val_anomaly_direction.pickle"
#val_label = "./data/processed_val_label.pickle"
#val_feature = "./data/val_fea.pickle"

gamma = 0.2
alpha = 0.4
de_times = 2.5
grop_num = 2
train = "./data_roadnetwork/chengdu/classfier_data/train_data_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_dir = "./data_roadnetwork/chengdu/classfier_data/train_direction_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_label = "./data_roadnetwork/chengdu/classfier_data/train_label_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))
train_feature = "./data_roadnetwork/chengdu/classfier_data/train_fea_{}_{}.pickle".format(np.round(alpha,1),np.round(de_times,1))

val = "./data_roadnetwork/chengdu/val_data.pickle"
val_dir = "./data_roadnetwork/chengdu/val_direction.pickle"
val_label = "./data_roadnetwork/chengdu/val_label.pickle"
val_feature = "./data_roadnetwork/chengdu/val_feature.pickle"

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

def sampler_(labels):
    _, counts = np.unique(labels, return_counts=True)
#    print(counts)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
#    print(sample_weights)
#    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

def load_raw_data(t_data, t_dir, t_label,t_feature):
    print("----data loading----")
#    t_data_1 = load_data(t_data)
#    t_dir_1 = load_data(t_dir)
#    t_label_1 = load_data(t_label)
#    t_feature_1 = load_data(t_feature)
    t_data = load_data(t_data)
    t_dir = load_data(t_dir)
    t_label = load_data(t_label)
    t_fea = load_data(t_feature)
#    t_data =t_data_1[:100000]
#    t_dir =t_dir_1[:100000]
#    t_label =t_label_1[:100000]
#    t_fea =t_feature_1[:100000]
#    for data, dirt, label in zip(t_data_1, t_dir_1,t_label_1):
#        if len(data) ==len(dirt)+2 and len(dirt)+2==len(label):
#            pass
#        else:
#            print(len(data))
#            print(len(dirt))
#            print(len(label))    
    
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
#                    print("******************************")
#                    print("item_dir:", item_dir)
#                    print("existing_anomaly:",existing_anomaly[0], existing_anomaly[-1])
#                    print("records:", records)
#                    print("tag_boundary:", tag_boundary)
#                    print("item_dir:", len(item_dir))
#                    print("len_tag_boundary:", len(tag_boundary))
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
#            print("len_tag_boundary:", len(tag_boundary))
#            print("record:", record)
    return t_data_, t_dir_, t_fea_, t_label_, tag_list, records
LABEL_IDS = {"O": 0, "Anomaly": 1}
LABEL_LIST = ["O","Anomaly"]
class endDataset(Dataset):
    def __init__(self, t_data, t_dir,t_label, t_feature, device, evaluating=False):
#    def __init__(self, t_data, device):
        super().__init__()
#        self.data_url = data_url
        self.label_ids = LABEL_IDS
        self.label_list = LABEL_LIST
        self.anomaly_train, self.anomaly_train_dir, self.train_feature, self.train_label, self.tag_list, self.records = load_raw_data(t_data, t_dir, t_label, t_feature)
#        print(self.anomaly_train[:5])
#        
#        self.anomaly_train = t_data
        self.device = device
        self.evaluating = evaluating
#        self.anomaly_train_dir = t_dir
#        self.train_feature = t_feature
#    

    def __getitem__(self, index):
        return self.anomaly_train[index], self.anomaly_train_dir[index], self.train_feature[index], self.train_label[index], self.tag_list[index], self.records[index]
#        return self.anomaly_train[index]
    
    def __len__(self):
        return len(self.anomaly_train)

    def collate_func(self, data_list):
#        print(data_list[0])
#        print(len(data_list[0][0]))
        
        data_list = sorted(data_list, key=lambda x: len(x[0]), reverse=True)
#        data_list = sorted(data_list, key=lambda x: len(x), reverse=True)

        
        traj_list, direction_list, feature_list, label_list, tag_list, records = zip(*data_list)

        traj_lengths = [len(traj) for traj in traj_list]

        traj_tensors = []
        dir_tensors = []
        fe_tensors=[]
        for item in traj_list:
            traj_tensors.append(torch.LongTensor(item))
        for sub_item in direction_list:
            sub_item.insert(0, 181)
            sub_item.append(181)
            sub_item.append(181)
            dir_tensors.append(torch.LongTensor(sub_item))
        for sub_ in feature_list:
           sub_.append(0)
           fe_tensors.append(torch.LongTensor(sub_))
        traj_pad = pad_sequence(traj_tensors, batch_first=True).to(self.device)
        dir_pad =  pad_sequence(dir_tensors, batch_first=True).to(self.device)
        fe_pad =  pad_sequence(fe_tensors, batch_first=True).to(self.device)
        max_sent_len = max(traj_lengths)
        
        traj_labels = [sub_traj_label+([0]*(max_sent_len-len(sub_traj_label))) for sub_traj_label in label_list]


        labels_pad = torch.LongTensor(traj_labels).to(self.device)

        if self.evaluating:
            return traj_pad, traj_lengths, dir_pad, fe_pad, labels_pad,  tag_list, records
        return traj_pad, traj_lengths, dir_pad, fe_pad,  labels_pad, tag_list, records
#        return traj_pad


def loader(train, val, train_label, val_label, train_dir, train_feature, val_dir, val_feature):
#    train = load_data(train)
#    train_dir = load_data(train_dir)
#    train_feature = load_data(train_feature)
#    val_dir = load_data(val_dir)
#    val_feature = load_data(val_feature)
#    val = load_data(val)
    train_label_ = load_data(train_label)
    train_label_c =[]
    val_label_c= []
#    [for item in train_label if sum(item)>0]
    for item in train_label_:
        if sum(item)>0:
            train_label_c.append(1)
        else:
            train_label_c.append(0)
    val_label_ = load_data(val_label)
    for item in val_label_:
        if sum(item)>0:
            val_label_c.append(1)
        else:
            val_label_c.append(0)
#    print("train_label:", train_label_c[:50])
#    print(len(train_label_c))
#    print("val_label:", val_label_c[:50])
#    print(len(val_label_c))
#    
#    images, labels, _ = parse_data(data_dir)
#    dataset = ImageDataset(imgages, labels, transform)
#    dataset_size = len(dataset)
    train_size = len(train)
    train_indices = list(range(train_size))
    val_size = len(val)
    val_indices = list(range(val_size))
#    np.random.shuffle(indices) # shuffle the dataset before splitting into train and val
#    split = int(np.floor(train_split * dataset_size))
    
#    train_indices, val_indices = indices[:split], indices[split:]
    train_indices, val_indices = train_indices, val_indices
#    train_labels = 
#    val_labels = 
    train_sampler, val_sampler = sampler_(train_label_c), sampler_(val_label_c)
  
#    trainloader = DataLoader(train, sampler=train_sampler, batch_size=1)
#    valloader = DataLoader(val, sampler=val_sampler, batch_size=1)
#    endDataset(Dataset)
    device='auto'
    device = get_device(device)
#    train_set = endDataset(train, device =device)
    train_set = endDataset(train, train_dir, train_label, train_feature, device=device)
    val_set = endDataset(val, val_dir, val_label, val_feature, device=device)
#    val_set = endDataset(val, device = device)
#    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
#                              collate_fn=train_set.collate_func)
    
#    trainloader = DataLoader(train_set, sampler=train_sampler, batch_size=128, collate_fn=train_set.collate_func)
#    valloader = DataLoader(val_set, sampler=val_sampler, batch_size=128, collate_fn=val_set.collate_func)
    trainloader = DataLoader(train_set, batch_size=128, collate_fn=train_set.collate_func)
    valloader = DataLoader(val_set, batch_size=128, collate_fn=val_set.collate_func)
#    trainloader = DataLoader(dataset, sampler=train_sampler)
#    valloader = DataLoader(dataset, sampler=val_sampler)
    return trainloader, valloader,train_sampler


trainloader, valloader,train_sampler  = loader(train, val, train_label, val_label, train_dir, train_feature, val_dir, val_feature)
#for data,label in zip(trainloader, train_sampler):
#    print("data:", data)
#    print(len(data))
#    print(label)
#    print(len(label))
##    print("label:", label)

from classifier_model import LSTMClassifier
def get_accuracy(truth, pred):
#     print(len(truth))
#     print(len(pred))
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
#     print(right/len(truth))
#     print("-----------")

     return right/len(truth)

def train():
#    train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_MR_data()
#    EMBEDDING_DIM = 50
#    HIDDEN_DIM = 50
    EPOCH = 100
#    best_dev_acc = 0.0
    best_dev_f1 = 0.0
#    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
#                           vocab_size=len(word_to_ix),label_size=len(label_to_ix))
    
    model = LSTMClassifier(hidden_dim=128, n_embeddings_position = 6113,n_embeddings_direction = 182, embedding_dim=300,label_size=2, batch_size=256, use_gpu=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #trainloader = trainloader.to(device)
    model = model.cuda(device)
    
#    loss_function = nn.NLLLoss()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    no_up = 0
    for i in range(EPOCH):
#        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        train_epoch(model, trainloader, loss_function, optimizer, i)
#        print('now best dev acc:',best_dev_acc)
#        model = LSTMClassifier(hidden_dim=128, n_embeddings_position = 8054,n_embeddings_direction = 182, embedding_dim=300,label_size=2, batch_size=128, use_gpu=True)
#        model = model.cuda(device)
#        model.load_state_dict(torch.load('./best_models_pnplus/posdir_mr_best_model_acc_9890_f1_9890.model'))
            
        dev_acc,prec, reca, dev_f1, pred_res = evaluate(model,valloader,loss_function,'dev')
        test_acc,test_prec, test_reca, test_f1, pred_res = evaluate(model, valloader, loss_function, 'test')
#        print("pred_res:",pred_res)
        print("test_acc",test_acc,"test_prec:", test_prec,  "test_reca:",test_reca, "test_f1:",test_f1)
        
#        if dev_acc > best_dev_acc:
#            best_dev_acc = dev_acc
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), './models/best_model_acc_' + str(int(test_acc*10000)) +'_f1_'+ str(int(test_f1*10000))+ '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

def evaluate(model, valloader, loss_function, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = list()
    pred_res = list()
    step=0
    for sentence, setence_lengths, dirt, fe, sentence_labels, tags_lists, records_list in valloader:
        
#        truth_res.append(label_to_ix[label])
#        # detaching it from its history on the last instance.
#        model.hidden = model.init_hidden()
#        sent = data_loader.prepare_sequence(sent, word_to_ix)
#        label = data_loader.prepare_label(label, label_to_ix)
        sentence_labels_sc = sentence_labels.tolist()
        sentence_labels_com = []
        for item in sentence_labels_sc:
            if sum(item)>0:
                sentence_labels_com.append(1)
            else:
                sentence_labels_com.append(0)
#        print(sentence_labels_com)
#        print(sum(sentence_labels_com))
#        print(len(sentence_labels_com))
#        
        sentence_labels_com_ten =  torch.tensor(sentence_labels_com)
        truth_res+= list(sentence_labels_com_ten.int())
        pred = model(sentence, setence_lengths, dirt, tags_lists, fe,  records_list= None, epoch=0, mode='test')
#        pred = model(sent)
#        pred_label = pred.data.cpu().max(1)[1].numpy()
#        pred_res.append(pred_label)
        pred = pred.view(pred.shape[0]).float()
        pred_idx = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
#        pred_idx = torch.max(pred, 1)[1]
        pred_res += list(pred_idx.data.cpu().int())
        # model.zero_grad() # should I keep this when I am evaluating the model?
        
        loss = loss_function(pred, torch.autograd.Variable(sentence_labels_com_ten).float().cuda())
        avg_loss += loss.item()
        step+=1
    avg_loss /= step
    acc = get_accuracy(truth_res, pred_res)
#    precision_score,f1_score,recall_score
    prec = precision_score(truth_res, pred_res)
    reca = recall_score(truth_res, pred_res)
    f1 = f1_score(truth_res, pred_res)
    
#    print(name + ' avg_loss:%g eval acc:%g eval prec:%g eval reca:%g eval f1:%g' % (avg_loss, acc, prec,reca,f1 ))
    return acc,prec, reca, f1,  pred_res



def train_epoch(model, trainloader, loss_function, optimizer,i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = list()
    pred_res = list()
    batch_sent = []

#    for sent, label in train_data:
    for sentence, setence_lengths, dirt, fe, sentence_labels, tags_lists, records_list in trainloader:
#            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        sentence_labels_sc = sentence_labels.tolist()
        sentence_labels_com = []
        for item in sentence_labels_sc:
            if sum(item)>0:
                sentence_labels_com.append(1)
            else:
                sentence_labels_com.append(0)
        sentence_labels_com_ten =  torch.tensor(sentence_labels_com)     
        truth_res+= list(sentence_labels_com_ten.int())

#        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
#        model.hidden = model.init_hidden()
#        sent = data_loader.prepare_sequence(sent, word_to_ix)
#        label = data_loader.prepare_label(label, label_to_ix)
#        pred = model(sent)
        pred = model(sentence, setence_lengths, dirt, tags_lists, fe,  records_list= None, epoch=0, mode='train')
#        print("pred:", pred)
#        print("ground_truth:", sentence_labels_com_ten)
#        print(sum(sentence_labels_com_ten))
#        
#        print("pred.shape[0]:", pred.shape[0])
        pred = pred.view(pred.shape[0]).float()
        
        pred_idx = torch.where(pred>0.5,torch.ones_like(pred),torch.zeros_like(pred))
#        print("pred_idx:",pred_idx)
#        
#        pred_label = predict.data.cpu().max(1)[1].numpy()
    
#        pred_idx = torch.max(pred, 1)[1]
        pred_res += list(pred_idx.data.cpu().int())
#        print("pred_res:", pred_res)
#        print("truth_res:",truth_res)
#        pritnln()
#        pred_res.append(pred_label)
        model.zero_grad()
        
#        print(pred)
        loss = loss_function(pred, torch.autograd.Variable(sentence_labels_com_ten).float().cuda())
#        print(loss.data.item())
#        print(loss.item())
#        
        avg_loss += loss.item()
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.item()))

        loss.backward()
        optimizer.step()
    avg_loss /= count
#    prec = precision_score(truth_res, pred_res)
#    reca = recall_score(truth_res, pred_res)
#    f1 = f1_score(truth_res, pred_res)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g, precision:%g,recall:%g,f1:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res),precision_score(truth_res, pred_res),recall_score(truth_res, pred_res), f1_score(truth_res, pred_res)))

train()

