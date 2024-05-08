# -*- coding: utf-8 -*-


#from utils.torch_util import set_random_seed

RANDOM_SEED = 233
#set_random_seed(RANDOM_SEED)
BATCH_SIZE_Classfier  = 128
import os
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
#from utils.path_util import from_project_root, exists
from utils.torch_util import get_device
from dataset import End2EndDataset, compuLoss_entity,lr_decay
from model import End2EndModel
from eval import evaluate_e2e
#from eval_classifier import evaluate_e2e
import pdb
#pdb.set_trace()
from itertools import groupby

EARLY_STOP = 5
LR = 0.001
BATCH_SIZE = 32
MAX_GRAD_NORM = 5
N_TAGS = 6
FREEZE_WV = True
LOG_PER_BATCH = 1

#CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.backends.cudnn.enabled=False
#PRETRAINED_URL = from_project_root("data/embedding/glove.840B.300d_cased.txt")

#TRAIN_URL = from_project_root("data/CONLL2003/train.txt")
#DEV_URL = from_project_root("data/CONLL2003/valid.txt")
#TEST_URL = from_project_root("data/CONLL2003/test.txt")

#thre = 0.3
#
#train = "./data_roadnetwork/chengdu/train_data.pickle"
#train_dir = "./data_roadnetwork/chengdu/train_direction.pickle"
#train_label = "./data_roadnetwork/chengdu/train_label.pickle"
#train_feature = "./data_roadnetwork/chengdu/train_fea_{}.pickle".format(thre)
#
#val = "./data_roadnetwork/chengdu/val_data.pickle"
#val_dir = "./data_roadnetwork/chengdu/val_direction.pickle"
#val_label = "./data_roadnetwork/chengdu/val_label.pickle"
#val_feature = "./data_roadnetwork/chengdu/val_fea_{}.pickle".format(thre)

#def train_end2end(n_epochs=50,freeze=FREEZE_WV,train_data = train, train_dir = train_dir,train_label = train_label,train_feature = train_feature,
#                  val_data = val, val_dir = val_dir,val_label = val_label,val_feature = val_feature,learning_rate=LR,batch_size=BATCH_SIZE,early_stop=EARLY_STOP,
#                  clip_norm=MAX_GRAD_NORM,bsl_model_url=None, beta=1.0,gamma=0.2,device='auto',save_only_best=True,test_url=None):
def train_end2end(n_epochs,freeze,train_data, train_dir,train_label,train_feature,
                  val_data, val_dir, val_label, val_feature, learning_rate, batch_size, early_stop,
                  clip_norm, bsl_model_url, beta, gamma, device, save_only_best, test_url, alpha, de_times, grop_num):

    # print arguments
#    arguments = json.dumps(vars(), indent=2)
#    print("arguments", arguments)
    start_time = datetime.now()

    device = get_device(device)
    flag =False
    train_set = End2EndDataset(train_data, train_dir, train_label, train_feature, flag = flag, device=device)
#    print("train_set:", train_set[:1])
#    print("train_set_length:", len(train_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)

    model = End2EndModel(hidden_size=128,bidirectional=True,lstm_layers=1,n_embeddings_position = 6113,n_embeddings_direction = 182, embedding_dim=300,freeze=freeze)
    print("load model is finished")
    
    def my_fscore_unify(Y_true, Y_score, flag='soft'):
        thres = set(sum(Y_score, []))
        res = []
        #print('thres', thres)
        for the in thres:
            fscore = []
            for y_true, y_score in zip(Y_true, Y_score):
                tmp =  [1 if s >= the else 0 for s in y_score]
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
    
    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
        # model = nn.DataParallel(model)
    else:
        print("using cpu\n")
    model = model.to(device)
    bsl_model = torch.load(bsl_model_url) if bsl_model_url else None

    criterion = F.cross_entropy

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate)
    cnt = 0

    max_f1, max_f1_epoch = 0, 0
    max_pre_test, max_re_test,max_f1_test = 0.0,0.0,0.0
    best_model_url = None
    for epoch in range(n_epochs):
       start_time = datetime.now()
       # switch to train mode
       model.train()
       batch_id = 0
       #optimizer = lr_decay(optimizer, epoch, learning_rate, decay_rate)
       print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
       for sentence, setence_lengths, dirt, fe, sentence_labels, tags_lists, records_list in train_loader:
#          print("sentence_labels:", sentence_labels[30][:setence_lengths[30]])
#          print("tags_lists:",tags_lists)
           
           import numpy as np
           optimizer.zero_grad()
#          probs_, probs_rl_,frag_all,region_outputs, region_labels_expand, region_flag,sentence_outputs = model.forward(*data, traj_len_lists=traj_lens,dirt =dirt, epoch=epoch)
           probs_, frag_all,region_outputs, region_labels_expand, region_flag, sentence_outputs = model.forward(sentence, setence_lengths, dirt,  tags_lists, fe, sentence_labels, records_list, epoch)
#          print("tags_lists:",tags_lists)
         
#          loss_sentence = criterion(sentence_outputs, sentence_labels)
           loss_bound = compuLoss_entity(probs_, tags_lists)
           
#          loss_bound_back = compuLoss_entity(probs_rl_, tag_boundary_reverse_list)
           loss_classification = criterion(region_outputs, region_labels_expand) if region_flag else 0
#          loss = gamma * loss_classification + (1 - gamma) * (loss_bound*0.5 + loss_bound_back*0.5+loss_sentence*beta)
#          loss = gamma * loss_classification + (1 - gamma) * (loss_bound*0.5 +loss_sentence*beta)
#          loss = loss_bound
           loss = gamma * loss_classification + (1 - gamma) * loss_bound
#            loss = (1 - gamma) * (loss_bound*0.5 +loss_sentence*beta)
           loss.backward()
            
           # gradient clipping
           if clip_norm > 0:
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
           optimizer.step()
           if batch_id % LOG_PER_BATCH == 0:
               print("epoch #%d, batch #%d, loss: %.12f, %s" %
                     (epoch, batch_id, loss.item(), datetime.now().strftime("%X")))
           batch_id += 1
       cnt += 1
#       evaluating model use development dataset or and additional test dataset
       print("--------------------evaluating--------------------")
#       precision, recall, f1 = evaluate_e2e(model, val_data, val_dir, val_label,val_feature)
       f1 = evaluate_e2e(model, val_data, val_dir, val_label,val_feature)
#       print("P: %.6f  R: %.6f  F1: %.6f" % (precision, recall, f1))
       print("F1: %.6f" % (f1))
#       print("max_f1:", max_f1)
#       print("f1:", f1)
       if f1 > max_f1:
           print("max_f1:", max_f1)
           print("--------------saving model--------------")
           max_f1, max_f1_epoch = f1, epoch
           name = 'split' if bsl_model else 'end2end'
           if save_only_best and best_model_url:
               os.remove(best_model_url)
           best_model_url = "./models/%s_model_epoch%d_%f_%f_%f.pt" % (name, epoch, f1,gamma,de_times)
           torch.save(model, best_model_url)
           torch.save(model.state_dict(), "./models/%s_model_epoch%d_%f_%f_%f.pth" % (name, epoch, f1,gamma,de_times))
#          torch.save(model.state_dict(), "./data/model/best_model.pth")
           cnt = 0

#      print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
#      print("training time:", str(datetime.now() - start_time).split('.')[0])


def main():
    start_time = datetime.now()
    
    val = "./data_roadnetwork/chengdu/val_data.pickle"
    val_dir = "./data_roadnetwork/chengdu/val_direction.pickle"
    val_label = "./data_roadnetwork/chengdu/val_label.pickle"
    val_feature = "./data_roadnetwork/chengdu/val_feature.pickle"
    
#    val = "./efficiency/combine_data"
#    val_dir = "./efficiency/combine_dir"
#    val_label = "./efficiency/combine_label"
#    val_feature = "./efficiency/combine_fe"
    
#    val = "./efficiency/2/untest_data_combine2"
#    val_label = "./efficiency/2/untest_label_combine2"
#    val_dir  = "./efficiency/2/untest_dir_combine2"
#    val_feature = "./efficiency/2/untest_fe_combine2"

    gamma = 0.2
    alpha = 0.4
    de_times = 2.5
    grop_num = 2
    train = "./data_roadnetwork/chengdu/train_data_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1))
    train_dir = "./data_roadnetwork/chengdu/train_direction_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1))
    train_label = "./data_roadnetwork/chengdu/train_label_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1))
    train_feature = "./data_roadnetwork/chengdu/train_fea_{}_{}_n.pickle".format(np.round(alpha,1),np.round(de_times,1))

    train_end2end(50,FREEZE_WV,train,train_dir,train_label,train_feature,val,val_dir,val_label,val_feature,LR,BATCH_SIZE,EARLY_STOP,MAX_GRAD_NORM,None,1.0,np.round(gamma,1),'auto',True,True,alpha,de_times,grop_num)

    print("finished in:", datetime.now() - start_time)
    pass

if __name__ == '__main__':
    main()
