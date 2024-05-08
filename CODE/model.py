# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torch.autograd as autograd
import dataset

class End2EndModel(nn.Module):
    def __init__(self, hidden_size=10,bidirectional=True,lstm_layers=1,n_embeddings_position = None,n_embeddings_direction = None, embedding_dim=None,freeze=False,is_GRU=True):
        super().__init__()

        self.embedding_position = nn.Embedding(n_embeddings_position, embedding_dim, padding_idx=0)
        self.embedding_direction = nn.Embedding(n_embeddings_direction, embedding_dim, padding_idx=0)
        self.embedding_fe = nn.Embedding(2, embedding_dim, padding_idx=0)
        self.embedding_dim_postion = self.embedding_position.embedding_dim
        self.embedding_dim_direction = self.embedding_direction.embedding_dim
        self.embedding_dim_fe = self.embedding_fe.embedding_dim

        self.word_repr_dim = self.embedding_dim_postion+self.embedding_dim_direction+self.embedding_dim_fe
        #self.word_repr_dim = self.embedding_dim + self.char_feat_dim * lstm_layers
        self.hidden_size = hidden_size
        self.is_GRU = is_GRU

        self.dropout = nn.Dropout(p=0.5) 
        #self.dropout1 = nn.Dropout(p=0.2)

        if is_GRU:
            self.enc = nn.GRU(self.word_repr_dim, self.hidden_size, num_layers=lstm_layers,bidirectional=True, batch_first=True)
            self.enc_bound = nn.GRU(self.word_repr_dim, self.hidden_size, num_layers=lstm_layers,bidirectional=True, batch_first=True)
            self.dec = nn.GRUCell(self.word_repr_dim+self.hidden_size * 2, self.hidden_size*2) # GRUCell's input is always batch first
            self.dec_back = nn.GRUCell(self.word_repr_dim+self.hidden_size * 2,self.hidden_size * 2)  # GRUCell's input is always batch first
            self.enc_frag = nn.LSTM(self.hidden_size * 2, self.hidden_size * 2, bidirectional=True, batch_first=True)
            #self.dec_rl = nn.GRUCell(self.emb_size, self.hidden_size*2) # GRUCell's input is always batch first
        else:
            self.enc = nn.LSTM(self.word_repr_dim, self.hidden_size, num_layers=lstm_layers,bidirectional=True, batch_first=True)
            #self.dec = nn.LSTMCell(self.emb_size, self.hidden_size*2) # LSTMCell's input is always batch first
            self.dec = nn.LSTMCell(self.word_repr_dim, self.hidden_size * 2)  # LSTMCell's input is always batch first
            #self.dec_rl = nn.LSTMCell(self.emb_size, self.hidden_size * 2)  # LSTMCell's input is always batch first
            self.enc_frag = nn.LSTM(self.hidden_size * 2, self.hidden_size * 2, bidirectional=True, batch_first=True)
        self.W1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)  # blending encoder
        self.fc = nn.Linear(self.hidden_size, 2, bias=False)  # encoder for classification
        #self.W2 = nn.Linear(self.hidden_size * 2*2, self.hidden_size * 2, bias=False)  # blending decoder
        self.W2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)  # blending decoder
        self.vt = nn.Linear(self.hidden_size * 2, 1, bias=False)  # scaling sum of enc and dec by v.T
        self.W1_rl = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)  # blending encoder
        self.W2_rl = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=False)  # blending decoder
        self.vt_rl = nn.Linear(self.hidden_size * 2, 1, bias=False)  # scaling sum of enc and dec by v.T
        self.lstm_layers = lstm_layers
#        self.n_tags = n_tags
#        self.n_hidden = (1 + bidirectional) * hidden_size
        self.n_hidden =  bidirectional* hidden_size

        # head and tail features classifier
        self.ht_labeler = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.region_clf = RegionCLF(
            # region_dim=self.hidden_size*2 * 4 + self.length_dim,
            region_dim=self.hidden_size * 2 *3,
            input_dim=self.hidden_size * 2,
            n_classes=2,
        )

    def SegmentRepresent_BiLSTM(self,hidden,start,end):
        frage_input = hidden[start:end].unsqueeze(0).transpose(1, 0).cuda()
        input_packed = torch.nn.utils.rnn.pack_padded_sequence(frage_input, [end - start])
        # Encoding
        encoder_states, hc = self.enc_frag(input_packed) # encoder_state: (bs, L, H)
        encoder_states, encoder_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(encoder_states)
        encoder_states = encoder_states.transpose(1, 0)
        #return torch.cat([encoder_states[0,0,:],encoder_states[0,-1,:]], dim=-1)
        return torch.cat([encoder_states[0, 0, :], torch.mean(encoder_states[0, 0:, :], dim=0), encoder_states[0, -1, :]], dim=-1)

    def addOverlaop(self,list1, list2):
        listOverloap = list1.copy()
        for li2 in list2:
            if (li2 not in listOverloap) and self.boundiscontain(li2, list1):
                listOverloap.append(li2)
        return listOverloap


    def boundiscontain(self,w, wordList):
        flag = False
        for wl in wordList:
            if max(w[0], wl[0]) <= min(w[1], wl[1]):
                flag = True
                break
        return flag
    
    
    def forward(self, sentences, sentence_lengths, dirt, tags_lists, fe, sentence_labels, records_list= None, epoch=0, mode='train'):

        # sentences (batch_size, max_sent_len)
        # sentence_length (batch_size)
#        print("sentences:", sentences)
#        print("sentence_labels:", sentence_labels)
#        print("sentence_labels_shape:", sentence_labels.shape)
        
        batch_labels = sentence_labels.tolist()
        batch_sentence_labels = []
        for item in batch_labels:
            if sum(item)>0:
                batch_sentence_labels.append([1])
            else:
                batch_sentence_labels.append([0])
        batch_labels_tensor = torch.tensor(batch_sentence_labels)
#        print(batch_sentence_labels)
#        
#        
        word_repr = self.embedding_position(sentences)
        max_value = max(sentence_lengths)
#        print("dirt:", dirt)
#        print("printning the shape of tensors")
#        print(sentences.shape[1])
#        print(dirt.shape[1])
        direction_repr = self.embedding_direction(dirt[:,:max_value])
        fe_rec = self.embedding_fe(fe[:,:max_value])
#        print("max_value:", max_value)
#        print("word_repr_shape:", word_repr.shape)
#        print("direction_repr_shape:", direction_repr.shape)
#       
        word_repr = torch.cat([word_repr, direction_repr,fe_rec], dim=-1)
#        alpha_repr = self.alpha_embedding(alphas)
#        word_repr = torch.cat([word_repr, alpha_repr], dim=-1)
        # word_feat shape: (batch_size, max_sent_len, embedding_dim)
        #word_repr = self.dropout(word_repr)
        # add character level feature
#        if self.char_feat_dim > 0:
#            # sentence_words (batch_size, *sent_len, max_word_len)
#            # sentence_word_lengths (batch_size, *sent_len)
#            # sentence_word_indices (batch_size, *sent_len, max_word_len)
#            # char level feature
#            char_feat = self.char_repr(sentence_words, sentence_word_lengths, sentence_word_indices)
#            # char_feat shape: (batch_size, max_sent_len, char_feat_dim)
#            # concatenate char level representation and word level one
#            word_repr = torch.cat([word_repr, char_feat], dim=-1)
#            # word_repr shape: (batch_size, max_sent_len, word_repr_dim)

        # drop out
        word_repr = self.dropout(word_repr)
        #context, _ = self.attention(word_repr,word_repr,word_repr,norm=False)
        #context_ = torch.cat([word_repr, context], dim=-1)
#        print(word_repr.shape)
        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
#        print(packed[0].shape)
#        
        out, hn = self.enc(packed)
    
        # out packed_sequence(batch_size, max_sent_len, n_hidden)
        # hn (n_layers * n_directions, batch_size, hidden_size)

        max_sent_len = sentences.shape[1]
#        print("max_sent_len:", max_sent_len)
        
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, total_length=max_sent_len, batch_first=True)

        # unpacked (batch_size, max_sent_len, n_hidden)

        # self-attention for boosting boundary detection module
        # unpacked,attn_weight = attention(unpacked,unpacked,unpacked)
        unpacked = self.dropout(unpacked)
        #context, _ = self.attention(unpacked , unpacked , unpacked , norm=False)
        #context_ = torch.cat([word_repr, context], dim=-1)
        ###first layer##
        packed_bound = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
        out_bound, hn_bound = self.enc(packed_bound)
#        oout = self.fc(hn_bound[0])
     
#        print("hn_bound[0]:", hn_bound[0].shape)
#        print("hn_bound[0]:", hn_bound[0][0])
#        print("hn_bound[0]:", hn_bound[0][0].size())
        # pred = output.data.max(1, keepdim=True)[1]
#        pred_labels = oout.data.max(1, keepdim=True)[1]
#        print("test_out:", oout.shape)
#        print("test_context:", oout)
#        print("pred_labels:", pred_labels)
#        print("labell_batch:", )
#        
        # out packed_sequence(batch_size, max_sent_len, n_hidden)
        # hn (n_layers * n_directions, batch_size, hidden_size)

        #max_sent_len_bound = sentences.shape[1]
        unpacked_bound, _ = nn.utils.rnn.pad_packed_sequence(out_bound, total_length=max_sent_len, batch_first=True)
#        print("unpacked_shape:", unpacked_bound.shape)
        
        # unpacked (batch_size, max_sent_len, n_hidden)
        #unpacked_bound = self.dropout(unpacked_bound)
        # self-attention for boosting boundary detection module
        # unpacked,attn_weight = attention(unpacked,unpacked,unpacked)
        unpacked_bound, attn_weight = self.attention(unpacked_bound, unpacked_bound, unpacked_bound)
#        print("after_attention:", unpacked_bound.shape)
        
        unpacked_bound = self.dropout(unpacked_bound)
#        print("after_dropout:", unpacked_bound.shape)
#        
        context_ = torch.cat([word_repr, unpacked_bound], dim=-1)
#        print("unpacked_bound:", unpacked_bound.shape)
        sent_first = unpacked_bound.transpose(0, 1)
#        print("sent_first:", sent_first.shape)
#        print("11:", sent_first)

        # sent_first (max_sent_len, batch_size, n_hidden)
        sentence_outputs = None
#        if mode=='train':
        sentence_outputs = torch.stack([self.ht_labeler(token) for token in sent_first], dim=-1)
#            print("sentence_outputs:", sentence_outputs)
#            print("sentence_outputs_shape:", sentence_outputs.shape)
        #sentence_outputs = None
        batch_size = unpacked.size(0)
#        print("sentence_lengths:", sentence_lengths)
#        
        
        hidden_ = [unpacked[i][index - 1] for i, index in enumerate(sentence_lengths)]  # last state
#        for i, index in enumerate(sentence_lengths):
#            print(i,index)
#            print(unpacked[i][index - 2].shape)
#        
#        hidden_rl_ = [unpacked[i][1] for i, index in enumerate(sentence_lengths)]  # first formal entity state
#        for i, index in enumerate(sentence_lengths):
#            print(i,index)
#            print(unpacked[i][1])
#        
        cell_state_ = autograd.Variable(torch.FloatTensor(torch.zeros(batch_size, self.hidden_size * 2))).cuda()
        cell_state_rl_ = autograd.Variable(torch.FloatTensor(torch.zeros(batch_size, self.hidden_size * 2))).cuda()
#        cell_state_ = autograd.Variable(torch.FloatTensor(torch.zeros(batch_size, self.hidden_size * 2)))
#        cell_state_rl_ = autograd.Variable(torch.FloatTensor(torch.zeros(batch_size, self.hidden_size * 2)))
        frag,frag_all,probs_,probs_rl_,probs_entity_ ,labels_entity_,hidden_back= [],[],[],[],[],[],[]
#        tag_rever = None
        frag_combin, frag_combin_,frag_overloap = [],[],[]
        for k in range(batch_size):
#            print("batch_size:", batch_size)
#            print("k:", k)
            cell_state = cell_state_[k].view(1, -1)
            hidden = hidden_[k].view(1, -1)
            cell_state_rl = cell_state_rl_[k].view(1, -1)
#            hidden_rl = hidden_rl_[k].view(1, -1)
            probs,probs_rl,indexs_prob,probs_entity,labels_entity = [],[],[],[],[]
#            ii = sentence_lengths[k] - 1
#            ii_back,kk = sentence_lengths[k] - 1,0
            
#            if mode == 'train':
#                tag_rever = tag_boundary_reverse_list[k]
            frag.clear()
#            frag_reverse.clear()
            hidden_back.clear()
            #left decoder
#            while ii_back > 1:
#                if self.is_GRU:
#                    hidden_rl = self.dec_back(context_ [k, ii_back-1, :].view(1, -1), hidden_rl)  # (h), (h)
#                else:
#                    hidden_rl, cell_state_rl = self.dec_back(context_ [k, ii_back-1, :].view(1,-1), (hidden_rl, cell_state_rl)) # (bs, h), (bs, h)
#                hidden_back.append(hidden_rl)
#                blend1_rl = self.W1_rl(unpacked[k, 0:ii_back, :])  # (bs, 0:ii, W)  bs=1
#                blend2_rl = self.W2_rl(hidden_rl)  # (W)
#                blend_sum_rl = F.tanh(blend1_rl + blend2_rl)  # (L+1, W)
#                out_rl = self.vt_rl(blend_sum_rl).squeeze()  # (L+1)
#                if ii_back == ii:
#                    probs_rl.append(out_rl)
#                    index_rl = out_rl.argmax().item()
##                    if mode == 'train':
##                        if index_rl != 0:
##                            frag_reverse.append((index_rl - 1, ii_back-2))
##                                # indexs_prob_reverse.append(out_rl)
##                        if tag_rever[kk] != 0:
##                            #jj = sentence_lengths[k] - tag_rever[kk]
##                            ii = tag_rever[kk]
##                        else:
##                            #jj += 1
##                            ii -= 1
##                    if mode == 'test':
##                        # out = F.softmax(out, dim=-1)
##                        if index_rl != 0:
##                            frag_reverse.append((index_rl - 1, ii_back - 2))
##                            # indexs_prob_reverse.append(out_rl)
##                            #jj = sentence_lengths[k] - index_rl
##                            ii = index_rl
##                        else:
##                            #jj += 1
##                            ii -= 1
##                    kk = kk + 1
##                ii_back -= 1
##            frag_all_reverse += [frag_reverse.copy()]
#            probs_rl_.append(probs_rl)
            i, j, m = 0, 0, 0
            #right decoder
            while i < (sentence_lengths[k] - 1):
                if self.is_GRU:
#                    print("context_[k, i, :]_shape:", context_[k, i, :].view(1, -1).shape)
                    hidden = self.dec(context_[k, i, :].view(1, -1), hidden)  # (h), (h)
                else:
                    hidden, cell_state = self.dec(context_[k, i, :].view(1,-1), (hidden, cell_state)) # (bs, h), (bs, h)
#                    hidden, cell_state = self.dec(self.decoderinput(encoder_states[k],i,inputlen_list[k]).view(1, -1), (hidden, cell_state))  # (bs, h), (bs, h)
                    # hidden, cell_state = self.dec(encoder_states[k, i, :].view(1, -1),(hidden, cell_state))  # (bs, h), (bs, h)
                    #hidden, cell_state = self.dec(
                        #self.decoderinput(input_[k, i, :], encoder_states[k], i, inputlen_list[k]).view(1, -1),
                        #(hidden, cell_state))  # (bs, h), (bs, h)
                
                
                
                #encoder output(why not each bit to compute each bit, how to represent inactive?????) 
                blend1 = self.W1(unpacked[k, i:sentence_lengths[k], :])  # (bs, i:L+1, W)  bs=1
#                print("blend1:", blend1.shape)
#                print("blend1:", blend1)
                #new_blend_rl = torch.cat([hidden_back[i-1], hidden], dim=-1)
                blend2 = self.W2(hidden)  # (W)
#                print("blend2:", blend2)
                
                #new_blend_rl = self.W2(new_blend_rl)  # (W)
                #blend_sum = F.tanh(blend1 + new_blend_rl)  # (L+1, W)
                blend_sum = F.tanh(blend1 + blend2)  # (L+1, W)
                # out = F.softmax(self.vt(blend_sum).squeeze(),dim=-1)  # (L+1)
                out = self.vt(blend_sum).squeeze()  # (L+1)
#                print("out_shape:", out.shape )
                
                probs.append(out)
#                print("probs_len:", len(probs))
                index = out.argmax().item()
#                print("sentence_lengths[k]:", sentence_lengths[k])
#                print("batch_labels:", sentence_labels[k])
#                print("index:", index)
#                
                #out1 = F.softmax(self.vt(blend_sum).squeeze(),dim=-1)  # (L+1)
                if mode == 'train':
#                    print("sentence_lengths[k] - 1:",sentence_lengths[k] - 1)
#                    print("len:",len(tags_lists[k]))
#                    print("sentence_labels[k]:",len(sentence_labels[k]))
#                    print("fe[k]:", fe[k])
#                    print("j:",j)
#                    print("m:",m)
#                    print("tags_lists[k]:", tags_lists[k])
#                    print("tags_lists[k][j]:",tags_lists[k][j])
#                   
#                    print("tags_lists[k][j] + j + m:",tags_lists[k][j] + j + m)
                    if index != (len(out) - 1):
                        #if out1[index].item()>(1.0/len(out) + 0.2):
                        #if out1[index].item() > 0.2:
#                        print("i-1:", i-1)
#                        print("index:", index)
#                        print("i-1+index:", i-1+index)
                        frag.append((i, i + index))
                        
                        #indexs_prob.append(out)
                    if (tags_lists[k][j] + j + m) != (sentence_lengths[k] - 1):
                        #probs_entity.append(out)
                        #labels_entity.append(tags_lists[k][j])
#                        print("tags_lists:", tags_lists[k])
##                        print("sentences[k]:", sentences[k])
#                        print("k:", k)
#                        print("j:", j)
#                        print("tags_lists[k][j]:",tags_lists[k][j])
#                        print("len(tags_lists[k]):", len(tags_lists[k]))
                        
#                        i = i + tags_lists[k][j] + 1
#                        m = m + tags_lists[k][j]
                        i = i + 1
                        m = m + 1
#                        print("i:", i)
#                        print("m:", m)
  
                    else:
                        i = i + 1
                if mode == 'test':
#                    print("test here 111")
                    # out = F.softmax(out, dim=-1)
#                    print("------------single------------")
#                    print("probs:", probs)
#                    print("index_test:", index)
#                    print("len_out_test:", len(out))
#                    print("out_test:", out)
                    if index != (len(out) - 1) and index!=0:
                        frag.append((i, i + index))
                        #indexs_prob.append(out)
                        i = i + index + 1
                    
                    else: 
                        i = i + 1
#                    print("frag_test:", frag)
               
                j = j + 1
            frag_all += [frag.copy()]
            probs_.append(probs)
#            frag_combin += [list({}.fromkeys(frag.copy() + frag_reverse.copy()).keys())]
#            frag_combin_ += [[b_p for b_p in frag.copy() if b_p in frag_reverse.copy()]]
#            frag_overloap += [self.addOverlaop(frag.copy(), frag_reverse.copy())]
            #probs_entity_.append(probs_entity)
            #labels_entity_.append(labels_entity)
        regions = list()
        region_labels_expand = list()
        if mode=='test':
#            print("test here 222")
            for hidden, bound_records in zip(unpacked, frag_all):
#            for hidden, bound_records in zip(unpacked, frag_overloap):
                for bound_record in bound_records:
                    # regions_labels_filter.append(self.label_ids[records[bound_record]])
                    # regions.append((hidden[bound_record[0]], hidden[bound_record[1]], hidden[bound_record[0]:bound_record[1]+1], self.length_embedding(torch.LongTensor([abs(bound_record[1]+1-bound_record[0])]).cuda())))
                    regions.append(
                        (hidden[bound_record[0]], hidden[bound_record[1]], hidden[bound_record[0]:bound_record[1] + 1]))
#                    print("regions_test:", regions)
#                    regions.append((hidden[bound_record[0]], hidden[bound_record[1]]))
#                    regions.append(self.SegmentRepresent_BiLSTM(hidden, bound_record[0], bound_record[1] + 1))
#                     regions.append(
#                        (hidden[bound_record[0]], hidden[bound_record[1]], self.seg_weight(hidden[bound_record[0]:bound_record[1] + 1])))
#                     regions.append((self.seg_weight(hidden[bound_record[0]:bound_record[1] + 1])))
        else:
            # region_num = 0 # region index
            for hidden, bound_records, records in zip(unpacked, frag_all, records_list):
#                print("records:", records)
                if records ==[]:
                    continue
                else:
                    
                    records_1 = records[0]
    #            for hidden, bound_records, records in zip(unpacked, frag_overloap, records_list):
                    #for bound_record in bound_records
                    for bound_record_true in records_1.keys():
    #                    print("bound_record_true:", bound_record_true)
                        # for bound_record in records.keys():
                        # regions_labels_filter.append(self.label_ids[records[bound_record]])
                        # regions.append((hidden[bound_record[0]], hidden[bound_record[1]], hidden[bound_record[0]:bound_record[1]+1], self.length_embedding(torch.LongTensor([abs(bound_record[1]+1-bound_record[0])]).cuda())))
                        regions.append((hidden[bound_record_true[0]], hidden[bound_record_true[1]],
                                        hidden[bound_record_true[0]:bound_record_true[1] + 1]))
                        #regions.append((hidden[bound_record_true[0]], hidden[bound_record_true[1]]))
                        #regions.append(self.SegmentRepresent_BiLSTM(hidden, bound_record_true[0], bound_record_true[1] + 1))
                        
    #                    print("bound_record_true:", bound_record_true)
    #                    print("records_1[bound_record_true]:", records_1[bound_record_true])
                        
                        region_labels_expand.append(dataset.LABEL_IDS[records_1[bound_record_true]])
    #                     regions.append((hidden[bound_record[0]], hidden[bound_record[1]],
#                                    self.seg_weight(hidden[bound_record[0]:bound_record[1] + 1])))
#                     regions.append((self.seg_weight(hidden[bound_record[0]:bound_record[1] + 1])))
#                    if bound_record in records.keys():
#                        region_labels_expand.append(dataset.LABEL_IDS[records[bound_record]])
#                    else:
#                        region_labels_expand.append(0)
                if epoch>20:
                    for bound_records_pre in bound_records:
                        # if (bound_records_train not in bound_records) and (random.random() >= 0.1):
                        # if (bound_records_train not in bound_records) and (bound_records_train not in records.keys()) and augmentOverlopping(bound_records_train,records):
                        if bound_records_pre not in records_1.keys():
                            # if bound_records_train not in records.keys():
                            # regions.append((hidden[bound_records_train[0]], hidden[bound_records_train[1]],
                            #               hidden[bound_records_train[0]:bound_records_train[1] + 1], self.length_embedding(torch.LongTensor([abs(bound_records_train[1]+1-bound_records_train[0])]).cuda())))
                            regions.append((hidden[bound_records_pre[0]], hidden[bound_records_pre[1]],
                                            hidden[bound_records_pre[0]:bound_records_pre[1] + 1]))
#                            regions.append((hidden[bound_records_pre[0]], hidden[bound_records_pre[1]]))
#                            regions.append(
#                                self.SegmentRepresent_BiLSTM(hidden, bound_records_pre[0], bound_records_pre[1] + 1))
#                             regions.append((hidden[bound_records_train[0]], hidden[bound_records_train[1]],
#                                            self.seg_weight(hidden[bound_records_train[0]:bound_records_train[1] + 1])))
#                             regions.append((self.seg_weight(hidden[bound_records_train[0]:bound_records_train[1] + 1])))
                            region_labels_expand.append(0)
#        regions = torch.cat(regions, dim=0)
##        regions: n_regions, 3*n_hidden
#        region_outputs = self.region_clf(regions)
#        if region_labels_expand != []:
#            region_labels_expand = torch.LongTensor(region_labels_expand).cuda()
#             regions_labels_filter = torch.LongTensor(regions_labels_filter).cuda()
#             shape of region_labels: (n_regions, n_classes)
        region_flag = True
        region_outputs = None
        if len(regions) != 0:
            region_outputs = self.region_clf(regions)
        else:
            region_flag = False
        if len(region_labels_expand) != 0:
            region_labels_expand = torch.LongTensor(region_labels_expand).cuda()
#            region_labels_expand = torch.LongTensor(region_labels_expand)
#        return probs_, probs_rl_,frag_overloap,region_outputs, region_labels_expand, region_flag,sentence_outputs
        return probs_, frag_all,region_outputs, region_labels_expand, region_flag, sentence_outputs


    def attention(self,query,key,value,mask=None,norm=None,dropout=None):
        # compute scaled dot prodcut attention
        d_k = query.size(-1)
        scores = torch.matmul(query,key.transpose(-2,-1))
        if norm:
            scores = scores/math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==1,-1e9)
        p_attn = F.softmax(scores,dim=-1)
        if dropout is not None:
            p_attn = F.dropout(p_attn,p=dropout)
        return torch.matmul(p_attn,value),p_attn


class RegionCLF(nn.Module):
    def __init__(self, region_dim, input_dim, n_classes):
        super().__init__()
        self.region_dim = region_dim
        self.input_dim = input_dim
        self.region_repr = CatRepr()
        self.n_classes = n_classes
#        self.repr_dim = 3 * input_dim
        #self.repr_dim = input_dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.region_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes),
        )

    def forward(self, data_list):
        data_repr = self.region_repr(data_list)
        # data_repr (batch_size, repr_dim)
        return self.fc(data_repr)
        # (batch_size, n_classes)


class CatRepr(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, data_list):
        # shape of data_list: list(batch_size, *input_len, input_dim)
        cat_regions = [torch.cat([hidden[0], hidden[1], torch.mean(hidden[2], dim=0)], dim=-1).view(1, -1)
                      for hidden in data_list]
        #cat_regions = [torch.cat([hidden[0], hidden[1]], dim=-1).view(1, -1)
        #               for hidden in data_list]
        #cat_regions = [torch.mean(hidden, dim=0).view(1, -1)
        #               for hidden in data_list]
#        cat_regions = [torch.cat([hidden[0], hidden[-1]], dim=-1).view(1, -1)
#                       for hidden in data_list]
        #cat_regions = [hidden.view(1, -1) for hidden in data_list]
        cat_out = torch.cat(cat_regions, dim=0)
        # regions (batch_size, 3*input_dim)
        return cat_out

