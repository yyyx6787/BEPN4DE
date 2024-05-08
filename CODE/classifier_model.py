# -*- coding: utf-8 -*-

import warnings
import numpy as np

warnings.filterwarnings('ignore')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#import torch.nn as nn
#import torch.nn.functional as F
#import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):

    def __init__(self, hidden_dim=128, n_embeddings_position = 6113,n_embeddings_direction = 182, embedding_dim=300,label_size=2, batch_size=128, use_gpu=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(n_embeddings_position, embedding_dim)
        self.dirt_embeddings = nn.Embedding(n_embeddings_direction, embedding_dim)
        self.schma_embeddings = nn.Embedding(2, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim*3, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.softmax = nn.LogSoftmax()
#        self.softmax = torch.sigmoid() 
        self.dropout_layer = nn.Dropout(p=0.5)
#        self.hidden2label = nn.Linear(hidden_dim, label_size)
#        self.hidden = self.init_hidden()

    
    		


    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda(),autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda())
#        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).cuda(),autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))
    def forward(self, sentence, sentence_lengths, dirt, tags_lists, fe, records_list= None, epoch=0, mode='train'):
#        print("sentence.size(0):", sentence.size(0))
        max_sent_len = sentence.shape[1]
#        print("max_sent_len:",max_sent_len)
        self.hidden = self.init_hidden(sentence.size(0))
        embeds_word = self.word_embeddings(sentence)
#        [:,:max_sent_len]
#        print("dirt:", type(dirt))
        embeds_dirt = self.dirt_embeddings(dirt[:,:max_sent_len])
        embeds_fe = self.schma_embeddings(fe[:,:max_sent_len])
#        print("embeds_word:",embeds_word.shape)
#        print("embeds_dirt:",embeds_dirt.shape)
#        print("embeds_fe:",embeds_fe.shape)
##        println()
        sentence_repr = torch.cat([embeds_word, embeds_dirt, embeds_fe], dim=-1)
#        sentence_repr = torch.cat([embeds_word,  embeds_dirt], dim=-1)
#        sentence_repr = embeds_word
        
#        packed_input = pack_padded_sequence(sentence_repr, sentence_lengths)
        packed_input = pack_padded_sequence(sentence_repr, sentence_lengths, batch_first=True)
#        print(packed_input[0].data)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
            
        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
#        output = self.softmax(output)
        output = torch.sigmoid(output) 
        
        return output
       




	



		
		

#class LSTMClassifier(nn.Module):
#    def __init__(self, hidden_size=128,bidirectional=True,lstm_layers=1,n_embeddings_position = 8054,n_embeddings_direction = 182, embedding_dim=300, output_size=2):
#        super(LSTMClassifier, self).__init__()
#        self.embedding_dim = embedding_dim
#        self.hidden_dim = hidden_size
#        self.embedding_1 = nn.Embedding(n_embeddings_position, embedding_dim)
#        self.embedding_2 = nn.Embedding(n_embeddings_direction, embedding_dim)
#        self.embedding_3 = nn.Embedding(2, embedding_dim)
#        self.lstm = nn.LSTM(3*self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
##        self.lstm = nn.LSTM(3*self.embedding_dim, self.hidden_dim, num_layers=1)
#        self.hidden2out = nn.Linear(self.hidden_dim, output_size)
#        self.softmax = nn.LogSoftmax()
#        self.dropout_layer = nn.Dropout()
#        # self.dense2 = torch.nn.Linear(128, 10)
#
##    def init_hidden(self, batch_size):
##        # print("......:",autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).shape,
##        # 				autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).shape)
##        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
##                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))
#
##    def attention_net(self, lstm_output, final_state):
##        hidden = final_state.squeeze(0)
##        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
##        soft_attn_weights = F.softmax(attn_weights, 1)
##        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
##
##        return new_hidden_state
#    def forward(self, sentences, sentence_lengths, dirt, tags_lists, fe, records_list= None, epoch=0, mode='train'):
#        max_sent_len = sentences.shape[1]
#        print("max_sent_len:", max_sent_len)
##        self.hidden = self.init_hidden(batch.shape[0])
#        word_repr = self.embedding_1(sentences)
#        # print("embeds_1:",embeds_1)
#        direction_repr = self.embedding_2(dirt)
#        fe_rec = self.embedding_3(fe)
##        embeds_1 = torch.reshape(embeds_1, (batch.shape[0], -1, self.embedding_dim * 1))
##        embeds_2 = torch.reshape(embeds_2, (batch.shape[0], -1, self.embedding_dim * 8))
#        word_repr = torch.cat([word_repr, direction_repr, fe_rec], dim=-1)
#        
#        print("after_concat:",word_repr.shape)
##        println()
##        word_repr_reshape = word_repr.view(word_repr.shape[0],1, -1)
##        print(word_repr_reshape.shape)
##        println()
#        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
##        print(type(sentence_lengths))
##        print(sentence_lengths)
##        sentence_lengths_tensor = torch.tensor(sentence_lengths)
##        packed = nn.utils.rnn.pack_padded_sequence(word_repr_reshape, sentence_lengths_tensor.cpu(), batch_first=True,enforce_sorted=False)
#        print(packed[0].data.shape)
##        print(packed[0].shape)
##        print(packed[0])
##        h0 = torch.zeros(self.num_layers * 2, word_repr.size(0), self.hidden_dim)  # 2 for bidirection
##        c0 = torch.zeros(self.num_layers * 2, word_repr.size(0), self.hidden_dim)
#        
#        outputs, (ht, ct) = self.lstm(packed)
##        print("ht_shape:", ht.shape)
##        println()
#        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=max_sent_len, batch_first=True)
##        attn_output = self.attention_net(outputs, ht)
##        
##        output  = self.hidden2out(attn_output)
#        print("after_pad:", unpacked.shape)
##        println()
#        unpacked_reshape = unpacked.reshape(unpacked.shape[0], -1)
#        # output = self.dropout_layer(output)
##        logits = self.softmax(ht)
##        print("unpacked_reshape:", unpacked_reshape.shape)
##        println()
#        output = self.dropout_layer(unpacked_reshape)
#        print("after_drop:", output.shape)
#        output = self.hidden2out(output)
#        print("after_linear:", output.shape)
#        output = output.squeeze(1)
#        print("after_squeeze:", output.shape)
#        # # output = output.squeeze(1)
#        # # print("output_hidden2out:", output)
#        # # print("output:", output.shape)
#        output = self.softmax(output)
#
#        return output
    

		

	



		
		

#class LSTMClassifier(nn.Module):
#    def __init__(self, hidden_size=128,bidirectional=True,lstm_layers=1,n_embeddings_position = 8054,n_embeddings_direction = 182, embedding_dim=300, output_size=2):
#        super(LSTMClassifier, self).__init__()
#        self.embedding_dim = embedding_dim
#        self.hidden_dim = hidden_size
#        self.embedding_1 = nn.Embedding(n_embeddings_position, embedding_dim)
#        self.embedding_2 = nn.Embedding(n_embeddings_direction, embedding_dim)
#        self.embedding_3 = nn.Embedding(2, embedding_dim)
#        self.lstm = nn.LSTM(3*self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
##        self.lstm = nn.LSTM(3*self.embedding_dim, self.hidden_dim, num_layers=1)
#        self.hidden2out = nn.Linear(self.hidden_dim, output_size)
#        self.softmax = nn.LogSoftmax()
#        self.dropout_layer = nn.Dropout()
#        # self.dense2 = torch.nn.Linear(128, 10)
#
##    def init_hidden(self, batch_size):
##        # print("......:",autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).shape,
##        # 				autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)).shape)
##        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
##                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))
#
##    def attention_net(self, lstm_output, final_state):
##        hidden = final_state.squeeze(0)
##        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
##        soft_attn_weights = F.softmax(attn_weights, 1)
##        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
##
##        return new_hidden_state
#    def forward(self, sentences, sentence_lengths, dirt, tags_lists, fe, records_list= None, epoch=0, mode='train'):
#        max_sent_len = sentences.shape[1]
#        print("max_sent_len:", max_sent_len)
##        self.hidden = self.init_hidden(batch.shape[0])
#        word_repr = self.embedding_1(sentences)
#        # print("embeds_1:",embeds_1)
#        direction_repr = self.embedding_2(dirt)
#        fe_rec = self.embedding_3(fe)
##        embeds_1 = torch.reshape(embeds_1, (batch.shape[0], -1, self.embedding_dim * 1))
##        embeds_2 = torch.reshape(embeds_2, (batch.shape[0], -1, self.embedding_dim * 8))
#        word_repr = torch.cat([word_repr, direction_repr, fe_rec], dim=-1)
#        
#        print("after_concat:",word_repr.shape)
##        println()
##        word_repr_reshape = word_repr.view(word_repr.shape[0],1, -1)
##        print(word_repr_reshape.shape)
##        println()
#        packed = nn.utils.rnn.pack_padded_sequence(word_repr, sentence_lengths, batch_first=True)
##        print(type(sentence_lengths))
##        print(sentence_lengths)
##        sentence_lengths_tensor = torch.tensor(sentence_lengths)
##        packed = nn.utils.rnn.pack_padded_sequence(word_repr_reshape, sentence_lengths_tensor.cpu(), batch_first=True,enforce_sorted=False)
#        print(packed[0].data.shape)
##        print(packed[0].shape)
##        print(packed[0])
##        h0 = torch.zeros(self.num_layers * 2, word_repr.size(0), self.hidden_dim)  # 2 for bidirection
##        c0 = torch.zeros(self.num_layers * 2, word_repr.size(0), self.hidden_dim)
#        
#        outputs, (ht, ct) = self.lstm(packed)
##        print("ht_shape:", ht.shape)
##        println()
#        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=max_sent_len, batch_first=True)
##        attn_output = self.attention_net(outputs, ht)
##        
##        output  = self.hidden2out(attn_output)
#        print("after_pad:", unpacked.shape)
##        println()
#        unpacked_reshape = unpacked.reshape(unpacked.shape[0], -1)
#        # output = self.dropout_layer(output)
##        logits = self.softmax(ht)
##        print("unpacked_reshape:", unpacked_reshape.shape)
##        println()
#        output = self.dropout_layer(unpacked_reshape)
#        print("after_drop:", output.shape)
#        output = self.hidden2out(output)
#        print("after_linear:", output.shape)
#        output = output.squeeze(1)
#        print("after_squeeze:", output.shape)
#        # # output = output.squeeze(1)
#        # # print("output_hidden2out:", output)
#        # # print("output:", output.shape)
#        output = self.softmax(output)
#
#        return output
    