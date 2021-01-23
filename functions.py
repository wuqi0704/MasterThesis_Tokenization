#!/usr/bin/env python
# coding: utf-8

#%%
import pickle
LanguageList = [
    'HEBREW',
    'ARABIC',
    'PORTUGUESE',
    'ITALIAN',
    'FRENCH',
    'SPANISH',
    'GERMAN',
    'ENGLISH',
    'RUSSIAN',
    'FINNISH',
    'VIETNAMESE',
    'KOREAN',
    'CHINESE',
    'JAPANESE'
]
data_train,data_test,data_dev=[],[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
        dev = pickle.load(f3)
    
    data_train += train; data_test += test; data_dev += dev


# %% character dictionary set and define other helper functions
import numpy as np
letter_to_ix = {}
for sent, tags in data_train+data_test:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)+1 # leave index 0 out 
print('Nr. of distinguish character: ',len(letter_to_ix.keys()))
# print(letter_to_ix.keys())

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4} 
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



#%%
import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class LSTMTagger(nn.Module):
    def __init__(self, character_size, embedding_dim, hidden_dim,num_layers,tagset_size,batch_size) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.character_embeddings = nn.Embedding(character_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self,sentence):
        if self.batch_size>1:
            embeds = self.character_embeddings(sentence)
            # print(embeds.shape)
            x = embeds.view(len(sentence), self.batch_size, -1)
            # print('x',x.shape)
            h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            out, _ = self.lstm(x, (h0, c0))
            # print(out.shape)
            tag_space = self.hidden2tag(out.view(len(sentence),self.batch_size, -1))
            # print(tag_space.shape)
            tag_scores = F.log_softmax(tag_space, dim=2) # dim = (len(sentence),batch,len(tag))
        
        else:   
            embeds = self.character_embeddings(sentence)
            # print(embeds.shape)
            x = embeds.view(len(sentence), 1, -1) # add one more dimension, batch_size = 1 
            # print('x',x.shape)
            out, _ = self.lstm(x)
            # print(out.shape)
            print(out.view(len(sentence), -1).shape)
            tag_space = self.hidden2tag(out.view(len(sentence), -1))
            # print(tag_space.shape)
            tag_scores = F.log_softmax(tag_space, dim=1)
            # print(tag_scores.shape)

        return tag_scores

print("yohooo")
# #%% try example 

# def prepare_batch(batch, to_ix):
#     tensor_list = []
#     for seq in batch:
#         idxs = [to_ix[w] for w in seq]
#         tensor = torch.tensor(idxs, dtype=torch.long)
#         tensor_list.append(tensor)
#     return pad_sequence(tensor_list,batch_first=False)

# tagset_size = 5
# embedding_dim = 256 
# hidden_dim = 256 
# EPOCH = 10
# learning_rate = 0.1
# batch_size = 2
# num_layers = 1
# character_size = len(letter_to_ix)

# # mini_batch of datasets
# train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

# # model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size=1)
# # (sentence,tag) = data_train[0]
# # sentence_in = prepare_sequence(sentence,letter_to_ix)
# # targets = prepare_sequence(tag,tag_to_ix)
# # tag_scores = model(sentence_in)
# # tag_scores.shape

# model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size=2)

# (data, tag) = list(train_loader)[9]

# batch_in = prepare_batch(data,letter_to_ix)
# targets = prepare_batch(tag,tag_to_ix)
# tag_scores = model(batch_in)


# print(tag_scores.shape)
# targets

# #%%

# tag_scores.shape
# targets
# #%%
# nn.NLLLoss(tag_scores,targets)