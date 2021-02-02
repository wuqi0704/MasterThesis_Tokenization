#!/usr/bin/env python
# coding: utf-8
#%% 
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
g1 = ['HEBREW','ARABIC']
g2 = ['PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH','FINNISH']
g3 = ['RUSSIAN', 'KOREAN']
g4 = ['CHINESE','JAPANESE']
g5 = ['VIETNAMESE']
GroupList = [g1,g2,g3,g4,g5]
GroupNameList = ['group%s'%str(i) for i in range(1,6)]

# character dictionary set and define other helper functions

import pickle
data_train,data_test,data_dev=[],[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
        dev = pickle.load(f3)
    
    data_train += train; data_test += test; data_dev += dev

import numpy as np
letter_to_ix = {}
letter_to_ix[''] = 0 # need this for padding
for sent, tags in data_train+data_test+data_dev:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ',len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4} 
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prediction_str(input):
        output = [np.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list:
            out_str += o 
        return out_str

# create token list from BIESX tag 
def find_token(sentence_str):
    token = []; word = ''
    
    for  i,tag in enumerate(sentence_str[1]):
        if tag == 'S':
            token.append(sentence_str[0][i])
            continue
        if tag == 'X': 
            continue 
        if (tag == 'B') | (tag == 'I'): 
            word += sentence_str[0][i] 
            continue
        if tag == 'E': 
            word+=sentence_str[0][i]
            token.append(word)
            word=''
    return token

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

from torch.nn.utils.rnn import pad_sequence
def prepare_batch(batch, to_ix):
    tensor_list = []
    for seq in batch:
        idxs = [to_ix[w] for w in seq]
        tensor = torch.tensor(idxs, dtype=torch.long)
        tensor_list.append(tensor)
    return pad_sequence(tensor_list,batch_first=False)
    # with batch_first=False, the dimension come as (len(seq)#length of longest sequence,len(batch)#batch_size)

def prepare_cse(sentence,batch_size=1):
    lm_f: LanguageModel = FlairEmbeddings('multi-forward').lm
    lm_b: LanguageModel = FlairEmbeddings('multi-backward').lm 
    if batch_size == 1:
        embeds_f = lm_f.get_representation([sentence],'\n','\n')[1:-1,:,:]
        embeds_b = lm_b.get_representation([sentence],'\n','\n')[1:-1,:,:]
    elif batch_size >1:
        embeds_f = lm_f.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
        embeds_b = lm_b.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
    return torch.cat((embeds_f,embeds_b),dim=2)


import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

class LSTMTagger(nn.Module):

    def __init__(self, 
                 character_size, 
                 embedding_dim, 
                 hidden_dim,
                 num_layers,
                 tagset_size,
                 batch_size,
                 use_CSE = False,
    ) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_CSE = use_CSE

        self.character_embeddings = nn.Embedding(character_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self,sentence):
        if self.batch_size > 1:
            if self.use_CSE == True:
                embeds = sentence
            elif self.use_CSE == False:
                embeds = self.character_embeddings(sentence) 

            x = embeds # #.view(len(embeds), self.batch_size, -1)
            h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            out, _ = self.lstm(x, (h0, c0))
            # tag_space = self.hidden2tag(out.view(len(sentence),self.batch_size, -1))
            tag_space = self.hidden2tag(out)
            tag_scores = F.log_softmax(tag_space, dim=2) # dim = (len(sentence),batch,len(tag))
        
        elif self.batch_size == 1:   
            if self.use_CSE == True:
                embeds = sentence
            elif self.use_CSE == False:
                embeds = self.character_embeddings(sentence) 

            x = embeds.view(len(sentence), 1, -1) # add one more dimension, batch_size = 1 
            out, _ = self.lstm(x)
            tag_space = self.hidden2tag(out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


#%% # Initialize network BiLSTM
# define hyper parameter 
tagset_size = len(tag_to_ix)
character_size = len(letter_to_ix)
embedding_dim = 256
hidden_dim = 256 
learning_rate = 0.1
num_layers = 1
batch_size = 1
use_CSE = False

MAX_EPOCH = 10
shuffle = True
batch_first = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

def initialize_model(tagset_size = tagset_size,
                     character_size = character_size,
                     embedding_dim = embedding_dim,
                     hidden_dim = hidden_dim,
                     num_layers = num_layers,                   
                     batch_size = batch_size,
                     use_CSE=use_CSE,
                     learning_rate = learning_rate,
                    ):
    model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size,use_CSE)
    
    if(torch.cuda.is_available()):
        print('GPU is available:',torch.cuda.current_device())

    model = model.to(device); model.train()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.NLLLoss()
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    return model, optimizer,loss_function,checkpoint



