#!/usr/bin/env python
# coding: utf-8

# # Evaluation multi-language model

# In[1]:
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

import pickle
from tqdm import tqdm

data_train,data_test=[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    data_train += train
    data_test += test  

letter_to_ix = {}
for sent, tags in data_train+data_test:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('Nr. of distinguish character: ',len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prediction(input):
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

# In[4]:
# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
# import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
# import torchvision.transforms as transforms  # Transformations we can perform on our dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

class LSTMTagger(nn.Module):
    def __init__(self, character_size, embedding_dim, hidden_dim,num_layers,tagset_size) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
#         self.flair_embeddings = language_model.get_representation()
        self.character_embeddings = nn.Embedding(character_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=False, bidirectional=True,dropout=0.5)
#         self.lm_f: LanguageModel = FlairEmbeddings('multi-forward').lm
#         self.lm_b: LanguageModel = FlairEmbeddings('multi-backward').lm  
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self,sentence):
        embeds = self.character_embeddings(sentence)
#         s_m,e_m = '\n','\n' # start_marker and end_marker
#         embeds_f = self.lm_f.get_representation([sentence],s_m,e_m)[1:-1,:,:] # 1:-1 because the start and end marker are not needed
#         embeds_b = self.lm_b.get_representation([sentence],s_m,e_m)[1:-1,:,:]
        # how to construct bi embedding using both forward and backward embedding? 
#         embeds = embeds_f 
        
        # figure out the output and the num_layers's role in the framework
        x = embeds.view(len(sentence), 1, -1) # add one more dimension, batch_size = 1 
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device) # batch size is 1 
        out, _ = self.lstm(x, (h0, c0))

        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# In[60]:

def save_checkpoint(state, filename=model_name):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# ### Initialize the model 

# In[61]:
# Hyperparameters
num_layers = 1
EMBEDDING_DIM = 2048
HIDDEN_DIM = 256 # HIDDEN_DIM = 6
EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1

# Initialize network
model = LSTMTagger(embedding_dim=EMBEDDING_DIM, num_layers = num_layers,hidden_dim=HIDDEN_DIM,tagset_size=5,character_size=len(letter_to_ix))

if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device)
model.train()

optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()
checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}


# In[2]:
# load test datasets
data_test={}
for language in LanguageList:
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    data_test[language]  = test 
# manually delete datasets that has a mismatch of the tag vs sentence length
# note: effective way to deal with the data , the error is inside WhiteSpace_After.Or sth else
data_test_correct = {}
for language in LanguageList:
    data_test_correct[language]=[]
    for sentence,tags in data_test[language]:
        if len(sentence) == len(tags):
            data_test_correct[language].append((sentence,tags))
    lost = len(data_test[language])-len(data_test_correct[language])
    if lost !=0: print('test',language,lost)
# update the datasets for training that has matching length of sentence and targets 
data_test = data_test_correct

#%%
### Word Level Evaluation 
# case 1 : BiLSTM_ML
# load trained model 
load_model = True
model_name = 'LSTM_V3_multilanguage.pth.tar'
if load_model:
    load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)

from tqdm import tqdm
import numpy as np
error_sentence_ML = {}; Result_ML = {}
with torch.no_grad():
    import sklearn
    for language in LanguageList:
        error_sentence_ML[language] = []
        R_score,P_score,F1_score = [],[],[]
        for element in tqdm(data_test[language],position=0):
            
            inputs = prepare_sequence(element[0],letter_to_ix)
            tag_scores = model(inputs)
            tag_predict = prediction(tag_scores)
    #         print(tag_predict)

            reference = find_token(element)
            candidate = find_token((element[0],tag_predict))

            inter = [c for c in candidate if c in reference]
            if len(candidate) !=0:
                R = len(inter) / len(reference)
                P = len(inter) / len(candidate)
            else: error_sentence_ML[language].append((element,tag_predict))
            if (R+P)  != 0 : 
                F1 = 2 * R*P / (R+P)
            else: 
                error_sentence_ML[language].append((element,tag_predict))
                F1=0
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_ML[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

import pickle
filename = 'BiLSTM_ML'
with open('./results/%s.pickle'%filename, 'wb') as f:
    pickle.dump(Result_ML, f)
#%%
import pandas as pd
results = pd.DataFrame.from_dict(Result_ML, orient='index')
results.columns = ['Recall','Precision','F1 score']
results

# %%
print(error_sentence_ML)
#%%
# type of issues:
# 1. fail to extract tokens
# 2. fail to match any correct tokens


#%%
### Word Level Evaluation 
# case 2 : BiLSTM_SL
# load trained model 
load_model = True
model_name = 'BiLSTM_SL%s.pth.tar'%language

from tqdm import tqdm;import numpy as np;import sklearn

error_sentence_SL = {}; Result_SL = {}
with torch.no_grad():
    for language in LanguageList:
        if load_model:
            load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)
        error_sentence_SL[language] = []
        R_score,P_score,F1_score = [],[],[]
        for element in tqdm(data_test[language],position=0):
            
            inputs = prepare_sequence(element[0],letter_to_ix)
            tag_scores = model(inputs)
            tag_predict = prediction(tag_scores)
    #         print(tag_predict)

            reference = find_token(element)
            candidate = find_token((element[0],tag_predict))

            inter = [c for c in candidate if c in reference]
            if len(candidate) !=0:
                R = len(inter) / len(reference)
                P = len(inter) / len(candidate)
            else: error_sentence_SL[language].append((element,tag_predict))
            if (R+P)  != 0 : 
                F1 = 2 * R*P / (R+P)
            else: 
                error_sentence_SL[language].append((element,tag_predict))
                F1=0
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_SL[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

import pickle
filename = 'BiLSTM_SL'
with open('./results/%s.pickle'%filename, 'wb') as f:
    pickle.dump(Result_SL, f)
#%%
import pandas as pd
results = pd.DataFrame.from_dict(Result_SL, orient='index')
results.columns = ['Recall','Precision','F1 score']
results

# %%
print(error_sentence_ML)
print(error_sentence_SL)






#%%
import numpy as np
sentence_str = '金钱买不到幸福。'

with torch.no_grad():
    inputs = prepare_sequence(sentence_str ,letter_to_ix)
    tag_scores = model(inputs)
    pred = prediction(tag_scores)
    print(pred)
    print(find_token((sentence_str,pred)))
len(letter_to_ix)