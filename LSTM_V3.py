#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# 
# Using contextual string embedding

# In[74]:

### Load Prepared Datasets

import pickle
language = 'CHINESE'
with open('./data/%s_Train.pickle'%language, 'rb') as f1:
     data_train = pickle.load(f1)
with open('./data/%s_Test.pickle'%language, 'rb') as f2:
     data_test = pickle.load(f2)   

# In[75]:
# data_train = data_train[0:10]
# data_test = data_test[0:3]


# In[76]:
### Modeling with Prepared Datasets


# In[78]:


# Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)


# In[82]:


letter_to_ix = {}
for sent, tags in data_train+data_test:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
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


# In[79]:
# sentence = data_train[0][0]
# sentence_in = prepare_sequence(sentence, letter_to_ix)
# print(sentence)
# print(len(sentence))


# In[80]:
# character_size= len(letter_to_ix)
# embedding_dim=2048
# emb = nn.Embedding(character_size, embedding_dim)
# emb(sentence_in).shape
# emb(sentence_in).view(len(sentence), 1, -1).shape


# In[81]:
# x = emb(sentence_in).view(len(sentence), 1, -1)
# x.size(0)


# In[83]:
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


# In[84]:


def save_checkpoint(state, filename="LSTM_V3.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# In[86]:


# Hyperparameters
# input_size = 2048 (embedding_size for each character)
# sequence_length = len(sentence)
# num_layers = 1
# hidden_size = 256
# num_classes = 5
# learning_rate = 0.1

num_layers = 1
EMBEDDING_DIM = 2048
HIDDEN_DIM = 256 # HIDDEN_DIM = 6
load_model = False
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

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# In[55]:


# # no training 
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], letter_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
# Try load checkpoint

if load_model: load_checkpoint(torch.load("LSTM_V3.pth.tar"), model, optimizer)

from tqdm import tqdm
import time

for epoch in tqdm(range(EPOCH)): 
    start_time = time.time()
    save_checkpoint(checkpoint)
    running_loss = 0
    for sentence, tags in tqdm(data_train,position=0,leave = True):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, letter_to_ix)
        sentence_in = prepare_sequence(sentence, letter_to_ix).to(device=device)
        targets = prepare_sequence(tags, tag_to_ix).to(device=device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores,targets)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("Loss: ",running_loss/len(data_train))
    print("--- %s seconds ---" % (time.time() - start_time))
    
# See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(data_train[0][0], letter_to_ix)
#     tag_scores = model(inputs)

#     print(tag_scores)


# import numpy as np
# with torch.no_grad():
#     inputs = prepare_sequence('今天天很好，我的心情也很好',letter_to_ix)
#     tag_scores = model(inputs)
#     pred = prediction(tag_scores)
#     print(inputs)
#     print(pred)

# In[90]:
### Word Level Evaluation 


# In[97]:

# def prediction(input):
#         output = [np.argmax(i) for i in input]
#         out_list = [ix_to_tag[int(o)] for o in output]
#         out_str = ''
#         for o in out_list:
#             out_str += o 
#         return out_str

# # create token list from BIESX tag 
# def find_token(sentence_str):
#     token = []; word = ''
    
#     for  i,tag in enumerate(sentence_str[1]):
#         if tag == 'S':
#             token.append(sentence_str[0][i])
#             continue
#         if tag == 'X': 
#             continue 

#         if (tag == 'B') | (tag == 'I'): 
#             word += sentence_str[0][i] 
#             continue
#         if tag == 'E': 
#             word+=sentence_str[0][i]
#             token.append(word)
#             word=''
#     return token


# # In[93]:


# from tqdm import tqdm
# import numpy as np
# with torch.no_grad():
#     import sklearn
#     R_score,P_score,F1_score = [],[],[]
#     for element in tqdm(data_test,position=0):
        
#         inputs = prepare_sequence(element[0],letter_to_ix)
#         tag_scores = model(inputs)
#         tag_predict = prediction(tag_scores)
# #         print(tag_predict)

#         reference = find_token(element)
#         candidate = find_token((element[0],tag_predict))

#         inter = [c for c in candidate if c in reference]
#         R = len(inter) / len(reference)
#         P = len(inter) / len(candidate)
#         F1 = 2 * R*P / (R+P)
        
#         R_score.append(R); P_score.append(P);F1_score.append(F1)
        
# #         print(reference)
# #         print(candidate)
#     print('/n')
#     print(np.mean(R_score))
#     print(np.mean(P_score))
#     print(np.mean(F1_score))


# # In[95]:


# element


# # In[96]:


# find_token((element[0],tag_predict))


# # In[5]:


# with torch.no_grad():
#     element = data_test[1]
#     inputs = prepare_sequence(element[0],letter_to_ix)
#     tag_scores = model(inputs)
#     tag_predict = prediction(tag_scores)
#     print(tag_predict)
    
#     reference = find_token(element)
#     candidate = find_token((element[0],tag_predict))
    
#     inter = [c for c in candidate if c in reference]
#     R = len(inter) / len(reference)
#     P = len(inter) / len(candidate)
#     F1 = 2 * R*P / (R+P)
#     print(reference)
#     print(candidate)
#     print(R)
#     print(P)
#     print(F1)
