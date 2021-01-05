#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# 
# Using contextual string embedding

# In[74]:

### Load Prepared Datasets

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
data_train,data_test=[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    
    data_train += train
    data_test += test 
    
# manually delete datasets that has a mismatch of the tag vs sentence length
# note: effective way to deal with the data , the error is inside WhiteSpace_After.Or sth else
data_train_correct,data_test_correct = [],[]
for sentence,tags in data_train:
    if len(sentence) == len(tags):
        data_train_correct.append((sentence,tags))
print(len(data_train)-len(data_train_correct))

for sentence,tags in data_test:
    if len(sentence) == len(tags):
        data_test_correct.append((sentence,tags))
print(len(data_test)-len(data_test_correct))

data_test,data_train = data_test_correct,data_train_correct

# data_train = data_train[0:10]
# data_test = data_test [0:10]

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
import numpy as np
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=True,dropout=0.5)
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

filename = "LSTM_V3_multilanguage.pth.tar"
def save_checkpoint(state, filename=filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# In[86]:


# Hyperparameters

# sequence_length = len(sentence)
# num_classes = 5
num_layers = 1 # if more 
EMBEDDING_DIM = 2048 # input_size = 2048 (embedding_size for each character)
HIDDEN_DIM = 256 # hidden_size = 256
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
# not using this since batch size is 1
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# In[55]:

checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
# Try load checkpoint

if load_model: load_checkpoint(torch.load(filename), model, optimizer)

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
    





# %%
# print(sentence)
# sentence_in = prepare_sequence(sentence, letter_to_ix)
# print(sentence_in.shape)
# model(sentence_in)
# len(targets)
# print(len(sentence))
# len(tags)

# print(len(data_train[11721][0]))
# print(len(data_train[11721][1]))

# In[74]:

### Load Prepared Datasets

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
data_train,data_test=[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    
    data_train += train
    data_test += test 
    
# manually delete datasets that has a mismatch of the tag vs sentence length
# note: effective way to deal with the data , the error is inside WhiteSpace_After.Or sth else
# print(len(data_train))

# k=0
# for sentence,tags in data_train:
#     if len(sentence) != len(tags):
#         data_train.remove((sentence,tags))
#         k += 1 # when an element is deleted, index need to change
# print(k)
# print(len(data_train))

# # for i,(sentence,tags) in enumerate(data_train):
# #     if len(sentence) != len(tags):
# #         del data_train[i]

# for i,(sentence,tags) in enumerate(data_train):
#     if len(sentence) != len(tags):
#         print(i)

