#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# train multilingual model using all languages as training set
# Mini batch training 
# %% Load Prepared Datasets
# %run functions.py 
from functions import LSTMTagger
# small sample for debugging
# data_train = data_train[0:100]
# data_test = data_test [0:100]
# data_dev = data_dev[0:100]

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
letter_to_ix[' '] = 0
for sent, tags in data_train+data_test+data_dev:
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


# %% Hyperparameters
tagset_size = len(tag_to_ix)
embedding_dim = 256 
hidden_dim = 256 
EPOCH = 10
learning_rate = 0.1
batch_size = 10
num_layers = 1
character_size = len(letter_to_ix)
shuffle = True

# mini_batch of datasets
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

# prepare sequence batch and padding
from torch.nn.utils.rnn import pad_sequence
def prepare_batch(batch, to_ix,pad = True):
    tensor_list = []
    for seq in batch:
        idxs = [to_ix[w] for w in seq]
        tensor = torch.tensor(idxs, dtype=torch.long)
        tensor_list.append(tensor)
    if pad == True:
        return pad_sequence(tensor_list,batch_first=False)
    else: return torch.tensor(tensor_list) # this does not work 
    # with batch_first=False, the dimension come as (len(seq)#length of longest sequence,len(batch)#batch_size)

# Initialize network

model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()

# %% Train Model
filename = "./trained_models/Batch_BiLSTM_ML.tar"
# For continusly training 
# load_model = True
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time
from torch.nn.utils.rnn import pack_padded_sequence

for epoch in tqdm(range(EPOCH)): 
    
    start_time = time.time()
    running_loss = 0
    for batch_idx, (data, tags) in tqdm(enumerate(train_loader),position=0,leave=True):
        # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
        model.zero_grad()
        
        # Step 2. Get our inputs ready for the network
        batch_in = prepare_batch(data,letter_to_ix).to(device=device)
        targets = prepare_batch(tags,tag_to_ix).to(device=device)
        
        # Step 3. Run our forward pass.
        length_list = []
        if batch_in.shape[1] != batch_size:
            continue 
        for sentence in data: 
            length_list.append(len(sentence))

        tag_scores = model(batch_in)
        tag_scores = pack_padded_sequence(tag_scores,length_list,enforce_sorted=False).data
        targets = pack_padded_sequence(targets,length_list,enforce_sorted=False).data
        loss = loss_function(tag_scores,targets) 

        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    # compute development set loss 
    dev_loss = 0
    for batch_idx, (data, tags) in tqdm(enumerate(dev_loader),position=0,leave=True):
        batch_in = prepare_batch(data,letter_to_ix).to(device=device)
        targets = prepare_batch(tags,tag_to_ix).to(device=device)
        length_list = []
        if batch_in.shape[1] != batch_size:
            continue 
        for sentence in data: 
            length_list.append(len(sentence))

        tag_scores = model(batch_in)
        tag_scores = pack_padded_sequence(tag_scores,length_list,enforce_sorted=False).data
        targets = pack_padded_sequence(targets,length_list,enforce_sorted=False).data
        loss = loss_function(tag_scores,targets) 
        dev_loss += loss.item()

    # save the best model using to dev loss 
    if epoch == 0: 
        lowest_dev_loss = dev_loss/len(dev_loader)
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint,filename)
    else : 
        if (dev_loss/len(dev_loader))<lowest_dev_loss : 
            lowest_dev_loss = dev_loss/len(dev_loader)
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint,filename)

    print("training Loss: ",running_loss/len(train_loader))
    print("develop Loss: ",dev_loss/len(dev_loader))
    print("--- %s seconds ---" % (time.time() - start_time))


# #%%
# from flair.models import SequenceTagger

# tagger: SequenceTagger = SequenceTagger(hidden_size=256,
#                                         embeddings=embeddings,
#                                         tag_dictionary=tag_dictionary,
#                                         tag_type=tag_type,
#                                         use_crf=False)

# from flair.trainers import ModelTrainer

# trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# trainer.train('resources/taggers/BiLSTM_ML',
#               learning_rate=0.1,
#               mini_batch_size=32,
#               max_epochs=10)

