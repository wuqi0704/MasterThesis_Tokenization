#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# train multilingual model using all languages as training set
# %% Load Prepared Datasets
import torch
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

# small sample for debugging
# data_train = data_train[0:10]
# data_test = data_test [0:10]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# %% character dictionary set and define other helper functions
import numpy as np
letter_to_ix = {}
letter_to_ix[''] = 0 # this is only needed for padding, but just for consistancy 
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


# %% Hyperparameters
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

from functions import LSTMTagger

tagset_size = len(tag_to_ix)
embedding_dim = 256 # embedding_dim = 2048 
hidden_dim = 256 
MAX_EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1
character_size = len(letter_to_ix)

# Initialize network
model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()

# %% Train Model

# filename = "./trained_models/BiLSTM_ML.tar"
filename = "./trained_models/BiLSTM_ML256.tar"
# For continusly training 
# load_model = True
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time

for epoch in tqdm(range(MAX_EPOCH)): 
    start_time = time.time()
    running_loss = 0
    for sentence, tags in tqdm(data_train,position=0,leave = True):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, letter_to_ix).to(device=device)
        targets = prepare_sequence(tags, tag_to_ix).to(device=device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores,targets)
        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    # calculate dev loss
    dev_loss = 0
    for sentence, tags in tqdm(data_dev,position=0,leave = True):
        sentence_in = prepare_sequence(sentence, letter_to_ix).to(device=device)
        targets = prepare_sequence(tags, tag_to_ix).to(device=device)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores,targets)

        dev_loss += loss.item()

    # save the best model using to dev loss 
    if epoch == 0: 
        lowest_dev_loss = dev_loss/len(data_dev)
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint,filename)
    else : 
        if (dev_loss/len(data_dev)) < lowest_dev_loss : 
            lowest_dev_loss = dev_loss/len(data_dev)
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint,filename)
        else:
            print('stop early, epoch = ',epoch)
            break

    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint,filename)

    print("Loss: ",running_loss/len(data_train))
    print("--- %s seconds ---" % (time.time() - start_time))

