#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# 
# Using contextual string embedding

# In[74]:
# train model for each cluster 

### Load Prepared Datasets
import pickle
g1 = ['HEBREW','ARABIC']
g2 = ['PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH','FINNISH']
g3 = ['RUSSIAN', 'KOREAN']
g4 = ['CHINESE','JAPANESE']
g5 = ['VIETNAMESE']

GroupList = [g1,g2,g3,g4,g5]

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
data_train,data_test={},{}
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 

        data_train[language] = train
        data_test[language]  = test 
    
# manually delete datasets that has a mismatch of the tag vs sentence length
# note: effective way to deal with the data , the error is inside WhiteSpace_After.Or sth else
data_train_correct,data_test_correct = {},{}

for language in LanguageList:
    data_train_correct[language], data_test_correct[language]=[],[]
    for sentence,tags in data_train[language]:
        if len(sentence) == len(tags):
            data_train_correct[language].append((sentence,tags))
    lost = len(data_train[language])-len(data_train_correct[language])
    if lost !=0: print('train',language,lost)

    for sentence,tags in data_test[language]:
        if len(sentence) == len(tags):
            data_test_correct[language].append((sentence,tags))
    lost = len(data_test[language])-len(data_test_correct[language])
    if lost !=0: print('test',language,lost)
# update the datasets for training that has matching length of sentence and targets 
data_test,data_train = data_test_correct,data_train_correct


# Import packages
import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim 
import torch.nn.functional as F 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


# In[82]:
# find character dictionary with training data
import numpy as np
letter_to_ix = {}
for language in LanguageList:
    for sent, tags in data_train[language]+data_test[language]:
        for letter in sent:
            if letter not in letter_to_ix:
                letter_to_ix[letter] = len(letter_to_ix)
print('Nr. of distinguish character: ',len(letter_to_ix.keys()))

# define tagging dictionary
tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}
# define prediction function
def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
# define function to transfer text sequence input to numeric imput using character dictionary     
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


for n,g in enumerate(GroupList):
    data_train['group%s'%str(n+1)]=[] # only training data need to be grouped 
    for language in g:
        data_train['group%s'%str(n+1)] += data_train[language]
        # small sample for testing and debugging
        # data_train['group%s'%str(n+1)] += data_train[language][0:10]

# In[83]:
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

class LSTMTagger(nn.Module):
    def __init__(self, character_size, embedding_dim, hidden_dim,num_layers,tagset_size) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.character_embeddings = nn.Embedding(character_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False, bidirectional=True,dropout=0.5)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self,sentence):
        embeds = self.character_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1) # add one more dimension, batch_size = 1 
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device) # batch size is 1 
        # out, _ = self.lstm(x, (h0, c0))
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# %%
def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Hyperparameters
num_classes = 5
num_layers = 1 # if more 
EMBEDDING_DIM = 2048 
HIDDEN_DIM = 256 
load_model = False
EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1

#%%
GroupNameList = list(data_train.keys())[-5:]

# %%
from tqdm import tqdm;import time

for groupname in GroupNameList:
    # initialize network
    model = LSTMTagger(embedding_dim=EMBEDDING_DIM, num_layers = num_layers,hidden_dim=HIDDEN_DIM,tagset_size=num_classes,character_size=len(letter_to_ix))
    if(torch.cuda.is_available()): print(torch.cuda.current_device())
    model = model.to(device); model.train()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.NLLLoss()
    
    for epoch in tqdm(range(EPOCH)): 
        start_time = time.time()
        running_loss = 0
        for sentence, tags in tqdm(data_train[groupname],position=0,leave = True):
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

        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint,filename = "./trained_models/BiLSTM_GL_%s.pth.tar"%groupname)

        print("Loss: ",running_loss/len(data_train[groupname]))
        print("--- %s seconds ---" % (time.time() - start_time))
    