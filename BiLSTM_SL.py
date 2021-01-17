#!/usr/bin/env python
# coding: utf-8

# # BiLSTM for Word Boundaries
# train model for each language

#%% Load Prepared Datasets

# import os 
# os.chdir('/Users/wuqi/MasterThesis_Tokenization')

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
data_train,data_test={},{}
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 

    data_train[language] = train
    data_test[language]  = test

    # # small sample for testing and debugging
    # data_train[language] = train[0:10]
    # data_test[language]  = test [0:5]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# %% character dictionary set and define other helper functions
import numpy as np
letter_to_ix = {}
for language in LanguageList:
    for sent, tags in data_train[language]+data_test[language]:
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

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# %% Hyperparameters
%run functions.py 
from functions import LSTMTagger

num_classes = 5
embedding_dim = 2048 # note: maybe try smaller size?
hidden_dim = 256 # hidden_size = 256
EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1
character_size = len(letter_to_ix)
# not using this since batch size is 1
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# %% Train Model

# For continusly training 
# load_model = True
# filename = ''
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time

for language in LanguageList:
    # Initialize network
    model = LSTMTagger(embedding_dim=embedding_dim, num_layers = num_layers,hidden_dim=hidden_dim,tagset_size=num_classes,character_size=character_size)
    if(torch.cuda.is_available()): print(torch.cuda.current_device())
    model = model.to(device); model.train()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.NLLLoss()

    for epoch in tqdm(range(EPOCH)): 
        start_time = time.time()
        running_loss = 0
        for sentence, tags in tqdm(data_train[language],position=0,leave = True):
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
        save_checkpoint(checkpoint,filename="./trained_models/BiLSTM_SL_%s.pth.tar"%language)

        print("Loss: ",running_loss/len(data_train[language]))
        print("--- %s seconds ---" % (time.time() - start_time))
