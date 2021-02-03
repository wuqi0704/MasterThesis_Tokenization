#!/usr/bin/env python
# coding: utf-8
#%%
from bilstm import *

#%%
language = 'CHINESE'
import pickle
with open('./data/%s_Train.pickle'%language, 'rb') as f1:
    data_train = pickle.load(f1)
with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
    data_dev = pickle.load(f3)

# %% Train Model
use_CSE = True
embedding_dim = 4096 # because using CSE 
batch_size = 10

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

# Initialize network
model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size,use_CSE=use_CSE)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()
filename = "./trained_models/BiLSTM_CSE_CN.tar"

# For continusly training 
# load_model = True
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time
from torch.nn.utils.rnn import pack_padded_sequence

for epoch in tqdm(range(MAX_EPOCH)): 
    
    start_time = time.time()
    running_loss = 0
    for batch_idx, (data, tags) in tqdm(enumerate(train_loader),position=0,leave=True):
        # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
        model.zero_grad()
        
        # Step 2. Get our inputs ready for the network
        if use_CSE == True: 
            batch_in = data
            # batch_in = prepare_cse(data,batch_size=batch_size).to(device=device) # shape len(longest sentence),batch_size,embedding_dim=4096
        elif use_CSE == False : 
            batch_in = prepare_batch(data,letter_to_ix).to(device=device)# shape len(longest sentence),batch_size
        
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
        if use_CSE == True:
            batch_in = prepare_cse(data,batch_size=batch_size).to(device=device)
        elif use_CSE == False : 
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
        if (dev_loss/len(dev_loader)) < lowest_dev_loss : 
            lowest_dev_loss = dev_loss/len(dev_loader)
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint,filename)
        else:
            print('stop early, epoch = ',epoch)
            break

    print("training Loss: ",running_loss/len(train_loader))
    print("develop Loss: ",dev_loss/len(dev_loader))
    print("--- %s seconds ---" % (time.time() - start_time))

#%%

# sentence = data
# embeds = prepare_cse(sentence,batch_size=batch_size)
# embeds = embeds.to(device=device)
# character_embeddings = nn.Embedding(character_size, embedding_dim)

# batch_in = prepare_batch(sentence,letter_to_ix)

# # %%
# lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)

# #%%
# x = embeds
# h0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)
# c0 = torch.zeros(num_layers * 2, batch_size, hidden_dim).to(device)
# out, _ = lstm(x, (h0, c0))
# %%
# embeds.shape

#%% without CSE
use_CSE = True
embedding_dim = 4096 # because using CSE 
batch_size = 10

train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

# Initialize network
model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size,use_CSE=use_CSE)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()
filename = "./trained_models/BiLSTM4096_CN.tar"

# For continusly training 
# load_model = True
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time
from torch.nn.utils.rnn import pack_padded_sequence

for epoch in tqdm(range(MAX_EPOCH)): 
    
    start_time = time.time()
    running_loss = 0
    for batch_idx, (data, tags) in tqdm(enumerate(train_loader),position=0,leave=True):
        # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
        model.zero_grad()
        
        # Step 2. Get our inputs ready for the network
        if use_CSE == True: 
            batch_in = prepare_cse(data,batch_size=batch_size).to(device=device) # shape len(longest sentence),batch_size,embedding_dim=4096
        elif use_CSE == False : 
            batch_in = prepare_batch(data,letter_to_ix).to(device=device)# shape len(longest sentence),batch_size
        
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
        if use_CSE == True:
            batch_in = prepare_cse(data,batch_size=batch_size).to(device=device)
        elif use_CSE == False : 
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
        if (dev_loss/len(dev_loader)) < lowest_dev_loss : 
            lowest_dev_loss = dev_loss/len(dev_loader)
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint,filename)
        else:
            print('stop early, epoch = ',epoch)
            break

    print("training Loss: ",running_loss/len(train_loader))
    print("develop Loss: ",dev_loss/len(dev_loader))
    print("--- %s seconds ---" % (time.time() - start_time))
