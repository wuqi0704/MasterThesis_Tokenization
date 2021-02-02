#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# train multilingual model using all languages as training set
# Mini batch training 
# %% Load Prepared Datasets

from bilstm import *
# small sample for debugging
# data_train = data_train[0:100]
# data_test = data_test [0:100]
# data_dev = data_dev[0:100]
#%% Hyperparameters - redefine hyperparameters if not insistant as in functions.py

batch_size = 10
MAX_EPOCH = 30

# mini_batch of datasets
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

filename = "./trained_models/BiLSTM_ML256.tar"
model, optimizer,loss_function,checkpoint = initialize_model(batch_size = 10)


# %% Train Model
filename = "./trained_models/Batch_BiLSTM_ML_E30.tar"
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


