#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# train multilingual model using all languages as training set

# small sample for debugging
# data_train = data_train[0:10]
# data_test = data_test [0:10]

# %run functions.py
# filename = "./trained_models/BiLSTM_ML.tar"
filename = "./trained_models/BiLSTM_ML256.tar"
# For continusly training 
# load_model = True
# if load_model: load_checkpoint(torch.load(filename), model, optimizer)

from tqdm import tqdm; import time

for epoch in tqdm(range(MAX_EPOCH)): 
    start_time = time.time()
    running_loss = 0
    for sentence, tags in data_train:
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

