#!/usr/bin/env python
# coding: utf-8
#%%

language = 'CHINESE'
import pickle
with open('./data/%s_Train.pickle'%language, 'rb') as f1:
    data_train = pickle.load(f1)
with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
    data_dev = pickle.load(f3)
#%%
# %% Train Model
use_CSE = False 
batch_size = 1
embedding_dim = 4096 # because using CSE 
hidden_dim = 256 

# train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=shuffle)
# dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=shuffle)

# Initialize network
filename = "./trained_models/BiLSTMCRF_CN.tar"

START_TAG = "<START>"; STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "I": 1, "E": 2,'S':3, 'X':4, START_TAG: 5, STOP_TAG: 6}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}
model = BiLSTM_CRF(len(letter_to_ix), tag_to_ix, embedding_dim, hidden_dim)
model = model.to(device); model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

from tqdm import tqdm 
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(MAX_EPOCH):  
    for sentence, tags in tqdm(data_train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, letter_to_ix).to(device=device)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device=device)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # compute development set loss 
    dev_loss = 0
    for sentence, tags in tqdm(data_dev):
        sentence_in = prepare_sequence(sentence,letter_to_ix).to(device=device)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).to(device=device)

        loss = model.neg_log_likelihood(sentence_in, targets)
        # tag_scores = model(batch_in)
        dev_loss += loss.item()

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
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint,filename)

#%%
# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(data_train[0][0], letter_to_ix)
    print(model(precheck_sent))

#%%
output = model(precheck_sent)[1]
output

[ix_to_tag[int(o)] for o in output]


#%%
tag_to_ix = {"B": 0, "I": 1, "E": 2,'S':3, 'X':4, START_TAG: 5, STOP_TAG: 6}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

#%%
def prediction_str(output):
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list:
            out_str += o 
        return out_str
#%%
prediction_str(output)
#%%
data_train[0][1]