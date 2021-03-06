#!/usr/bin/env python
# coding: utf-8

# # LSTM for Word Boundaries
# train model for each cluster 

from functions import *

data_train,data_test,data_dev={},{},{}
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
        dev = pickle.load(f3) 

    data_train[language] = train; data_test[language]  = test ; data_dev[language] = dev

for n,g in enumerate(GroupList):
    data_train['group%s'%str(n+1)]=[] 
    data_dev['group%s'%str(n+1)]=[]
    for language in g:
        data_train['group%s'%str(n+1)] += data_train[language]
        data_dev['group%s'%str(n+1)] += data_dev[language]
        # small sample for testing and debugging
        # data_train['group%s'%str(n+1)] += data_train[language][0:10]
        # data_dev['group%s'%str(n+1)] += data_dev[language][0:10]


#%% Hyperparameters - redefine hyperparameters if not insistant as in functions.py


# %% Train Model
from tqdm import tqdm;import time

for groupname in GroupNameList:
    # initialize network
    model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size)    if(torch.cuda.is_available()): print(torch.cuda.current_device())
    model = model.to(device); model.train()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.NLLLoss()
    
    for epoch in tqdm(range(MAX_EPOCH)): 
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
    
# for batch_size >1 
if batch_size > 1:
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=data_dev, batch_size=batch_size, shuffle=True)