#!/usr/bin/env python
# coding: utf-8

# # BiLSTM for Word Boundaries
# train model for each language

from functions import *
# # small sample for testing and debugging
# for language in LanguageList:
    # data_train[language] = train[0:10]
    # data_test[language]  = test [0:5]

# %% Train Model

from tqdm import tqdm; import time

for language in LanguageList:
    # Initialize network
    model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size)

    if(torch.cuda.is_available()):
        print(torch.cuda.current_device())
    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_function = nn.NLLLoss()

    for epoch in tqdm(range(EPOCH)): 
        start_time = time.time()
        running_loss = 0
        for sentence, tags in tqdm(data_train[language],position=0,leave = True):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, letter_to_ix).to(device=device)
            targets = prepare_sequence(tags, tag_to_ix).to(device=device)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores,targets)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint,filename="./trained_models/BiLSTM_SL_%s.pth.tar"%language)

        print("Loss: ",running_loss/len(data_train[language]))
        print("--- %s seconds ---" % (time.time() - start_time))
