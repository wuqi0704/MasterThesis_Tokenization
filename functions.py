#!/usr/bin/env python
# coding: utf-8

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
        # h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        # c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device) # batch size is 1 
        # out, _ = self.lstm(x, (h0, c0))
        lstm_out, _ = self.lstm(x)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores