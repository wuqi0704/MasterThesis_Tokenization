#!/usr/bin/env python
# coding: utf-8
#%%

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
g1 = ['HEBREW','ARABIC']
g2 = ['PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH','FINNISH']
g3 = ['RUSSIAN', 'KOREAN']
g4 = ['CHINESE','JAPANESE']
g5 = ['VIETNAMESE']
GroupList = [g1,g2,g3,g4,g5]
GroupNameList = ['group%s'%str(i) for i in range(1,6)]

# character dictionary set and define other helper functions

import pickle
data_train,data_test,data_dev=[],[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
        dev = pickle.load(f3)
    
    data_train += train; data_test += test; data_dev += dev

import numpy as np
letter_to_ix = {}
letter_to_ix[''] = 0 # need this for padding
for sent, tags in data_train+data_test+data_dev:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ',len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4} 
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prediction_str(input):
        output = [np.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list:
            out_str += o 
        return out_str

# create token list from BIESX tag 
def find_token(sentence_str):
    token = []; word = ''
    
    for  i,tag in enumerate(sentence_str[1]):
        if tag == 'S':
            token.append(sentence_str[0][i])
            continue
        if tag == 'X': 
            continue 
        if (tag == 'B') | (tag == 'I'): 
            word += sentence_str[0][i] 
            continue
        if tag == 'E': 
            word+=sentence_str[0][i]
            token.append(word)
            word=''
    return token

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

from torch.nn.utils.rnn import pad_sequence
def prepare_batch(batch, to_ix):
    tensor_list = []
    for seq in batch:
        idxs = [to_ix[w] for w in seq]
        tensor = torch.tensor(idxs, dtype=torch.long)
        tensor_list.append(tensor)
    return pad_sequence(tensor_list,batch_first=False)
    # with batch_first=False, the dimension come as (len(seq)#length of longest sequence,len(batch)#batch_size)

def prepare_cse(sentence,batch_size=1):
    lm_f: LanguageModel = FlairEmbeddings('multi-forward').lm
    lm_b: LanguageModel = FlairEmbeddings('multi-backward').lm 
    if batch_size == 1:
        embeds_f = lm_f.get_representation([sentence],'\n','\n')[1:-1,:,:]
        embeds_b = lm_b.get_representation([sentence],'\n','\n')[1:-1,:,:]
    elif batch_size >1:
        embeds_f = lm_f.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
        embeds_b = lm_b.get_representation(list(sentence),'\n','\n')[1:-1,:,:]
    return torch.cat((embeds_f,embeds_b),dim=2)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

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


class LSTMTagger(nn.Module):

    def __init__(self, 
                 character_size, 
                 embedding_dim, 
                 hidden_dim,
                 num_layers,
                 tagset_size,
                 batch_size,
                 use_CSE = False,
    ) :
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.use_CSE = use_CSE

        self.character_embeddings = nn.Embedding(character_size, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self,sentence):
        if self.batch_size > 1:
            if self.use_CSE == True:
                embeds = sentence
            elif self.use_CSE == False:
                embeds = self.character_embeddings(sentence) 

            x = embeds # #.view(len(embeds), self.batch_size, -1)
            h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(device)
            out, _ = self.lstm(x, (h0, c0))
            # tag_space = self.hidden2tag(out.view(len(sentence),self.batch_size, -1))
            tag_space = self.hidden2tag(out)
            tag_scores = F.log_softmax(tag_space, dim=2) # dim = (len(sentence),batch,len(tag))
        
        elif self.batch_size == 1:   
            if self.use_CSE == True:
                embeds = sentence
            elif self.use_CSE == False:
                embeds = self.character_embeddings(sentence) 

            x = embeds.view(len(sentence), 1, -1) # add one more dimension, batch_size = 1 
            out, _ = self.lstm(x)
            tag_space = self.hidden2tag(out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores




#%%
class BiLSTM_CRF(nn.Module):

    def __init__(self, 
                 character_size, 
                 tag_to_ix, 
                 embedding_dim, 
                 hidden_dim):

        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.character_size = character_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(character_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
#%%
# Initialize network
tagset_size = len(tag_to_ix)
embedding_dim = 256
hidden_dim = 256 
MAX_EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1
character_size = len(letter_to_ix)
shuffle = True
batch_first = False 

model = LSTMTagger(character_size,embedding_dim,hidden_dim, num_layers,tagset_size,batch_size)

if(torch.cuda.is_available()):
    print(torch.cuda.current_device())
model = model.to(device)
model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()
checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}


#%%
# out, _ = lstm(x, (h0, c0))
# out.shape