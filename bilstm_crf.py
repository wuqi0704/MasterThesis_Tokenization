#%% # langugae list, character dictionary set and other helper functions

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

def prediction(tag_seq):
        return [ix_to_tag[int(o)] for o in tag_seq]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prediction_str(tag_seq):
        out_list = [ix_to_tag[int(o)] for o in tag_seq]
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
def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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
#%% # define class BiLSTM_CRF
import torch
# import torchvision
import torch.nn as nn  
import torch.optim as optim  
# import torch.nn.functional as F  
from torch.utils.data import DataLoader
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

class BiLSTM_CRF(nn.Module):

    def __init__(self, 
                 character_size, 
                 tag_to_ix, 
                 embedding_dim, 
                 hidden_dim,
                 batch_size,
                 START_TAG = "<START>",
                 STOP_TAG = "<STOP>"
                 ):

        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.character_size = character_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.character_embeds = nn.Embedding(character_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim , 
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

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
        return (torch.randn(2, 1, self.hidden_dim ),
                torch.randn(2, 1, self.hidden_dim ))

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
        embeds = self.character_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim *2)
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
        return tag_seq

print('successfully imported...')

#%% # Initialize network BiLSTM_CRF
# # define hyper parameter 

embedding_dim = 256
hidden_dim = 256 
# learning_rate = 0.01
# num_layers = 1
batch_size = 1
# use_CSE = False

# MAX_EPOCH = 10
# shuffle = True
# batch_first = False 

START_TAG = "<START>"; STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "I": 1, "E": 2,'S':3, 'X':4, START_TAG: 5, STOP_TAG: 6}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

def initialize_model(character_size = len(letter_to_ix), 
                     tag_to_ix = tag_to_ix, 
                     embedding_dim= embedding_dim, 
                     hidden_dim= hidden_dim,
                     batch_size=batch_size):

    model = BiLSTM_CRF(character_size, tag_to_ix, embedding_dim, hidden_dim,batch_size)
    model = model.to(device); model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    return model, optimizer,checkpoint
# why is the lr so low here? compare to bilstm?
