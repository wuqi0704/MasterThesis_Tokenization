
#%%
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel

# a detail: before and after the text we always append a newline since the language model was trained this way
start_marker = '\n'
end_marker = '\n'

# load the language model (for instance multi-forward)
language_model: LanguageModel = FlairEmbeddings('multi-forward').lm
print(f'This language model has {language_model.hidden_size} hidden states')

# comment this in to print all characters in language model dictionary
#for vocabulary_item_id in range(len(language_model.dictionary)):
#    print(language_model.dictionary.get_item_for_index(vocabulary_item_id))

# example string you want to tokenize
text = "It's not Mr. Smith."

print(f'The text has {len(text)} characters (+2 for start and end marker)')

# the method always wants a list of sentences, even if you only have one sentence you must make it a list
sentence_list = [text]

# pass text through language model - this returns a tensor of size (#characters, #sentences, #hidden states)
output_states = language_model.get_representation(sentence_list, start_marker, end_marker)

# print the size of the output tensor
print(output_states.size())

# to get the embedding for character at position i, do:
i = 3
print(f"Character at position {i} is {text[i]}")
print(f"Its embedding is: {output_states[i+len(start_marker)]}")
# %%
from flair.data import Corpus

from flair.datasets import SentenceDataset
from flair.embeddings import token
from tokenizer_model import FlairTokenizer
from tokenizer_model import LabeledString
import torch 
import pickle
import flair
import numpy as np
import torch.optim as optim

language = 'ENGLISH'

# 1. load your data and convert to list of LabeledString
data_train, data_test, data_dev = [],[],[]

with open('resources/%s_Train.pickle' % language, 'rb') as f1:
    train = pickle.load(f1)
with open('resources/%s_Test.pickle' % language, 'rb') as f2:
    test = pickle.load(f2)
with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
    dev = pickle.load(f3)

data_train = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train]
data_test = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test]
data_dev = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev]

corpus: Corpus = Corpus(SentenceDataset(data_train), SentenceDataset(data_test), SentenceDataset(data_dev))
letter_to_ix = {}
letter_to_ix[''] = 0  # need this for padding

for sentence in corpus.get_all_sentences():
    for letter in sentence.string:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ', len(letter_to_ix.keys()))

# small sample for debugging
data_train = data_train[0:3]
data_test = data_test [0:3]
data_dev = data_dev [0:3]
#%%
# initialize tokenizer
import torch.nn.functional as F

tokenizer: FlairTokenizer = FlairTokenizer(
    letter_to_ix=letter_to_ix,
    embedding_dim=4096,
    hidden_dim=128,
    num_layers=1,
    use_CSE=True,
    use_CRF=False,
)

optimizer: torch.optim.Optimizer = optim.SGD(tokenizer.parameters(), lr=0.1)

batch_size = 1 
from tqdm import tqdm; import time

start_time = time.time()
running_loss = 0
# for sentence, tags in tqdm(data_train):
for E in range(2): # number of epochs
    for data in tqdm(data_train):
        sentence = [data.string]
        tags = [str(data.get_labels()[0])[:-6]]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        tokenizer.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        embeds = tokenizer.prepare_cse(sentence, letter_to_ix)
        print(embeds)
        targets = tokenizer.prepare_batch(tags, tokenizer.tag_to_ix)
        batch_input_tags = tokenizer.prepare_batch(tags, tokenizer.tag_to_ix)


        # Step 3. Run our forward pass.
        # tag_scores = (embeds)
        out, _ = tokenizer.lstm(embeds)
        tag_space = tokenizer.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=2)  # dim = (len(data_points),batch,len(tag))
        length_list = [len(tags[0])]

        packed_tag_space = torch.tensor([], dtype=torch.long, device=flair.device)
        packed_tag_scores = torch.tensor([], dtype=torch.long, device=flair.device)
        packed_batch_input_tags = torch.tensor([], dtype=torch.long, device=flair.device)

        for i in np.arange(batch_size):
            packed_tag_scores = torch.cat((packed_tag_scores, tag_scores[:length_list[i], i, :]))
            packed_tag_space = torch.cat((packed_tag_space, tag_space[:length_list[i], i, :]))
            packed_batch_input_tags = torch.cat((packed_batch_input_tags, batch_input_tags[:length_list[i], i]))

        # loss = tokenizer.loss_function(tag_scores,targets)
        loss = tokenizer.loss_function(packed_tag_scores, packed_batch_input_tags)
        # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

#%%
tokenizer.parameters()
# %%
