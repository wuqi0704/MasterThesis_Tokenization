# %% BiLSTM model ML
from flair.data import Corpus
from flair.datasets import SentenceDataset
from tokenizer_model import FlairTokenizer
from tokenizer_model import LabeledString

import pickle

language = 'ENGLISH'

# 1. load your data and convert to list of LabeledString
# data_train, data_test, data_dev = [], [], []
# with open('resources/%s_Train.pickle' % language, 'rb') as f1:
#     train = pickle.load(f1)
# with open('resources/%s_Test.pickle' % language, 'rb') as f2:
#     test = pickle.load(f2)
# with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
#     dev = pickle.load(f3)
#
# data_train.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train])
# data_test.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test])
# data_dev.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev])

# 2. make a Corpus object
# corpus: Corpus = Corpus(SentenceDataset(data_train), SentenceDataset(data_test), SentenceDataset(data_dev))

# print(corpus)

tokenizer: FlairTokenizer = FlairTokenizer.load("resources/taggers/tokenizer_english_200-32/best-model.pt")
print(tokenizer.tag_to_ix)

string = LabeledString('This is great').set_label('tokenization', 'biieXbeXbiiie'.upper())
string2 = LabeledString('This is great').set_label('tokenization', 'biieXbeXbiiie'.upper())
tokenizer.evaluate([string, string2], mini_batch_size=1)

print(string)
print(string2)

