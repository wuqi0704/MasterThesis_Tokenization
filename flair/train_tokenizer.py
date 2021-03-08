#%% BiLSTM model ML 
from flair.data import Corpus

from flair.datasets import SentenceDataset
from flair.embeddings import token
from tokenizer_model import FlairTokenizer
from tokenizer_model import LabeledString


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
import pickle

# 1. load your data and convert to list of LabeledString
data_train, data_test, data_dev = [], [], []
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train])
    data_test.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test])
    data_dev.extend([LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev])


#%%
# 2. make a Corpus object
corpus: Corpus = Corpus(SentenceDataset(data_train), SentenceDataset(data_test), SentenceDataset(data_dev))
# corpus = corpus.downsample(0.01)
# 3. make the letter dictionary from the corpus
letter_to_ix = {}
letter_to_ix[''] = 0  # need this for padding

for sentence in corpus.get_all_sentences():
    for letter in sentence.string:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('functions.py : Nr. of distinguish character: ', len(letter_to_ix.keys()))
#%%
# 4. initialize tokenizer
for HIDDEN_DIM in [32,64,128,256,512]:
    tokenizer: FlairTokenizer = FlairTokenizer(
        letter_to_ix=letter_to_ix,
        embedding_dim=256,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        use_CSE=False,
        use_CRF=False,
    )

    # 5. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tokenizer, corpus)

    # 6. train
    trainer.train(
        "resources/taggers/1_h%s"%HIDDEN_DIM,
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=30,
    )