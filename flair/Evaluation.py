#%% initialize model and load model

# from tokenizer_model import FlairTokenizer
# tokenizer: FlairTokenizer = FlairTokenizer()
# 1. load your data and convert to list of LabeledString
from flair.data import Corpus

from flair.datasets import SentenceDataset
from flair.embeddings import token
from tokenizer_model import FlairTokenizer
from tokenizer_model import LabeledString
import pickle 

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
data_test,data_dev = {},{}
for language in LanguageList:
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_test[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test]
    data_dev[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev]

#%%
import torch 
state = torch.load('./resources/taggers/MLbilstm256/best-model.pt',map_location=torch.device('cpu'))
from tokenizer_model import FlairTokenizer
tokenizer: FlairTokenizer = FlairTokenizer() 
model = tokenizer._init_model_with_state_dict(state)
#%%
from tqdm import tqdm
for language in tqdm(LanguageList[8:]):
    result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
    print(language,result.detailed_results)
#%%
language = 'JAPANESE'
model.evaluate(data_test[language],mini_batch_size=1)
# %%
LanguageList[8:]
# %%
