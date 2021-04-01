#%% initialize model and load model

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
#additionally delete the sentences with length 1, which also make no sense in a tokenization task.
# and cause bug for dimension problem.
for language in LanguageList:
    for item in data_test[language]:
        if len(item.string)==1:
            data_test[language].remove(item)

import torch 
from tqdm import tqdm
import pandas as pd
# state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
# model_names = ['2_e64','2_e256','2_e512','2_e1024']
# for model_name in model_names:
model_name = '2_e64'
state = torch.load('/Users/qier/Downloads/ML_Tagger/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
tokenizer = FlairTokenizer() 
model = tokenizer._init_model_with_state_dict(state)
#%%
output = {}
for language in tqdm(LanguageList):
    result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
    obj = result.detailed_results
    output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

out_dataframe = pd.DataFrame.from_dict(output, orient='index')
out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/%s.csv'%model_name)

#%%
model.parameters