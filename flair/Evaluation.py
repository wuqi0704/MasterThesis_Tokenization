#%% initialize model and load model

# from tokenizer_model import FlairTokenizer
# tokenizer: FlairTokenizer = FlairTokenizer()
# 1. load your data and convert to list of LabeledString
import sys
sys.path.insert(1, '/Users/qier/opt/anaconda3/lib/python3.7/site-packages/')

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
# and cause debug for dimension problem.
for language in LanguageList:
    for item in data_test[language]:
        if len(item.string)==1:
            data_test[language].remove(item)

import torch 
model_name = '1_h512'
state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
from tokenizer_model import FlairTokenizer
tokenizer = FlairTokenizer() 
model = tokenizer._init_model_with_state_dict(state)
#%%
# from tqdm import tqdm
# import pandas as pd
# output = {}
# for language in tqdm(LanguageList):
#     result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
#     obj = result.detailed_results
#     output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

# out_dataframe = pd.DataFrame.from_dict(output, orient='index')
# out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
# out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/%s.csv'%model_name)

# #%%
# language = 'CHINESE'
# result = model.evaluate(data_test[language],mini_batch_size=1)
# result[0].detailed_results
# #%%
# # %%
# test_sentence = LabeledString('使館工作人員很快就用滅火器撲滅了建築外牆前的火。')
# test_sentence.set_label('tokenization','BEBEBEBESSBIEBESBEBESSSS')
# loss,packed_sent,packed_tags,tag_predict = model.forward_loss(test_sentence,foreval=True)
# model.find_token((packed_sent, tag_predict))

# #%%
# test_sentence = LabeledString('Pack Mr.Smiths belongings and take them to his new home.')
# test_sentence.set_label('tokenization','BIIEXBIIIIIIIEXBIIIIIIIIEXBIEXBIIEXBIIEXBEXBIEXBIEXBIIES')
# loss,packed_sent,packed_tags,tag_predict = model.forward_loss(test_sentence,foreval=True)
# model.find_token((packed_sent, tag_predict))
# # %%
# model.evaluate(test_sentence)[0].detailed_results

#%% # draw perfermance graph 
# s = pd.DataFrame()
# for HIDDEN_DIM in ['32','64','128','256','512']:
#     a = (pd.read_csv("/Users/qier/MasterThesis_Tokenization/results/1_h%s.csv"%HIDDEN_DIM))
#     a['hidden_dim'] = HIDDEN_DIM
#     s = pd.concat([s,a],axis=0)
# #%%
# s.groupby(['Unnamed: 0']).mean()
# # %%
# s = s.rename(columns={'Unnamed: 0':'language'})
# import matplotlib.pyplot as plt
# plt.figure(figsize = [10,10])
# for language in LanguageList:
#     c = s[s['language']==language]
#     plt.plot(c.hidden_dim,c['F1-score'])
# %% Debug evaluation errors : 

test_results, test_loss = model.evaluate(
    # self.corpus.test,
    data_test['CHINESE'][0:2],
    mini_batch_size=2,
    num_workers=8,
)

test_results, test_loss = model.evaluate(
    # self.corpus.test,
    data_test['CHINESE'][0:2],
    mini_batch_size=1,
    num_workers=8,
)
print(test_results.detailed_results)

#%%
# from flair.datasets import DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence
# import torch
# import torch.nn.functional as F
# test1 = [data_test['CHINESE'][1],data_test['CHINESE'][1]]
# test2 = data_test['CHINESE'][0:2]

# data_loader = DataLoader(test1, batch_size=2, num_workers=8)
# inter = []
# for batch in data_loader:
#     data_points = batch
#     try: 
#         sent_string,tags = [],[]
#         for sentence in data_points: 
#             sent_string.append((sentence.string))
#             tags.append(sentence.get_labels('tokenization')[0]._value)
#         batch_size=len(data_points)
#         if batch_size == 1: # if only one element, then get rid of list. 
#             sent_string = sent_string[0]
#             tags = tags[0]
#     except:
#         sent_string = data_points.string
#         tags = data_points.get_labels('tokenization')[0]._value
#         batch_size = 1

#     targets = model.prepare_batch(tags, model.tag_to_ix).squeeze()

#     embeds = model.prepare_batch(sent_string, model.letter_to_ix)
#     embeds = model.character_embeddings(embeds)
        
#     h0 = torch.zeros(model.num_layers * 2, embeds.shape[1], model.hidden_dim)
#     c0 = torch.zeros(model.num_layers * 2, embeds.shape[1], model.hidden_dim)
#     out, _ = model.lstm(embeds, (h0, c0))
#     tag_space = model.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
#     tag_scores = F.log_softmax(tag_space, dim=2).squeeze()  # dim = (len(data_points),batch,len(tag))
#     if (batch_size == 1):
#         packed_sent,packed_tags = sent_string,tags
#     elif (batch_size > 1) : # if the input is more than one datapoint
#         length_list = []
#         for sentence in data_points: 
#             length_list.append(len(sentence.string))
        
#         packed_sent,packed_tags = '',''
#         for sent in sent_string: packed_sent += sent 
#         for tag in tags: packed_tags += tag

#         print(self.prediction_str(tag_scores))
#         tag_scores = pack_padded_sequence(tag_scores, length_list, enforce_sorted=False).data#FIXME
#         targets = pack_padded_sequence(targets, length_list, enforce_sorted=False).data
#         tag_space = pack_padded_sequence(tag_space, length_list, enforce_sorted=False).data

#     tag_predict = model.prediction_str(tag_scores)
#     loss = model.loss_function(tag_scores, targets)

#     reference = model.find_token((packed_sent, packed_tags))
#     candidate = model.find_token((packed_sent, tag_predict))
#     inter.append( [c for c in candidate if c in reference])



