#%% initialize model and load model

# from tokenizer_model import FlairTokenizer
# tokenizer: FlairTokenizer = FlairTokenizer()
# 1. load your data and convert to list of LabeledString
# import sys
# sys.path.insert(1, '/Users/qier/opt/anaconda3/lib/python3.7/site-packages/')

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
# model_name = '1_h512'
# state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
model_name = '2_e64'
state = torch.load('/Users/qier/Downloads/ML_Tagger/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
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
import pandas as pd
s = pd.DataFrame()
for HIDDEN_DIM in ['32','64','128','256','512']:
    a = (pd.read_csv("/Users/qier/MasterThesis_Tokenization/results/1_h%s.csv"%HIDDEN_DIM))
    a['hidden_dim'] = HIDDEN_DIM
    s = pd.concat([s,a],axis=0)
# s.groupby(['Unnamed: 0']).mean()
s
# %%
s = s.rename(columns={'Unnamed: 0':'language'})
import matplotlib.pyplot as plt
g1 = s[s['F1-score']>0.95]
g2 = s[s['F1-score']<=0.95]
f = plt.figure(figsize = [12,5])
ax = f.add_subplot(121)
for language in g1.language.unique():
    c = g1[g1['language']==language]
    ax.plot(c.hidden_dim,c['F1-score'],label = language)
plt.legend(loc = 'lower right',prop={'size': 8})
plt.title('group1: F1-score > 0.95')
ax.set_xlabel('hidden dimension')
ax2 = f.add_subplot(122)
for language in g2.language.unique():
    c = g2[g2['language']==language]
    ax2.plot(c.hidden_dim,c['F1-score'],label=language)
plt.legend()
plt.title('group2: F1-score <= 0.95')
ax2.set_xlabel('hidden dimension')
plt.savefig('hidden.png')
# %% Debug evaluation errors : 

test_results, test_loss = model.evaluate(
    # self.corpus.test,
    data_test['CHINESE'][0:2],
    mini_batch_size=2,
    num_workers=8,
)
print(test_results.detailed_results)

test_results, test_loss = model.evaluate(
    # self.corpus.test,
    data_test['CHINESE'][0:2],
    mini_batch_size=1,
    num_workers=8,
)
print(test_results.detailed_results)

#%% Debug evaluation error 
import flair
from flair.datasets import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn.functional as F

test1 = [data_test['CHINESE'][1],data_test['CHINESE'][1]]
test2 = data_test['CHINESE'][0:10]

data_loader = DataLoader(test1, batch_size=2, num_workers=8)


#%%
import numpy as np
inter = []
for batch in data_loader:# data_loader always has input as a list
    data_points = batch
    input_sent,input_tags = [],[]
    for sent in data_points: #Make sure data_points is always a list, doesn't matter how many elements are inside
        input_sent.append((sent.string))
        input_tags.append(sent.get_labels('tokenization')[0].value)
    
    batch_size=len(data_points)

    batch_input_tags = model.prepare_batch(input_tags, model.tag_to_ix)#.squeeze()
    batch_input_sent = model.prepare_batch(input_sent , model.letter_to_ix)
    
    embeds = model.character_embeddings(batch_input_sent)
    out, _ = model.lstm(embeds)
    tag_space = model.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
    tag_scores = F.log_softmax(tag_space, dim=2)#.squeeze()  # dim = (len(data_points),batch,len(tag))

    length_list = []
    for sent in data_points: 
        length_list.append(len(sent.string))
        
    packed_sent,packed_tags = '',''
    for string in input_sent: packed_sent += string
    for tag in input_tags: packed_tags += tag

    packed_tag_space =torch.tensor([],dtype=torch.long, device=flair.device)
    packed_tag_scores = torch.tensor([],dtype=torch.long, device=flair.device)
    packed_batch_input_tags = torch.tensor([],dtype=torch.long, device=flair.device)
    for i in np.arange(batch_size):
        packed_tag_scores = torch.cat((packed_tag_scores,(tag_scores[:length_list[i],i,:])))
        packed_tag_space = torch.cat((packed_tag_space,(tag_space[:length_list[i],i,:])))
        packed_batch_input_tags = torch.cat((packed_batch_input_tags,(batch_input_tags[:length_list[i],i])))
        

    packed_tag_predict = model.prediction_str(packed_tag_scores)
    loss = model.loss_function(packed_tag_scores, packed_batch_input_tags)

    reference = model.find_token((packed_sent, packed_tags))
    candidate = model.find_token((packed_sent, packed_tag_predict))
    inter.append( [c for c in candidate if c in reference])
print(loss)

#%% #############################################################################
# debug forward loss
# def forward_loss(
#         data_points: Union[List[DataPoint], DataPoint],
#         foreval=False,
# ) -> torch.tensor:
for batch in data_loader:# data_loader always has input as a list
    data_points = batch

if isinstance(data_points, LabeledString): # make sure data_points is a list, doesn't matter how many elements inside 
    data_points = [data_points]
input_sent, input_tags = [], []
for sent in data_points:
    input_sent.append((sent.string))
    input_tags.append(sent.get_labels('tokenization')[0].value)
batch_size = len(data_points)

batch_input_tags = model.prepare_batch(input_tags, model.tag_to_ix).to(flair.device)

batch_input_sent = model.prepare_batch(input_sent, model.letter_to_ix)
embeds = model.character_embeddings(batch_input_sent)

out, _ = model.lstm(embeds)
tag_space = model.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
tag_scores = F.log_softmax(tag_space, dim=2)#.squeeze()  # dim = (len(data_points),batch,len(tag))

length_list = []
for sentence in data_points:
    length_list.append(len(sentence.string))

packed_sent,packed_tags = '',''
for string in input_sent: packed_sent += string
for tag in input_tags: packed_tags += tag

# packed_tag_space =torch.tensor([],dtype=torch.long, device=flair.device)
# packed_tag_scores = torch.tensor([],dtype=torch.long, device=flair.device)
# packed_batch_input_tags = torch.tensor([],dtype=torch.long, device=flair.device)
loss = 0
for i in np.arange(batch_size):
    packed_tag_scores = torch.cat((packed_tag_scores,tag_scores[:length_list[i],i,:]))
    # packed_tag_space = torch.cat((packed_tag_space,tag_space[:length_list[i],i,:]))
    # packed_batch_input_tags = torch.cat((packed_batch_input_tags,batch_input_tags[:length_list[i],i]))

    loss += model.loss_function(tag_space[:length_list[i],i,:],batch_input_tags[:length_list[i],i])


tag_predict = model.prediction_str(packed_tag_scores)
# loss = model.loss_function(packed_tag_space, packed_batch_input_tags)




