#%% initialize model and load model

# 1. load your data and convert to list of LabeledString

from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.embeddings import token
from tokenizer_model import FlairTokenizer
from tokenizer_model import LabeledString
from tqdm import tqdm 
import pickle 
import torch

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
# import random
# random.seed(123)
# for language in LanguageList:
#     data_test[language]=random.choices(data_test[language],k=400)
#     data_dev[language]=random.choices(data_dev[language],k=400)

for language in LanguageList:
    for item in data_test[language]:
        if len(item.string)==1:
            data_test[language].remove(item)

import torch 
from tqdm import tqdm
import pandas as pd
#%%
# # state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
# # model_names = ['1_h8','1_h32','1_h64','1_h128','1_h256']

# output = {}
# for language in tqdm(LanguageList):

#     state = torch.load('/Users/qier/Downloads/ML_Tagger/3_SL/3_%s/best-model.pt'%language,map_location=torch.device('cpu'))
#     tokenizer = FlairTokenizer() 
#     model = tokenizer._init_model_with_state_dict(state)
#     result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
#     obj = result.detailed_results
#     output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

# out_dataframe = pd.DataFrame.from_dict(output, orient='index')
# out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
# # out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/3_SL.csv')

# # state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
# # model_names = ['1_h8','1_h32','1_h64','1_h128','1_h256']

# g1 = ['HEBREW','ARABIC']
# g2 = ['PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH','FINNISH']
# g3 = ['RUSSIAN', 'KOREAN']
# g4 = ['CHINESE','JAPANESE']
# g5 = ['VIETNAMESE']
# GroupList = [g1,g2,g3,g4,g5]


# output = {}
# for i,g in enumerate(GroupList):
#     state = torch.load('/Users/qier/Downloads/ML_Tagger/4_group%s/best-model.pt'%str(i+1),map_location=torch.device('cpu'))
#     tokenizer = FlairTokenizer() 
#     model = tokenizer._init_model_with_state_dict(state)
#     for language in tqdm(GroupList[i]):
#         result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
#         obj = result.detailed_results
#         output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

# out_dataframe = pd.DataFrame.from_dict(output, orient='index')
# out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
# out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/4_GL.csv')

# %%
# # state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
# # model_names = ['1_h8','1_h32','1_h64','1_h128','1_h256']

# output = {}
# for language in tqdm(['CHINESE','VIETNAMESE']):

#     state = torch.load('/Users/qier/Downloads/ML_Tagger/5_CRF_256_%s/best-model.pt'%language,map_location=torch.device('cpu'))
#     tokenizer = FlairTokenizer() 
#     model = tokenizer._init_model_with_state_dict(state)
#     result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
#     obj = result.detailed_results
#     output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

# out_dataframe = pd.DataFrame.from_dict(output, orient='index')
# out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
# out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/5_RF_256.csv')

# %%
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

for hd in tqdm([32,64,128,256,512,1024,2048,4096]):
    output = {}
    state = torch.load(f'/Users/qier/Downloads/Tagger/2/2_e{hd}/best-model.pt',map_location=torch.device('cpu'))
    tokenizer = FlairTokenizer() 
    model = tokenizer._init_model_with_state_dict(state)
    for language in LanguageList:
        result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
        obj = result.detailed_results
        output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

    out_dataframe = pd.DataFrame.from_dict(output, orient='index')
    out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
    # out_dataframe.to_csv(f'/Users/qier/MasterThesis_Tokenization/results/2_e{hd}.csv')
    # out_dataframe.to_csv(f'/Users/qier/Downloads/Tagger/2_e{hd}.csv')

# %%
output = {}
LanguageList = [
    'RUSSIAN',
    'KOREAN',
]
import pickle
# 1. load your data and convert to list of LabeledString
data_train, data_test, data_dev = {}, {}, {}
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train]
    data_test[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test]
    data_dev[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev]

import numpy as np
import random
N = np.array([1,2,3,4,5])*3000
random.seed(123)
for n in N:
    for language in LanguageList:
        data_train[language]=random.choices(data_train[language],k=n)
        data_test[language]=random.choices(data_test[language],k=np.int(n/10))
        data_dev[language]=random.choices(data_dev[language],k=np.int(n/10))

        state = torch.load('/Users/qier/Downloads/Tagger/6_SL/6_%s_%s/best-model.pt'%(language,n),map_location=torch.device('cpu'))
        tokenizer = FlairTokenizer() 
        model = tokenizer._init_model_with_state_dict(state)
        result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
        obj = result.detailed_results
        output[(language,n)] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

out_dataframe = pd.DataFrame.from_dict(output, orient='index')
out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
out_dataframe.to_csv('RK_SL_downsized.csv')
# %% evaluation for SL downsized 
import pandas as pd
LanguageList = [
    # 'HEBREW',
    # 'ARABIC',
    # 'PORTUGUESE',
    'ITALIAN',
    'FRENCH',
    'SPANISH',
    'GERMAN',
    # 'ENGLISH',
    # 'RUSSIAN',
    # 'FINNISH',
    # 'VIETNAMESE',
    # 'KOREAN',
    # 'CHINESE',
    # 'JAPANESE'
]
import pickle

# 1. load your data and convert to list of LabeledString
data_train, data_test, data_dev = {}, {}, {}
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train]
    data_test[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test]
    data_dev[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev]
output = {}
import numpy as np
import random
N = np.array([1,3,5,7,9,11])*1000
# N = np.array([1,5,9,13,17,21])*1000
# random.seed(123)
for n in N:
    for language in tqdm(LanguageList):
        # data_train[language]=random.choices(data_train[language],k=n)
        # data_test[language]=random.choices(data_test[language],k=np.int(n/10))
        # data_dev[language]=random.choices(data_dev[language],k=np.int(n/10))

        state = torch.load(f'/Users/qier/Downloads/Tagger/3_SL_downsized/6_SL_dropout0.5/6_{language}_{n}/best-model.pt',map_location=torch.device('cpu'))
        tokenizer = FlairTokenizer() 
        model = tokenizer._init_model_with_state_dict(state)
        result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
        obj = result.detailed_results
        output[(language,n)] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

out_dataframe = pd.DataFrame.from_dict(output, orient='index')
out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/6_SL_downsized_drop.csv')


# %% Evaluation for SL
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
data_train, data_test, data_dev = {}, {}, {}
for language in LanguageList:
    with open('resources/%s_Train.pickle' % language, 'rb') as f1:
        train = pickle.load(f1)
    with open('resources/%s_Test.pickle' % language, 'rb') as f2:
        test = pickle.load(f2)
    with open('resources/%s_Dev.pickle' % language, 'rb') as f3:
        dev = pickle.load(f3)

    data_train[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in train]
    data_test[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in test]
    data_dev[language] = [LabeledString(pair[0]).set_label('tokenization', pair[1]) for pair in dev]
output = {}
import numpy as np

for language in tqdm(LanguageList):

    state = torch.load(f'/Users/qier/Downloads/Tagger/3_SL/3_SL_dropout/6_SL_dropout0.5_{language}/best-model.pt',map_location=torch.device('cpu'))
    tokenizer = FlairTokenizer() 
    model = tokenizer._init_model_with_state_dict(state)
    result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
    obj = result.detailed_results
    output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

out_dataframe = pd.DataFrame.from_dict(output, orient='index')
out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
out_dataframe.to_csv('3_SL_drop.csv')
