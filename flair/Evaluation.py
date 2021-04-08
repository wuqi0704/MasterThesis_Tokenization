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
#%%
import torch 
from tqdm import tqdm
import pandas as pd
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
import torch 
from tqdm import tqdm
import pandas as pd
# # state = torch.load('./resources/taggers/%s/best-model.pt'%model_name,map_location=torch.device('cpu'))
# # model_names = ['1_h8','1_h32','1_h64','1_h128','1_h256']

output = {}
for language in tqdm(['CHINESE','VIETNAMESE']):

    state = torch.load('/Users/qier/Downloads/ML_Tagger/5_CRFCSE_4096%s/best-model.pt'%language,map_location=torch.device('cpu'))
    tokenizer = FlairTokenizer() 
    model = tokenizer._init_model_with_state_dict(state)
    result, eval_loss = model.evaluate(data_test[language],mini_batch_size=1)
    obj = result.detailed_results
    output[language] = [float(item.split(':')[1]) for item in obj.split('\n-')[1:]]

out_dataframe = pd.DataFrame.from_dict(output, orient='index')
out_dataframe.columns = ['F1-score','Precision-score','Recall-score']
# out_dataframe.to_csv('/Users/qier/MasterThesis_Tokenization/results/3_SL.csv')


# %%
torch.load('/Users/qier/Downloads/ML_Tagger/5_CRFCSE_4096CHINESE/best-model.pt',map_location=torch.device('cpu'))
# %%