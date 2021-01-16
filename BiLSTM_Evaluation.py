#!/usr/bin/env python
# coding: utf-8

# # Evaluation 
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
g1 = ['HEBREW','ARABIC']
g2 = ['PORTUGUESE','ITALIAN','FRENCH','SPANISH','GERMAN','ENGLISH','FINNISH']
g3 = ['RUSSIAN', 'KOREAN']
g4 = ['CHINESE','JAPANESE']
g5 = ['VIETNAMESE']
GroupList = [g1,g2,g3,g4,g5]
GroupNameList = ['group%s'%str(i) for i in range(1,6)]

#%% define functions
import pickle
data_train,data_test=[],[]
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    data_train += train
    data_test += test  

letter_to_ix = {}
for sent, tags in data_train+data_test:
    for letter in sent:
        if letter not in letter_to_ix:
            letter_to_ix[letter] = len(letter_to_ix)
print('Nr. of distinguish character: ',len(letter_to_ix.keys()))

tag_to_ix = {'B': 0, 'I': 1,'E':2,'S':3,'X':4}
ix_to_tag = {y:x for x,y in tag_to_ix.items()}

def prediction(input):
        output = [np.argmax(i) for i in input]
        return [ix_to_tag[int(o)] for o in output]
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prediction(input):
        output = [np.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list:
            out_str += o 
        return out_str

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

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


#%% Hyperparameters
%run functions.py 
from functions import LSTMTagger

num_classes = 5
embedding_dim = 2048 # note: maybe try smaller size?
hidden_dim = 256 # hidden_size = 256
EPOCH = 10
learning_rate = 0.1
batch_size = 1
num_layers = 1
character_size = len(letter_to_ix)
# not using this since batch size is 1
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



#%% # Initialize network
model = LSTMTagger(embedding_dim=embedding_dim, num_layers = num_layers,hidden_dim=hidden_dim,tagset_size=num_classes,character_size=character_size)
if(torch.cuda.is_available()):
	print(torch.cuda.current_device())
model = model.to(device)
model.train()
optimizer = optim.SGD(model.parameters(), learning_rate)
loss_function = nn.NLLLoss()
checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}

# %% # load test datasets and calcultae evaluation metrics 
data_test={}
for language in LanguageList:
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    data_test[language]  = test 

#%%
# type of issues:
# 1. fail to extract any tokens -> len(candidate) = 0
# 2. fail to match any correct tokens -> len(inter) = 0

#%%
### Word Level Evaluation 
# case 1 : BiLSTM_ML

model_name = '.\trained_models\BiLSTM_ML.tar'
load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)

from tqdm import tqdm; import numpy as np; import sklearn; import pandas as pd
error_sentence_ML = {}; Result_ML = {}
with torch.no_grad():
    for language in LanguageList:
        error_sentence_ML[language] = []
        R_score,P_score,F1_score = [],[],[]
        for element in tqdm(data_test[language],position=0):
            
            inputs = prepare_sequence(element[0],letter_to_ix)
            tag_scores = model(inputs)
            tag_predict = prediction(tag_scores)

            reference = find_token(element)
            candidate = find_token((element[0],tag_predict))

            inter = [c for c in candidate if c in reference]
            if len(candidate) !=0: 
                R = len(inter) / len(reference) 
                P = len(inter) / len(candidate)
            else: 
                R,P = 0,0 # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                error_sentence_ML[language].append((element,tag_predict))
            
            if (len(candidate) !=0) & ((R+P)  != 0) : # if R = P = 0, meaning len(inter) = 0, R+P = 0
                F1 = 2 * R*P / (R+P)
            else: 
                F1=0 
                if (element,tag_predict) not in error_sentence_ML:
                    error_sentence_ML[language].append((element,tag_predict))
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_ML[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

results = pd.DataFrame.from_dict(Result_ML, orient='index')
results.columns = ['Recall','Precision','F1 score']
results.to_csv('./results/BiLSTM_ML.csv')

# case 2 : BiLSTM_SL

from tqdm import tqdm;import numpy as np;import sklearn

error_sentence_SL = {}; Result_SL = {}
with torch.no_grad():
    for language in LanguageList:
        model_name = './trained_models/BiLSTM_SL_%s.pth.tar'%language
        load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)
        error_sentence_SL[language] = []
        R_score,P_score,F1_score = [],[],[]
        for element in tqdm(data_test[language],position=0):
            
            inputs = prepare_sequence(element[0],letter_to_ix)
            tag_scores = model(inputs)
            tag_predict = prediction(tag_scores)
    #         print(tag_predict)

            reference = find_token(element)
            candidate = find_token((element[0],tag_predict))

            inter = [c for c in candidate if c in reference]
            if len(candidate) !=0: 
                R = len(inter) / len(reference) 
                P = len(inter) / len(candidate)
            else: 
                R,P = 0,0 # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                error_sentence_SL[language].append((element,tag_predict))
            
            if (len(candidate) !=0) & ((R+P)  != 0) : # if R = P = 0, meaning len(inter) = 0, R+P = 0
                F1 = 2 * R*P / (R+P)
            else: 
                F1=0 
                if (element,tag_predict) not in error_sentence_SL:
                    error_sentence_SL[language].append((element,tag_predict))
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_SL[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

results = pd.DataFrame.from_dict(Result_SL, orient='index')
results.columns = ['Recall','Precision','F1 score']
results.to_csv('./results/BiLSTM_SL.csv')

#%% ### Word Level Evaluation 
# case 3 : BiLSTM_GL
from tqdm import tqdm;import numpy as np;import sklearn;import pandas as pd;import pickle
error_sentence_GL = {}; Result_GL = {}
with torch.no_grad():
    for language in LanguageList:
        index = [language in group for group in GroupList]
        groupname = GroupNameList[np.where(np.array(index)==True)[0][0]]
        print(language,groupname)
        model_name = './trained_models/BiLSTM_GL_%s.pth.tar'%groupname
        load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)

        error_sentence_GL[language] = []
        R_score,P_score,F1_score = [],[],[]
        for element in tqdm(data_test[language],position=0):
            inputs = prepare_sequence(element[0],letter_to_ix)
            tag_scores = model(inputs)
            tag_predict = prediction(tag_scores)

            reference = find_token(element)
            candidate = find_token((element[0],tag_predict))

            inter = [c for c in candidate if c in reference]
            if len(candidate) !=0: 
                R = len(inter) / len(reference) 
                P = len(inter) / len(candidate)
            else: 
                R,P = 0,0 # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                error_sentence_GL[language].append((element,tag_predict))
            
            if (len(candidate) !=0) & ((R+P)  != 0) : # if R = P = 0, meaning len(inter) = 0, R+P = 0
                F1 = 2 * R*P / (R+P)
            else: 
                F1=0 
                if (element,tag_predict) not in error_sentence_GL:
                    error_sentence_GL[language].append((element,tag_predict))
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_GL[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

results = pd.DataFrame.from_dict(Result_GL, orient='index')
results.columns = ['Recall','Precision','F1 score']
results.to_csv('./results/BiLSTM_GL.csv')

#%% check the sentences that the model fail completely. 
print(error_sentence_ML)
print(error_sentence_SL)
print(error_sentence_GL)

#%% 
import numpy as np
sentence_str = '該劇在超過一百個國家播放，而且後續的重播依然有良好的收視。'
language='CHINESE'
model_name = './trained_models/BiLSTM_SL_%s.pth.tar'%language
load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)
with torch.no_grad():
    inputs = prepare_sequence(sentence_str ,letter_to_ix)
    tag_scores = model(inputs)
    pred = prediction(tag_scores)
    print(pred)
    print(find_token((sentence_str,pred)))

model_name = './BiLSTM_ML.tar'
sentence_str = '該劇在超過一百個國家播放，而且後續的重播依然有良好的收視。'
if load_model:
    load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)
with torch.no_grad():
    inputs = prepare_sequence(sentence_str ,letter_to_ix)
    tag_scores = model(inputs)
    pred = prediction(tag_scores)
    print(pred)
    print(find_token((sentence_str,pred)))


#%% Test out the unexpected results

# for Hebrew , see why the metrics are > 1 
language = 'HEBREW'
from tqdm import tqdm;import numpy as np;import sklearn

with torch.no_grad():
    load_model = True
    model_name = './trained_models/BiLSTM_SL_%s.pth.tar'%language
    if load_model:
        load_checkpoint(torch.load(model_name,map_location=torch.device('cpu')), model, optimizer)
    error_sentence_SL[language] = []
    R_score,P_score,F1_score = [],[],[]
    for element in tqdm(data_test[language],position=0):
        
        inputs = prepare_sequence(element[0],letter_to_ix)
        tag_scores = model(inputs)
        tag_predict = prediction(tag_scores)
#         print(tag_predict)

        reference = find_token(element)
        candidate = find_token((element[0],tag_predict))

        inter = [c for c in candidate if c in reference]
        if len(candidate) !=0:
            R = len(inter) / len(reference)
            if R>1: break
            P = len(inter) / len(candidate)
        else: error_sentence_SL[language].append((element,tag_predict))
        if (R+P)  != 0 : 
            F1 = 2 * R*P / (R+P)
        else: 
            error_sentence_SL[language].append((element,tag_predict))
            F1=0
        R_score.append(R); P_score.append(P);F1_score.append(F1)
        
    Result_SL[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))
