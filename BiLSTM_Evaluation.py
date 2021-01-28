#!/usr/bin/env python
# coding: utf-8

# # Evaluation 

#%% Hyperparameters
from functions import *

# load test datasets and calcultae evaluation metrics 
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


#%%
### Word Level Evaluation 
# case 1 : Batch_BiLSTM_ML

model_name = "./trained_models/Batch_BiLSTM_ML.tar"
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
            tag_predict = prediction_str(tag_scores)

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
                if (element,tag_predict) not in error_sentence_ML[language]:
                    error_sentence_ML[language].append((element,tag_predict))
            R_score.append(R); P_score.append(P);F1_score.append(F1)
            
        Result_ML[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

results = pd.DataFrame.from_dict(Result_ML, orient='index')
results.columns = ['Recall','Precision','F1 score']
results.to_csv('./results/Batch_BiLSTM_ML.csv')