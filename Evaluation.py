#!/usr/bin/env python
# coding: utf-8

# # Evaluation 
#%% # load test datasets and calcultae evaluation metrics 
from bilstm import LanguageList; import pickle
data_test={}
for language in LanguageList:
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        test = pickle.load(f2) 
    data_test[language]  = test 

def evaluation(model_list,file_name,data_test):
    from tqdm import tqdm; import numpy as np; import pandas as pd
    
    error_sentence = {}; Result = {}
    with torch.no_grad():
        for i,language in enumerate(LanguageList):
            load_checkpoint(torch.load(model_list[i],map_location=torch.device('cpu')), model, optimizer)
            error_sentence[language] = []; R_score,P_score,F1_score = [],[],[]
            
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
                    error_sentence[language].append((element,tag_predict))
                
                if (len(candidate) !=0) & ((R+P)  != 0) : # if R = P = 0, meaning len(inter) = 0, R+P = 0
                    F1 = 2 * R*P / (R+P)
                else: 
                    F1=0 
                    if (element,tag_predict) not in error_sentence[language]:
                        error_sentence[language].append((element,tag_predict))
                R_score.append(R); P_score.append(P);F1_score.append(F1)
                
            Result[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))

    results = pd.DataFrame.from_dict(Result, orient='index')
    results.columns = ['Recall','Precision','F1 score']
    results.to_csv('./results/%s.csv'%file_name)
    pd.DataFrame.from_dict(error_sentence, orient='index').T.to_csv('./failed_sentences/%s.csv'%file_name)
    return error_sentence,results

# type of issues: these will be saved in ./failed_sentences
# 1. fail to extract any tokens -> len(candidate) = 0
# 2. fail to match any correct tokens -> len(inter) = 0 

#%% ### Word Level Evaluation 
# case 1 : BiLSTM_ML
from bilstm import *
model, optimizer,loss_function,checkpoint = initialize_model()
model_list = ['./trained_models/BiLSTM_ML256.tar']*len(LanguageList)
file_name = 'BiLSTM_ML256_test'
error_sentence,results = evaluation(model_list,file_name,data_test)

#%% # case 2 : BiLSTM_SL
embedding_dim=2048
character_size = 6499

model, optimizer,loss_function,checkpoint = initialize_model(character_size = 6499,embedding_dim=2048)
model_list = []
for language in LanguageList:
    model_list.append('./trained_models/BiLSTM_SL/BiLSTM_SL_%s.pth.tar'%language)
file_name = 'BiLSTM_SL_test'
error_sentence,results = evaluation(model_list,file_name,data_test)
#%%


#%% # case 3 : BiLSTM_GL
embedding_dim=2048
character_size = 6499

model, optimizer,loss_function,checkpoint = initialize_model(character_size = 6499,embedding_dim=2048)
model_list = []
for language in LanguageList:
    index = [language in group for group in GroupList]
    groupname = GroupNameList[np.where(np.array(index)==True)[0][0]]
    model_list.append('./trained_models/BiLSTM_GL/BiLSTM_GL_%s.pth.tar'%groupname)
file_name = 'BiLSTM_GL'
error_sentence,results = evaluation(model_list,file_name,data_test)



#%%
### Word Level Evaluation 
# case 1 : BiLSTM_CRF_CN
from bilstm_crf import *
model_name = "./trained_models/Batch_BiLSTM_ML.tar"

file_name = 
results.to_csv('./results/Batch_BiLSTM_ML.csv')

#%% 
use_CSE = False
embedding_dim = 4096 # because using CSE 
model, optimizer,loss_function,checkpoint = initialize_model(embedding_dim=4096)

            





