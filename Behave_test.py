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
    error_sentence[language] = []
    R_score,P_score,F1_score = [],[],[]
    for element in tqdm(data_test[language],position=0):
        
        inputs = prepare_sequence(element[0],letter_to_ix)
        tag_scores = model(inputs)
        tag_predict = prediction_str(tag_scores)
#         print(tag_predict)

        reference = find_token(element)
        candidate = find_token((element[0],tag_predict))

        inter = [c for c in candidate if c in reference]
        if len(candidate) !=0:
            R = len(inter) / len(reference)
            if R>1: break
            P = len(inter) / len(candidate)
        else: error_sentence[language].append((element,tag_predict))
        if (R+P)  != 0 : 
            F1 = 2 * R*P / (R+P)
        else: 
            error_sentence[language].append((element,tag_predict))
            F1=0
        R_score.append(R); P_score.append(P);F1_score.append(F1)
        
    Result[language] = (np.mean(R_score), np.mean(P_score),np.mean(F1_score))