#!/usr/bin/env python
# coding: utf-8
#%%
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

def token_boundary_tag(corpus_element):
    for token in corpus_element.tokens:
        if len(token.text)==1: token.add_tag(tag_value='S',tag_type='BIESX')
        else: token.add_tag(tag_value='B'+'I'*(len(token.text)-2)+'E',tag_type='BIESX')

# store the length of datasets
length = {}

#%%
# Note that Vietnamese is not in flair.datasets yet
from flair import datasets
import time; from tqdm import tqdm

for language in tqdm(LanguageList[]):
    start_time = time.time()
#     corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
    corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")
    corpus = corpus_unsplit
    # get datasets length i.e. number of sentences in train, test and development sets.
    length[language] = []
    length[language].append(len(corpus.get_all_sentences()))
    length[language].append(len(corpus.get_all_sentences().datasets[0]))
    length[language].append(len(corpus.get_all_sentences().datasets[1]))
    length[language].append(len(corpus.get_all_sentences().datasets[2]))
    # tag token 
    for corpus_element in list(corpus.get_all_sentences()):
            token_boundary_tag(corpus_element)
    print("--- %s seconds ---" % (time.time() - start_time)) # print out the time needed to load each language
    
    # prepare train and test sentence list
    train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
    test_sentence  = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]
    dev_sentence   = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[2]]
    
    # prepare tag list 
    # Remark: add tag X for white space here 
    train_tag,test_tag,dev_tag =[],[],[]
    for corpus_element in corpus.get_all_sentences().datasets[0]:
        tag = ''
        for i,token in enumerate(corpus_element):
            tag += token.get_tag('BIESX').value
            # if the token is not the last token in a sentence, and the white space after the token is equal true, add X tag
            # for some datasets, the last token of a sentence have Whitespace_after == True, though it should be False
            if (i != len(corpus_element)-1) & (token.whitespace_after == True):
                tag +='X'
        train_tag.append(tag) 
            
    for corpus_element in corpus.get_all_sentences().datasets[1]:
        tag = ''
        for i,token in enumerate(corpus_element):
            tag += token.get_tag('BIESX').value
            # if the token is not the last token in a sentence, and the white space after the token is equal true, add X tag
            # for some datasets, the last token of a sentence have Whitespace_after == True, though it should be False
            if (i != len(corpus_element)-1) & (token.whitespace_after == True):
                tag +='X'
        test_tag.append(tag) 
    
    for corpus_element in corpus.get_all_sentences().datasets[2]:
        tag = ''
        for i,token in enumerate(corpus_element):
            tag += token.get_tag('BIESX').value
            # if the token is not the last token in a sentence, and the white space after the token is equal true, add X tag
            # for some datasets, the last token of a sentence have Whitespace_after == True, though it should be False
            if (i != len(corpus_element)-1) & (token.whitespace_after == True):
                tag +='X'
        dev_tag.append(tag) 

    # prepare train and test data by zip and save file 
    train_data = list(zip(train_sentence,train_tag))
    test_data  = list(zip(test_sentence ,test_tag))
    dev_data  = list(zip(dev_sentence ,dev_tag))
    
    import pickle
    with open('./data/%s_Train.pickle'%language, 'wb') as f1:
        pickle.dump(train_data, f1)
    with open('./data/%s_Test.pickle'%language, 'wb') as f2:
        pickle.dump(test_data, f2)
    with open('./data/%s_Dev.pickle'%language, 'wb') as f3:
        pickle.dump(dev_data, f3)
  

#%% 
# show dataset size:
import pandas as pd
l = pd.DataFrame.from_dict(length,orient="index")
l.columns = ['total','train','test','dev']
l.to_csv('./data/datasets_size.csv')
print(l)

#%%
# double check if tag set and sentence set are of the same length
import pickle
for language in LanguageList:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
        data_train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
        data_test = pickle.load(f2)  
    with open('./data/%s_Dev.pickle'%language, 'rb') as f3:
        data_dev = pickle.load(f3)

    for element in data_train:
        if len(element[0]) != len(element[1]):
            print(language, element)
