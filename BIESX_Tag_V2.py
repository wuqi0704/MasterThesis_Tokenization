#!/usr/bin/env python
# coding: utf-8

# In[22]:


LanguageList_Shao = [
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


# In[211]:


import flair
language = 'CHINESE'
corpus = flair.datasets.UD_CHINESE()


# In[2]:


LanguageList_C1 = ['CHINESE','JAPANESE']


# In[102]:


from flair import datasets
language = 'CHINESE'
# corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")


# In[4]:


def token_boundary_tag(corpus_element):
    for token in corpus_element.tokens:
        if len(token.text)==1: token.add_tag(tag_value='S',tag_type='boundary')
        else: token.add_tag(tag_value='B'+'I'*(len(token.text)-2)+'E',tag_type='boundary')


# In[56]:


corpus=corpus_unsplit.downsample(0.1)
print(len(corpus.train.dataset))
print(len(corpus.get_all_sentences().datasets[0]))


# In[52]:


# prepare train and test data_sentence
import time
start_time = time.time()

for corpus_element in corpus.get_all_sentences().datasets[0]:
        token_boundary_tag(corpus_element)
        
for corpus_element in corpus.get_all_sentences().datasets[1]:
        token_boundary_tag(corpus_element)

print("--- %s seconds ---" % (time.time() - start_time))


# In[58]:


train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
test_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]


# In[65]:


corpus.get_all_sentences().datasets[0][0][1].get_tag('boundary')


# In[86]:


train_tag,test_tag =[],[]
for element in corpus.get_all_sentences().datasets[0]:
    temp_tag = [i.get_tag('boundary').value for i in element]
    tag = ''
    for i in temp_tag: tag += i
    train_tag.append(tag)
for element in corpus.get_all_sentences().datasets[1]:
    temp_tag = [i.get_tag('boundary').value for i in element]
    tag = ''
    for i in temp_tag: tag += i
    test_tag.append(tag)


# In[88]:


print(train_sentence[0])
print(train_tag[0])


# ### Get X Tag 

# In[89]:


from flair import datasets
language = 'GERMAN'
corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")


# In[109]:


corpus_unsplit.get_all_sentences()[0][0].whitespace_after


# In[91]:


# prepare train and test data_sentence
corpus = corpus_unsplit.downsample(0.1)
import time
start_time = time.time()

for corpus_element in corpus.get_all_sentences().datasets[0]:
        token_boundary_tag(corpus_element)
        
for corpus_element in corpus.get_all_sentences().datasets[1]:
        token_boundary_tag(corpus_element)

print("--- %s seconds ---" % (time.time() - start_time))


# In[92]:


train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
test_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]


# In[95]:


train_sentence[0]


# In[100]:


type(token)


# In[125]:


element = corpus.get_all_sentences().datasets[0][0]
token = element[0]
token.whitespace_after


# In[140]:


train_tag,test_tag =[],[]
for element in corpus.get_all_sentences().datasets[0]:
    tag = ''
    for token in element:
        tag += token.get_tag('boundary').value
        if token.whitespace_after == True:
            tag +='X'
    train_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == True, though it should be False
for element in corpus.get_all_sentences().datasets[1]:
    tag = ''
    for token in element:
        tag += token.get_tag('boundary').value
        if token.whitespace_after == True:
            tag +='X'
    test_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == True, though it should be False


# In[139]:


print(train_sentence[15])
print(train_tag[15])


# In[142]:


train_data = zip(train_sentence,train_tag)
train_data=list(train_data)

test_data = zip(test_sentence,test_tag)
test_data=list(test_data)


# In[148]:


type(test_data[0])


# In[178]:


import pickle
with open('./data/%s_Train.pickle'%language, 'wb') as f:
    pickle.dump(train_data, f)


# In[179]:


with open('./data/%s_Train.pickle'%language, 'rb') as f:
     data_train = pickle.load(f)


# ### Try to tag every language in the List and export training and test dataset

# In[184]:


LanguageList_Shao = [
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


# In[185]:


def token_boundary_tag_BIESX(corpus_element):
    for token in corpus_element.tokens:
        if len(token.text)==1: token.add_tag(tag_value='S',tag_type='BIESX')
        else: token.add_tag(tag_value='B'+'I'*(len(token.text)-2)+'E',tag_type='BIESX')


# In[188]:


import time
from tqdm import tqdm

for language in tqdm(LanguageList_Shao):
    start_time = time.time()
#     corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
    corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")
    corpus = corpus_unsplit

    # tag token 
    for corpus_element in corpus.get_all_sentences().datasets[0]:
            token_boundary_tag(corpus_element)

    for corpus_element in corpus.get_all_sentences().datasets[1]:
            token_boundary_tag(corpus_element)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    # prepare train and test sentence list
    train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
    test_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]
    
    # prepare train and test tag list
    train_tag,test_tag =[],[]
    for corpus_element in corpus.get_all_sentences().datasets[0]:
        tag = ''
        for token in corpus_element:
            tag += token.get_tag('boundary').value
            if token.whitespace_after == True:
                tag +='X'
        train_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == True, though it should be False
    for corpus_element in corpus.get_all_sentences().datasets[1]:
        tag = ''
        for token in corpus_element:
            tag += token.get_tag('boundary').value
            if token.whitespace_after == True:
                tag +='X'
        test_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == True, though it should be False

    # prepare train and test data by zip and save file 
    train_data = list(zip(train_sentence,train_tag))
    test_data  = list(zip(test_sentence ,test_tag))
    
    import pickle
    with open('./data/%s_Train.pickle'%language, 'wb') as f1:
        pickle.dump(train_data, f1)
    with open('./data/%s_Test.pickle'%language, 'wb') as f2:
        pickle.dump(test_data, f2)


# In[189]:


language = 'JAPANESE'
with open('./data/%s_Train.pickle'%language, 'rb') as f1:
     data_train = pickle.load(f1)
with open('./data/%s_Test.pickle'%language, 'rb') as f2:
     data_test = pickle.load(f2)       


# In[193]:


len(data_train[15][0])


# In[194]:


len(data_train[15][1])


# In[200]:


for language in LanguageList_Shao:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
         data_train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
         data_test = pickle.load(f2)   
    print(language)
    print(len(data_train[15][0]))
    print(len(data_train[15][1]))


# In[198]:


data_train[15]


# ### Manual Correct for Chinese and Japanese

# In[203]:


LanguageList_C1 = ['CHINESE','JAPANESE']


# In[204]:


import time
from tqdm import tqdm

for language in tqdm(LanguageList_C1):
    start_time = time.time()
#     corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
    corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")
    corpus = corpus_unsplit

    # tag token 
    for corpus_element in corpus.get_all_sentences().datasets[0]:
            token_boundary_tag(corpus_element)

    for corpus_element in corpus.get_all_sentences().datasets[1]:
            token_boundary_tag(corpus_element)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    # prepare train and test sentence list
    train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
    test_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]
    
    # prepare train and test tag list
    train_tag,test_tag =[],[]
    for corpus_element in corpus.get_all_sentences().datasets[0]:
        tag = ''
        for token in corpus_element:
            tag += token.get_tag('boundary').value
            if token.whitespace_after == True:
                tag +='X'
        train_tag.append(tag) # the last token of a sentence have Whitespace_after == False only correct for CN and JP
    for corpus_element in corpus.get_all_sentences().datasets[1]:
        tag = ''
        for token in corpus_element:
            tag += token.get_tag('boundary').value
            if token.whitespace_after == True:
                tag +='X'
        test_tag.append(tag) # the last token of a sentence have Whitespace_after == False only correct for CN and JP

    # prepare train and test data by zip and save file 
    train_data = list(zip(train_sentence,train_tag))
    test_data  = list(zip(test_sentence ,test_tag))
    
    import pickle
    with open('./data/%s_Train.pickle'%language, 'wb') as f1:
        pickle.dump(train_data, f1)
    with open('./data/%s_Test.pickle'%language, 'wb') as f2:
        pickle.dump(test_data, f2)


# In[206]:


for language in LanguageList_C1:
    with open('./data/%s_Train.pickle'%language, 'rb') as f1:
         data_train = pickle.load(f1)
    with open('./data/%s_Test.pickle'%language, 'rb') as f2:
         data_test = pickle.load(f2)   
    print(language)
    print(len(data_train[20][0]))
    print(len(data_train[20][1]))


# ### Correct for 'PORTUGUESE'

# In[12]:


language = 'PORTUGUESE'

import time
from tqdm import tqdm

start_time = time.time()
corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")
corpus = corpus_unsplit

# tag token 
for corpus_element in corpus.get_all_sentences().datasets[0]:
        token_boundary_tag(corpus_element)

for corpus_element in corpus.get_all_sentences().datasets[1]:
        token_boundary_tag(corpus_element)

print("--- %s seconds ---" % (time.time() - start_time))

# prepare train and test sentence list
train_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[0]]
test_sentence = [corpus_element.to_plain_string() for corpus_element in corpus.get_all_sentences().datasets[1]]

# prepare train and test tag list
train_tag,test_tag =[],[]
for corpus_element in corpus.get_all_sentences().datasets[0]:
    tag = ''
    for token in corpus_element:
        tag += token.get_tag('boundary').value
        if token.whitespace_after == True:
            tag +='X'
    train_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == False only correct for CN and JP
for corpus_element in corpus.get_all_sentences().datasets[1]:
    tag = ''
    for token in corpus_element:
        tag += token.get_tag('boundary').value
        if token.whitespace_after == True:
            tag +='X'
    test_tag.append(tag[:-1]) # the last token of a sentence have Whitespace_after == False only correct for CN and JP

# prepare train and test data by zip and save file 
train_data = list(zip(train_sentence,train_tag))
test_data  = list(zip(test_sentence ,test_tag))

# import pickle
# with open('./data/%s_Train.pickle'%language, 'wb') as f1:
#     pickle.dump(train_data, f1)
# with open('./data/%s_Test.pickle'%language, 'wb') as f2:
#     pickle.dump(test_data, f2)


# In[17]:


for i in range(len(train_data)):
    if len(train_data[i][0]) != len(train_data[i][1]):
        print(i) 


# for i in range(len(test_data)):
#     if len(test_data[i][0]) != len(test_data[i][1]):
#         print(i) 

# In[35]:


### Check data Length

