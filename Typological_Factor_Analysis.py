#!/usr/bin/env python
# coding: utf-8

# %% Typological Analysis

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

#%% Get statistic from UD datasets
from flair import datasets
from tqdm import tqdm ; import time
import numpy as np; import pandas as pd

TF = {}
for language in tqdm(LanguageList_Shao):
    TF[language] = []
    corpus_split   = eval('datasets.'+'UD_'+language + "(split_multiwords=True)")
    corpus_unsplit = eval('datasets.'+'UD_'+language + "(split_multiwords=False)")
    
    # unsplit loop: get tokens
    sentences = corpus_unsplit.get_all_sentences()
    letter_to_ix, token_to_ix,intspace_token_to_ix= {},{},{}
    space_count = 0; internal_space_count = 0 ; nospace_after_token_count = 0 
    total_token_nr = 0 ; total_character_nr = 0

    for sent in sentences:
        total_token_nr += len(sent.tokens)
        space_count += sent.to_plain_string().count(' ')
        sent_text = sent.to_plain_string()
        letter_list = list(sent_text)
        total_character_nr += len(list(sent_text))
        
        for token in sent.tokens:
            if token.text.count(' ')!= 0:
                if token.text not in intspace_token_to_ix:
                    intspace_token_to_ix[token.text] = len(intspace_token_to_ix)
            internal_space_count += token.text.count(' ')
            
            if token.text not in token_to_ix:
                token_to_ix[token.text] = len(token_to_ix)
            if token.whitespace_after == False: nospace_after_token_count += 1
        
        for letter in letter_list:
            if letter not in letter_to_ix:
                letter_to_ix[letter] = len(letter_to_ix)
    
    # split loop: get words
    sentences = corpus_split.get_all_sentences()
    word_to_ix = {}
    total_word_nr = 0
    for sent in sentences:
        total_word_nr += len(sent.tokens)
        for word in sent.tokens:
            if word.text not in word_to_ix:
                word_to_ix[word.text] = len(word_to_ix)
    
    multiword_token_to_ix = {}
    for token in token_to_ix:
        if token not in word_to_ix:
            multiword_token_to_ix[token] = len(multiword_token_to_ix)
    
    total_multiword_token_nr = total_word_nr-total_token_nr # accurate only when 1 token = 2 words however does not impact much the analysis
    
    # starts to calculate statistics for next step analysis
    TC = len(letter_to_ix.keys()) 
    TT = len(token_to_ix.keys())
    TW = len(word_to_ix.keys())
    TM = len(multiword_token_to_ix)
    TI = len(intspace_token_to_ix.keys())
    AL = sum([len(i) for i in list(token_to_ix.keys())])/len(list(token_to_ix.keys()))
    SD = np.std(np.array([len(i) for i in list(token_to_ix.keys())]))
    PNS = nospace_after_token_count /total_token_nr
    PI = internal_space_count/space_count
    PS = space_count/total_character_nr
    PM = total_multiword_token_nr/total_token_nr # accurate only when 1token = 2words 
    SF = total_token_nr/(space_count - internal_space_count)
    
    TF[language].append(TC) # type of characters
    TF[language].append(TT) # type of tokens
    TF[language].append(TW) # type of words
    TF[language].append(TM) # type of multiword token 
    TF[language].append(TI) # type of token with internal space

    TF[language].append(AL) # avg length of tokens
    TF[language].append(SD) # standard deviation of token length

    TF[language].append(PNS) # proportion of token that are not followed by space 
    TF[language].append(PI)  # proportion of token internal space
    TF[language].append(PS)  # proportion of space # space_count/total_character_nr
    TF[language].append(PM)  # proportion of multiword token
    
    TF[language].append(SF) # total_token_nr/segment space count
    
    TF[language].append(space_count) # NS
    TF[language].append(internal_space_count) # NI
    TF[language].append(total_character_nr ) # NC
    TF[language].append(total_token_nr) # NT
    TF[language].append(total_word_nr) # NW
    TF[language].append(total_multiword_token_nr) # NM
    

# %% save data 
import pickle
name = 'TF_LanguageList'
def save_obj(obj, name ):
    with open('./TF/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(TF,name)

#%% ### Start Analysis here
def load_obj(name ):
    with open('./TF/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# define different combinations of Typological Factors
col = ['TC','TT','TW','TM','TI','AL','SD','PNS','PI','PS','PM','SF','NS','NI','NC','NT','NW','NM']
col1  = ['TC','TT','TW','TM','TI','AL','SD','PNS','PI','PS','PM','SF']
col2  = ['TC','TT','TW','TM','TI','AL','SD','PNS','PI','PS','PM']
col3  = ['TC','TT','TM','TM','TI','AL','SD','PNS','PI',     'PM']

name = 'TF_LanguageList'
TF = load_obj(name)
TF_DF_full = pd.DataFrame.from_dict(TF, orient='index',columns=col)

col_list = [col,col1,col2,col3]

# Run TF Analysis for 4 different factor cominations

# preprocessing : columns selection and normalization
import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

explained = [] # explained variance by PC 
for n,col_name in enumerate(col_list):
    TF_DF = TF_DF_full.loc[:,col_name]
    scaled = preprocessing.StandardScaler().fit_transform(TF_DF.values)
    TF_DF_scaled = pd.DataFrame(scaled,columns=TF_DF.columns,index=LanguageList)

    # PCA Analysis
    from sklearn.decomposition import PCA
    pca = PCA(); X = TF_DF_scaled
    components = pca.fit_transform(X)
    var,per = pca.explained_variance_ , pca.explained_variance_ratio_
    plt.figure();plt.plot(range(1,len(per)+1),per);plt.title('explained percentage by PC'); plt.savefig('./TF/var_%s.png'%n)
    explained.append(per)

    ### The Optimal number of Cluster 
    from sklearn.metrics import silhouette_score
    sil = [];kmax = len(LanguageList);X = TF_DF_scaled
    for k in range(2, kmax): # minimum number of clusters should be 2
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeans.labels_
        sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    sil = pd.Series(sil,index=range(2,kmax))
    plt.figure(); plt.plot(sil); plt.title('silhouette score for different number of clusters');plt.savefig('./TF/sil_%s.png'%n)
    optimal_cluster = np.argmax(sil)

    # K-means clustering 
    from sklearn.cluster import KMeans
    pca = PCA(n_components=5)
    components = pca.fit_transform(TF_DF_scaled)

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0).fit(TF_DF_scaled.values)
    TF_DF_scaled.loc[:,'cluster'] = kmeans.labels_

    kmeans = KMeans(n_clusters=optimal_cluster, random_state=0).fit(components)
    TF_DF_scaled.loc[:,'cluster_PCA'] = kmeans.labels_

    # plot and save plots 
    from adjustText import adjust_text

    x,y = components.T[0],components.T[1]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x,y, c=TF_DF_scaled.cluster_PCA); plt.title('optimal cluster = %s'% optimal_cluster)
    texts = [plt.text(x[i], y[i], TF_DF.index[i], ha='center', va='center') for i in range(len(TF_DF))]
    adjust_text(texts)

    plt.savefig('./TF/cluster_%s.png'%n)

# Remark: The results of all 4 combinations are quite similar, and 
# col2  = ['TC','TT','TW','TM','TI','AL','SD','PNS','PI','PS','PM']
# is chosen to be reported in the thesis 
