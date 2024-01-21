import os
import numpy as np

from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer
#preprocessing
       
files_name=natsorted(os.listdir('files'))

document_of_terms=[]
for files in files_name:
    with open(f'files/{files}','r') as f:
        document=f.read()
 
    tokenized_documents=word_tokenize(document)
    terms=[]
    for word in tokenized_documents:
            terms.append(word)
    document_of_terms.append(terms)

print('                                               *****    Documents in  Terms   *****                                                                        \n    '
    ,  document_of_terms,'\n')    


#stemming 
stemmer=PorterStemmer()
document_of_stemms=[]
for terms in document_of_terms:
     for word in terms :
         stemmed_terms=[stemmer.stem(word)]
         document_of_stemms.append(stemmed_terms)

print('                                               *****     Stemmed Terms   *****                                                                        \n    '
    ,  document_of_stemms,'\n')        
  

#positional_index

document_number = 0
positional_index = {}

print() 
print("                                               *****     positions    *****                                       \n   ")
for document in document_of_terms:
    for positional, term in enumerate(document):
        if term in positional_index:
            positional_index[term][0] = positional_index[term][0] + 1

            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)
            else:
                positional_index[term][1][document_number] = [positional]

        else:
            positional_index[term] = []
            positional_index[term].append(1)
            positional_index[term].append({})
            positional_index[term][1][document_number] = [positional]

    document_number += 1
    

print(positional_index,'\n')

#query_preprocessing

query='fools fear'

final_list=[[]for i in range (10)]
print('                                    ***** all response list *****                                                     ')
for word in query.split():    
    if word in positional_index.keys():
        for key in positional_index[word][1].keys():
            #print(key)
            if final_list[key-1]!=[]:
                
                 if final_list[key-1][-1] == positional_index[word][1][key][0]-1:
                        final_list[key-1].append(positional_index[word][1][key][0])
        
            else:
                 final_list[key-1].append(positional_index[word][1][key][0])
            
print(final_list,'\n')

print('                                  *****response list to query*****                                              ')
for position,list in enumerate(final_list,start=1):
    if len(list)==len(query.split()):
      print(position,list)     


#find tf and wtf
import pandas as pd
import math


all_words=[]
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)
 #tf       
def get_term_freq(doc):
    words_found=dict.fromkeys(all_words,0)
    for word in doc:
        words_found[word]+=1
    return words_found

term_freq=pd.DataFrame(get_term_freq(document_of_terms[0]).values(),index=get_term_freq(document_of_terms[0]).keys())              
print('                                  *****term freq for each  term in doc*****                                              ')
print()        
for i in range(1,len(document_of_terms)):
    term_freq[i]=get_term_freq(document_of_terms[i]).values()
    
term_freq.columns=['doc'+str(i) for i in range (1,11)]    
print(term_freq)     

#wtf

def get_weighted_term_freq(x):
    if x>0:
      return math.log(x)+1
    return 0
     
print('                                  ***** weighted term freq for each  term in doc*****                                              ')
print()        
for i in range(1,len(document_of_terms)+1):
    term_freq['doc'+str(i)]= term_freq['doc'+str(i)].apply(get_weighted_term_freq) 
    
print(term_freq,'\n')
print()
 
#idf and tf*idf
 
tfd=pd.DataFrame(columns=['df','idf'])

print('                                  *****        IDF          *****                                              ')
for i in range(len(term_freq)):
    
    frequancy=term_freq.iloc[i].values.sum()
    
    tfd.loc[i,'df']=frequancy
    tfd.loc[i,'idf']=math.log10(10/(float(frequancy)))
    
tfd.index=term_freq.index
    
print(tfd)
# tf*idf

tf_idf=term_freq.multiply(tfd['idf'],axis=0)

print('                                  *****         TF*IDF             *****                                              ')
print()      
print(tf_idf)
# doc length

def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x:x**2).sum())

doc_len=pd.DataFrame()
for col in tf_idf.columns:
    doc_len.loc['length', col+'__length']=get_doc_len(col)   
print('                                  *****         Doc Length             *****                                              ') 
print(doc_len,'\n')    



# normalization
norm_tf_idf= tf_idf.divide(doc_len.values[0],axis=1)
print('                                  *****        Normalized TF*IDF           *****                                              ')
print()
print(norm_tf_idf)

'''
norm_tf_idf= tf_idf.divide(document_length.values[0],axis=1)
print('                                  *****        Normalized TF*IDF           *****                                              ')
print()
print(norm_tf_idf)
'''

# input query
input_q=input('write your query:')

#fun to calculate w_tf for query
def get_w_tf(x):
    try:
        return math.log10(x)+1
    except:
        return 0

# query processing
import math
query=pd.DataFrame(index=norm_tf_idf.index)

#tf 
query['tf']=[1 if x in input_q.split() else 0 for x in (norm_tf_idf.index)]

#w_tf
query['w_tf']=query['tf'].apply(lambda x:get_w_tf(x))

#dot_product  
product=norm_tf_idf.multiply(query['w_tf'],axis=0)
query['idf']=tfd['idf']*query['w_tf']
query['tf_idf']=query['w_tf']*query['idf']
query['norm']=0
for i in range(len(query)):
    query['norm'].iloc[i]=float(query['idf'].iloc[i])/math.sqrt(sum(query['idf'].values**2))

product2=product.multiply(query['norm'],axis=0)   

#
'''
query['idf'].loc[input_q.split()]

product2.loc[input_q.split()].values


scores={}
for col in product2.columns:
    if 0 in product2[col].loc[input_q.split()].values:
        pass
    else:
        scores[col]=product2[col].sum()
        
scores

product2[scores.keys()]

product2[scores.keys()].loc[input_q.split()]

prod_res=product2[scores.keys()]

prod_res.sum()

final_score=sorted(scores.items(),key=lambda x:x[1],reverse=True)

for doc in final_score:
    print(doc[0],end='')'''