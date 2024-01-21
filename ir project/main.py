import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
from nltk.stem import PorterStemmer
#preprocessing
stopwords=stopwords.words('english')

stopwords.remove('in')
stopwords.remove('to')
stopwords.remove('where')
       
files_name=natsorted(os.listdir('files'))
print('                                              *****    Documents     ***** ' )
document_of_terms=[]
for files in files_name:
    with open(f'files/{files}','r') as f:
        document=f.read()
        
        print(document)
 
    tokenized_documents=word_tokenize(document)
    terms=[]
    for word in tokenized_documents:
        if word not in stopwords:
            terms.append(word)
    document_of_terms.append(terms)

print('                                               *****    Documents in  Terms   *****                                                                        \n    '
    ,  document_of_terms,'\n')                        

#########
'''antony brutus caeser cleopatra mercy worser
antony brutus caeser calpurnia 
mercy worser
brutus caeser mercy worser
caeser mercy worser
antony caeser mercy
angels fools fear in rush to tread where
angels fools fear in rush to tread where
angels fools in rush to tread where
fools fear in rush to tread where'''
'''stemmer=PorterStemmer()
stemmed_document=[]
for terms in document_of_terms:
    stemmed_terms=[]
    for word in terms:
         stemmer.stem(nd(stemmed_terword)
         stemmed_document.appems)
         
 stemmer=PorterStemmer()
stemmed_document=[]
for terms in document_of_terms:
    stemmed_terms=[stemmer.stem(word) for word in terms ]
    stemmed_document.append(stemmed_terms)
    
print(stemmed_document)   ///        
#####  
////
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Example document
document = "Stemming is an important technique for search engines"

# Tokenize the document into words
tokenized_words = word_tokenize(document)

# Apply stemming to each word
stemmed_words = [stemmer.stem(word) for word in tokenized_words]

# Join the stemmed words back into a document
stemmed_document = ' '.join(stemmed_words)

# Print the stemmed document
print(stemmed_document)
'''       
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
            if final_list[key]!=[]:
                
                 if final_list[key][-1] == positional_index[word][1][key][0]-1:
                        final_list[key].append(positional_index[word][1][key][0])
        
            else:
                 final_list[key].append(positional_index[word][1][key][0])
            
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
'''document_length=pd.DataFrame({
    f'doc{i}_length':np.sqrt(tf_idf[f'doc{i}'].apply(lambda x:x**2).sum()) for i in range (1,10)
                             
},index=[0])
norm_tf_idf= tf_idf.divide(doc_len.values[0],axis=1)
print(norm_tf_idf)'''
'''
# doc length

def get_doc_len(col):
    return np.sqrt(tf_idf[col].apply(lambda x:x**2).sum())

doc_len=pd.DataFrame(columns=['doc length'])
for col in tf_idf.columns:
    doc_len.loc[0, col+'__length']=get_doc_len(col)   
print('                                  *****         Doc Length             *****                                              ') 
print(doc_len,'\n')    

'''
'''
#ranked retrival

import pandas as pd
import math

document_of_terms=[
   ['antony', 'brutus', 'caeser', 'cleopatra', 'mercy', 'worser'],
    ['antony', 'brutus', 'caeser', 'calpurnia'],
    ['mercy', 'worser'], 
    ['brutus', 'caeser', 'mercy', 'worser'],
    ['caeser', 'mercy', 'worser'], 
    ['antony', 'caeser', 'mercy'], 
    ['angels', 'fools', 'fear', 'in', 'rush', 'to', 'tread', 'where'], 
    ['angels', 'fools', 'fear', 'in', 'rush', 'to', 'tread', 'where'],
    ['angels', 'fools', 'in', 'rush', 'to', 'tread', 'where'], 
    ['fools', 'fear', 'in', 'rush', 'to', 'tread', 'where']   
]

# term frequancy
all_words=[]
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)
       
def get_term_freq(doc):
    words_found=dict.fromkeys(all_words,0)
    for word in doc:
        words_found[word]+=1
    return words_found

term_freq=pd.DataFrame(get_term_freq(document_of_terms[0]).values(),index=get_term_freq(document_of_terms[0]).keys())              
print('                                  *****     tf for each  term in doc     *****                                              ')
print()        
for i in range(1,len(document_of_terms)):
    term_freq[i]=get_term_freq(document_of_terms[i]).values()
    
term_freq.columns=['doc'+str(i) for i in range (1,11)]    
print(term_freq) 
print()    

#weighted term frequancy

def get_weighted_term_freq(x):
    if x>0:
      return 1+math.log(x)
    return 0
     
print('                                  *****     weighted tf for each  term in doc    *****                                              ')
       
for i in range(1,len(document_of_terms)+1):
    term_freq['doc'+str(i)]= term_freq['doc'+str(i)].apply(get_weighted_term_freq) 
    
print(term_freq,'\n')
print()

#idf

tfd=pd.DataFrame(columns=['df','idf'])

print('                                  *****        IDF          *****                                              ')
for i in range(len(term_freq)):
    
    frequancy=term_freq.iloc[i].values.sum()
    
    tfd.loc[i,'df']=frequancy
    tfd.loc[i,'idf']=math.log(10/(float(frequancy)))
    
tfd.index=term_freq.index
    
print(tfd)    

# tf*idf

term_freq_inve_doc_freq=term_freq.multiply(tfd['idf'],axis=0)

print('                                  *****         TF*IDF             *****                                              ')
print()      
print(term_freq_inve_doc_freq)




'''
'''
document_length=pd.DataFrame({
    f'doc{i}_length':np.sqrt(tf_idf[f'doc{i}'].apply(lambda x:x**2).sum()) for i in range (1,11)
                             
},index=[0])
print('                                  *****         Doc Length             *****                                              ') 
print()
print(document_length,'\n') 

norm_tf_idf= tf_idf.divide(document_length.values[0],axis=1)
print('                                  *****        Normalized TF*IDF           *****                                              ')
print()
print(norm_tf_idf)

# get query from user

input_q=input('write your query:')

def get_w_tf(x):
    try:
        return math.log10(x)+1
    except:
        return 0

import math
query=pd.DataFrame(index=norm_tf_idf.index) 
'''