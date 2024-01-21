import pandas as pd
import numpy as np
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
    tfd.loc[i,'idf']=math.log10(10/(float(frequancy)))
    
tfd.index=term_freq.index
    
print(tfd)    

# tf*idf

tf_idf=term_freq.multiply(tfd['idf'],axis=0)

print('                                  *****         TF*IDF             *****                                              ')
print()      
print(tf_idf)

# doc length

document_length=pd.DataFrame()
def get_documents_length(col):
    return np.sqrt(tf_idf[col].apply(lambda x:x**2).sum())


for column in tf_idf.columns:
    document_length.loc[0,column+'_len']=get_documents_length(column)
print('                                  *****         Doc Length             *****                                              ') 
print(document_length,'\n')    


# normalized tf*idf
norm_tf_idf = pd.DataFrame()
def get_normalized(col, x):
    try:
        return x / document_length[col + '__len'].values[0]
    except:
        return 0
for column in tf_idf.columns:
    norm_tf_idf[column] = tf_idf[column].apply(lambda x: get_normalized(column, x))

print('                                  *****        Normalized TF*IDF           *****                                              ')
print(norm_tf_idf, '\n')