import os
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer
       
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
            terms.append(word)
    document_of_terms.append(terms)

print('                                               *****    Tokenized  Terms   *****                                                                        \n    '
    ,  document_of_terms,'\n')    

stemmer=PorterStemmer()
document_of_stemms=[]
for terms in document_of_terms:
     for word in terms :
         stemmed_terms=[stemmer.stem(word)]
         document_of_stemms.append(stemmed_terms)

print('                                               *****     Stemmed Terms   *****                                                                        \n    '
    ,  document_of_stemms,'\n')        
  
