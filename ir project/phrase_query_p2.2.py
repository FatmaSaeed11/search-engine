positional_index={
    'antony': [3, {0: [0], 1: [0], 5: [0]}],
     'brutus': [3, {0: [1], 1: [1], 3: [0]}],
     'caeser': [5, {0: [2], 1: [2], 3: [1], 4: [0], 5: [1]}],
     'cleopatra': [1, {0: [3]}], 
     'mercy': [5, {0: [4], 2: [0], 3: [2], 4: [1], 5: [2]}],
     'worser': [4, {0: [5], 2: [1], 3: [3], 4: [2]}], 
     'calpurnia': [1, {1: [3]}],
     'angels': [3, {6: [0], 7: [0], 8: [0]}], 
     'fools': [4, {6: [1], 7: [1], 8: [1], 9: [0]}], 
     'fear': [3, {6: [2], 7: [2], 9: [1]}], 
     'in': [4, {6: [3], 7: [3], 8: [2], 9: [2]}],
     'rush': [4, {6: [4], 7: [4], 8: [3], 9: [3]}],
     'to': [4, {6: [5], 7: [5], 8: [4], 9: [4]}], 
     'tread': [4, {6: [6], 7: [6], 8: [5], 9: [5]}],
     'where': [4, {6: [7], 7: [7], 8: [6], 9: [6]}]
     }

query='fools fear'

final_list=[[]for i in range (10)]
print('                                    ***** all response list *****                                                     ')

for word in query.split():    
        for key in positional_index[word][1].keys():
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
 