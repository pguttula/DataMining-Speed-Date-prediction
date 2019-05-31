import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
inputfile = sys.argv[1]
outputfile = sys.argv[2]
data = pd.read_csv(inputfile)
labels = range(5)

list_col =  list(data.columns.values)
list_col_full = list(data.columns.values)
list_nobins = ['gender','race','race_o','samerace','field','decision']
print_list = list_col_full
for i in ['gender','race','race_o','samerace','field','decision']:
    list_col.remove(i)
list_col_part1=['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important',\
        'pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
        'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
list_col_age=['age','age_o']
list_Range_10 = ['importance_same_race','importance_same_religion','attractive', 'sincere',\
                 'intelligence', 'funny', 'ambition', 'attractive_partner', 'sincere_partner',\
                 'intelligence_parter', 'funny_partner', 'ambition_partner', \
                 'shared_interests_partner', 'sports', 'tvsports', 'exercise',\
                 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading',\
                 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga',\
                 'expected_happy_with_sd_people', 'like']
list_correlate = ['interests_correlate']
minval_age=18
maxval_age=58
minval_part1=0
maxval_part1=1
minval_range10 =0
maxval_Range10=10
minval_correlate=-1
maxval_correlate=1
t3 = data[list_col_full] 
for i in list_col_full:
    if i in list_nobins:
        t3[i+"_bin"] =  t3[i]
    elif i in list_col_part1:
        t3[i+"_bin"] = pd.cut(t3[i], bins = np.linspace(minval_part1, maxval_part1, 6),include_lowest=True,labels=labels)
    elif i in list_col_age:
        t3[i+"_bin"] = pd.cut(t3[i],bins = np.linspace(minval_age, maxval_age, 6)\
                              ,include_lowest=True,labels=labels)
    elif i in list_Range_10:
        t3[i+"_bin"] = pd.cut(t3[i], bins = np.linspace(minval_range10, maxval_Range10, 6)\
                              ,include_lowest=True,labels=labels)
    elif i in list_correlate:
        t3[i+"_bin"] = pd.cut(t3[i], bins = np.linspace(minval_correlate, maxval_correlate, 6)\
                              ,include_lowest=True,labels=labels)
    else:
        t3[i+"_bin"] = pd.cut(t3[i], 5, labels=labels)  
for i in list_col:
    data[i] = t3[i+"_bin"]
for i in list_col:    
    df = data[i].value_counts()
    print i,":", df.values.tolist()

data.to_csv(outputfile,index = False)
