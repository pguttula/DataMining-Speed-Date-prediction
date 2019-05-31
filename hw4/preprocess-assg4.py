import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
data = pd.read_csv('dating-full.csv',nrows = 6500)

df = data;

df.drop("race",axis=1, inplace=True)
df.drop("race_o",axis=1, inplace=True)
df.drop("field",axis=1, inplace=True)
#df will have 6500 rows and 50 cols

t1 = df[['gender']]
A = pd.DataFrame(np.sort(t1['gender'].unique()),columns=['gender'])
A['gender_enum'] = range(len(t1['gender'].unique()))
t1 = t1.merge(A, on= 'gender', how= 'left')
#df has its 3 clumns dropped and gender values encoded
df['gender'] = t1['gender_enum']
def prune_gaming(a):    
    if a>10: 
        return 10
    else: 
        return a
def prune_reading(a): 
    if a>10: 
        return 10
    else: 
        return a
df['gaming'] = df['gaming'].map(prune_gaming)
df['reading'] = df['reading'].map(prune_reading)

modcol = ['attractive_important','sincere_important','intelligence_important',\
            'funny_important','ambition_important','shared_interests_important',\
            'pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
            'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

t3=df[modcol]
type1_list = ['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']
type2_list = ['pref_o_attractive','pref_o_sincere','pref_o_intelligence',\
            'pref_o_funny','pref_o_ambitious','pref_o_shared_interests']
t3['total'] = \
        t3[type1_list].sum(axis=1)
for i in type1_list:            
    t3[i+"_new"] = t3[i].div(t3.total,axis="index")
t3['total1'] = \
        t3[type2_list].sum(axis=1)
for i in type2_list:
    t3[i+"_new"] = t3[i].div(t3.total1,axis="index")
# In[ ]:
for i in modcol:
    df[i] = t3[i+"_new"]

#continuousvaluedcols: All except[gender, race, race o, samerace, field, decision]
labels = range(2)
list_col =  list(df.columns.values)
list_col_full = list(df.columns.values)
list_nobins = ['gender','samerace','decision']
print_list = list_col_full
for i in ['gender','samerace','decision']:
    list_col.remove(i)
list_col_part1=[modcol]
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
t4 = df[list_col_full] 
for i in list_col_full:
    if i in list_nobins:
        t4[i+"_bin"] =  t4[i]
    elif i in list_col_part1:
        t4[i+"_bin"] = pd.cut(t4[i], bins = 2,labels=labels)
    elif i in list_col_age:
        t4[i+"_bin"] = pd.cut(t4[i],bins = 2,labels=labels)
    elif i in list_Range_10:
        t4[i+"_bin"] = pd.cut(t4[i], bins = 2,labels=labels)
    elif i in list_correlate:
        t4[i+"_bin"] = pd.cut(t4[i], bins = 2,labels=labels)
    else:
        t4[i+"_bin"] = pd.cut(t4[i], bins = 2, labels=labels)  
for i in list_col:
    df[i] = t4[i+"_bin"]
data_test = df.sample(random_state=47,frac=0.2)
data_test.to_csv('testSet.csv',index = False)
data_train = df[~df.index.isin(data_test.index)]
data_train.to_csv('trainingSet.csv',index = False)
