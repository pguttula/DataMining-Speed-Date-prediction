
# coding: utf-8

# In[3]:


import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os


# In[54]:


data = pd.read_csv('dating-full.csv')
def rem_col(df):
    df['string_race'] = 0
    df['count_race'] = 0
    if str(df['race'])[0] == str("\'") and str(df['race'])[-1] == str("\'"):
        df['string_race'] = str(df['race'])[1:-1]
        df['count_race'] = 1
    else:
        df['string_race'] = str(df['race'])
        df['count_race'] = 0
    df['string_race_o'] = 0
    
    df['count_race_o'] = 0
    if str(df['race_o'])[0] == str("\'") and str(df['race_o'])[-1] == str("\'"):
        df['string_race_o'] = str(df['race_o'])[1:-1]
        df['count_race_o'] = 1
    else:
        df['string_race_o'] = str(df['race_o'])
        df['count_race_o'] = 0
    df['string_field'] = 0
    
    df['count_field'] = 0
    if str(df['field'])[0] == str("\'") and str(df['field'])[-1] == str("\'"):
        df['string_field'] = str(df['field'])[1:-1]
        df['count_field'] = 1
    else:
        df['string_field'] = str(df['field'])
        df['count_field'] = 0
    return df

t = data[['race','race_o','field']].apply(rem_col,axis=1)
data['race'] = t['string_race']
data['race_o'] = t['string_race_o']
data['field'] = t['string_field']

print("Total number of preprocessed/changes cells is", sum(t['count_race'])+sum(t['count_race_o'])+sum(t['count_field']))


# In[77]:


data[['age','attractive','funny_partner']].mean()


# In[79]:


data[data['gender']=='male']


# In[ ]:


data.to_csv('dating.csv')

