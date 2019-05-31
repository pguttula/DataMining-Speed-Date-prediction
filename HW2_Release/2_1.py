import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os
import matplotlib.pyplot as plt

data = pd.read_csv('dating.csv')
data_m = data[data['gender'] == 1]
data_f = data[data['gender']== 0]
tupm=()
tupf=()
tm = data_m[['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']]
for i in ['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']:
    tupm = tupm + (round(tm[i].mean(),2),)
    
tf = data_f[['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']]
for i in ['attractive_important','sincere_important','intelligence_important',\
        'funny_important','ambition_important','shared_interests_important']:
    tupf = tupf + (round(tf[i].mean(),2),)

n_groups = 6
#fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width =0.35
opacity = 0.6
plt.figure(figsize=(12, 10))
rects1 = plt.bar(index, tupf, bar_width,alpha=opacity,color='tab:pink',label='Female')
rects2 = plt.bar(index + bar_width, tupm, bar_width,alpha=opacity,color='tab:blue',label='Male')
plt.title('Preference Scores Of Participant -Male vs Female',fontsize=28)
plt.xlabel('Preference Scores Of Participant',fontsize=16)
plt.xticks(index + 0.5*bar_width, ['attractive', 'sincere', 'intelligence', 'funny','ambition',\
'shared_interests'],fontsize=16)
plt.ylabel('Mean',fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('Fig-2_1.png', dpi=100)
