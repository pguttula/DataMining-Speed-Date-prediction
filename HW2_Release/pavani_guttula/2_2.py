import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
inputfile = sys.argv[1]

data = pd.read_csv(inputfile)
temp = data[['attractive_partner','sincere_partner','intelligence_parter','funny_partner','ambition_partner','shared_interests_partner','decision']]
t_attractive = temp['attractive_partner'].unique()
t_sincere = temp['sincere_partner'].unique()
t_intelligence = temp['intelligence_parter'].unique()
t_funny = temp['funny_partner'].unique()
t_ambition =temp['ambition_partner'].unique()
t_shared_interests =temp['shared_interests_partner'].unique()

dict_attractive={}

for i in t_attractive:
    dict_attractive[i]= temp[(temp['attractive_partner'] == i) &
        (temp['decision']==1)].count().div(temp[temp['attractive_partner']\
            == i].count(),axis = 'index')[0]
key_attractive=[]
for key in dict_attractive:
    key_attractive.append(key)
value_attractive=[]
for value in dict_attractive:
    value_attractive.append(dict_attractive[value])
dict_sincere={}
for i in t_sincere:
    dict_sincere[i]= temp[(temp['sincere_partner'] == i) & (temp['decision']==1)].count().div(temp[temp['sincere_partner'] == i].count(),axis = 'index')[0]
key_sincere=[]
for key in dict_sincere:
    key_sincere.append(key)
value_sincere=[]
for value in dict_sincere:
    value_sincere.append(dict_sincere[value])
dict_intelligence={}

for i in t_intelligence:
    dict_intelligence[i]= temp[(temp['intelligence_parter'] == i) & (temp['decision']==1)].count().div(temp[temp['intelligence_parter'] == i].count(),axis = 'index')[0]
key_intelligence=[]
for key in dict_intelligence:
    key_intelligence.append(key)
value_intelligence=[]
for value in dict_intelligence:
    value_intelligence.append(dict_intelligence[value])

dict_funny={}
for i in t_funny:
    dict_funny[i]= temp[(temp['funny_partner'] == i) & (temp['decision']==1)].count().div(temp[temp['funny_partner'] == i].count(),axis = 'index')[0]
key_funny=[]
for key in dict_funny:
    key_funny.append(key)
value_funny=[]
for value in dict_funny:
    value_funny.append(dict_funny[value])

dict_ambition={}

for i in t_ambition:
    dict_ambition[i]= temp[(temp['ambition_partner'] == i) & (temp['decision']==1)].count().div(temp[temp['ambition_partner'] == i].count(),axis = 'index')[0]
key_ambition=[]
for key in dict_ambition:
    key_ambition.append(key)
value_ambition=[]
for value in dict_ambition:
    value_ambition.append(dict_ambition[value])

dict_shared_interests={}

for i in t_shared_interests:
    dict_shared_interests[i]= temp[(temp['shared_interests_partner'] == i) & (temp['decision']==1)].count().div(temp[temp['shared_interests_partner'] == i].count(),axis = 'index')[0]
key_shared_interests=[]
for key in dict_shared_interests:
    key_shared_interests.append(key)
value_shared_interests=[]
for value in dict_shared_interests:
    value_shared_interests.append(dict_shared_interests[value])
#scatter plots

x = key_attractive
y = value_attractive
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('attractive_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_attractive.png')
plt.close()

x = key_sincere
y = value_sincere
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('sincere_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_sincere.png')
plt.close()

x = key_intelligence
y = value_intelligence
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('intelligence_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_intelligence.png')
plt.close()

x = key_funny
y = value_funny
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('funny_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_funny.png')
plt.close()

x = key_ambition
y = value_ambition
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('ambition_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_ambition.png')
plt.close()

x = key_shared_interests
y = value_shared_interests
area = np.pi*10
plt.scatter(x, y, s=area, c='g',alpha=0.5)
plt.title('Rating Of Partner From Participant',fontsize=24,color='purple')
plt.xlabel('shared_interests_partner rating',fontsize=14,color='purple')
plt.ylabel('sucess rate',fontsize=14,color='purple')
#plt.show()
#plt.savefig('Fig-2_2_shared_interests.png')
plt.close()


