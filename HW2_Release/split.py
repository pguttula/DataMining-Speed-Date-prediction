import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

data = pd.read_csv('dating-binned.csv')
data_test = data.sample(random_state=47,frac=0.2)
#print data_test
data_test.to_csv('testSet.csv',index = False)
data_train = data[~data.index.isin(data_test.index)]
#print data_train
data_train.to_csv('trainingSet.csv',index = False)
