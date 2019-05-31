import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os

inputfile = sys.argv[1]
outputtrain = sys.argv[2]
outputtest = sys.argv[3]
data = pd.read_csv(inputfile)
data_test = data.sample(random_state=47,frac=0.2)
data_test.to_csv(outputtest,index = False)
data_train = data[~data.index.isin(data_test.index)]
data_train.to_csv(outputtrain,index = False)
