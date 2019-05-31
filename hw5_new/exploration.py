import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
import random
import matplotlib.pyplot as plt
np.random.seed(0)
#embed = pd.read_csv('digits-embedding.csv',header=None)
from PIL import Image
data = pd.read_csv('digits-raw.csv',header=None)
colors = ['aquamarine', 'g', 'r', 'coral', 'm', 'khaki', 'turquoise', 'navy', 'yellowgreen', 'pink']
embed = pd.read_csv('digits-embedding.csv', header=None, names=['id', 'label', 'X', 'Y'])
grouped = data.groupby(data.iloc[:,1])
df = grouped.apply(lambda data: data.sample(n=1, replace=True))

def show_digit(df,i):
    x = np.array(df.iloc[i,2:])
    im = x.reshape(28, 28)
    plt.gray()
    plt.imshow(im)
    plt.savefig('Fig-'+str(i)+'.png', dpi=100)
for i in range(10):
    show_digit(df,i)

colors = ['aquamarine', 'g', 'r', 'coral', 'm', 'khaki', 'turquoise', 'navy', 'yellowgreen', 'pink']
embed = pd.read_csv('digits-embedding.csv', header=None, names=['id', 'label', 'X', 'Y'])

def visualizeFeatures(data):
    examples = np.random.randint(0, len(data), size=1000)
    Examples = data.iloc[examples]
    numLabels = len(data['label'].unique())
    Examples = Examples.groupby('label')
    fig, ax = plt.subplots()
    for i in range(numLabels):
        X = Examples.get_group(i).X
        Y = Examples.get_group(i).Y
        ax.scatter(X, Y, s = 20,c=colors[i], label=i)
    ax.legend()
    plt.xlabel("image-embedding feature_1",fontsize=12,color='purple')
    plt.ylabel("image-embedding feature_2",fontsize=12,color='purple')
    plt.savefig('Fig-cluster.png', dpi=80)
    plt.close()
visualizeFeatures(embed)
