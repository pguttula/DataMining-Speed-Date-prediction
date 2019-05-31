import sys
import pandas as pd
import numpy as np
import scipy as sp
import warnings
import pprint
import sys
import time
import random
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import   linkage, single, complete, average, dendrogram, cut_tree, inconsistent
from collections import Counter
#from operator import itemgetter
import matplotlib.pyplot as plt
import copy
from PIL import Image


np.random.seed(0)
colors = ['aquamarine', 'g', 'r', 'coral', 'm', 'khaki', 'turquoise', 'navy', 'yellowgreen', 'pink']

embed = 'digits-embedding.csv'
embeddedData = pd.read_csv(embed, header=None, names=['id', 'label', 'featureX', 'featureY'])

embedded = embeddedData.groupby(embeddedData.iloc[:, 1]).apply(lambda x: x.sample(n=10)).reset_index(drop=True)


def hierarchicalClustering(data, method):
    dataFeatures = data.iloc[:, [2, 3]]
    distances = pdist(dataFeatures)
    if method == 1:
        clusters = single(distances)
    elif method == 2:
        clusters = complete(distances)
    else:
        clusters = average(distances)

    dendrogram(clusters)
    plt.savefig('Fig-Dendogram'+str(method)+'.png', dpi=100)
    plt.close()

def getClusters(z,clusters,level):
    numClusters = len(clusters) 
    for i in range(level):
        cluster1 = z[i][0]
        cluster2 = z[i][1]
        clusters[clusters == cluster1] = numClusters
        clusters[clusters == cluster2] = numClusters
        numClusters += 1           
    uniqueClusters = list(set(clusters))    
    for i in range(len(uniqueClusters)):
        clusters[clusters == uniqueClusters[i]] = i
    return clusters

def hierarchicalClusteringPartitions(data, method, K):
  dataFeatures = data.iloc[:, [2, 3]]
  distances = pdist(dataFeatures)
  if method == 1:
    z = single(distances)
  elif method == 2:
    z = complete(distances)
  else:
    z = average(distances)
  clusters = np.array(range(len(data)))
  level = len(data) - K
  assignedClusters = getClusters(z,clusters,level)  
  data['cluster'] = assignedClusters
  clusterAverages = {}    
  uniqueAssignedClusters = list(set(assignedClusters))    
  for cluster in range(len(uniqueAssignedClusters)):  
    currentCluster = uniqueAssignedClusters[cluster]  
    clusterData = data[data['cluster'] == currentCluster]        
    clusterAverages[cluster] = clusterData.iloc[:,[2,3]].mean()         
  distanceToCentroid = [0]*len(data)       
  i = 0
  for index, row in data.iterrows():
    currentCluster = row[4]
    distanceToCentroid[i] = np.linalg.norm((np.array(row[2:4]))-(np.array(clusterAverages[currentCluster]))) 
    i = i + 1                   
  data['distanceToCentroid'] = distanceToCentroid
  # print(data)
  return data

def withinClusterSSD(data):
  data['squaredDistance'] = data['distanceToCentroid'] * data['distanceToCentroid']
  return data['squaredDistance'].sum()    

# silhouette coeeficient
def silhouetteCoefficient(data):
  dataFeatures = data.iloc[:, [2,3]]
  distances = squareform(pdist(dataFeatures))
  A = np.zeros(len(data))
  B = np.zeros(len(data))
  S = np.zeros(len(data))
  for i in range(len(data)):
    cluster = data.loc[i, 'cluster']
    withincluster = data[data['cluster'] == cluster].index.tolist()
    outsidecluster = data[data['cluster'] != cluster].index.tolist()
    A[i] = np.mean(distances[i][withincluster])
    B[i] = np.mean(distances[i][outsidecluster])
    S[i] = (B[i]-A[i])/max(A[i], B[i])
  return np.mean(S)

#hierarchicalClustering(embedded,1)
#hierarchicalClustering(embedded,2)
#hierarchicalClustering(embedded,3)

wcSSDMethod1 = list()
wcSSDMethod2 = list()
wcSSDMethod3 = list()
SCMethod1 = list()
SCMethod2 = list()
SCMethod3 = list()

for K in [2, 4, 8, 16, 32]:
  for method in [1, 2, 3]:
    data = hierarchicalClusteringPartitions(embedded, method, K)
    wcSSD = withinClusterSSD(data)
    SC = silhouetteCoefficient(data)
    if method == 1:
      wcSSDMethod1.append(wcSSD)
      SCMethod1.append(SC)
    elif method == 2:
      wcSSDMethod2.append(wcSSD)
      SCMethod2.append(SC)
    else:
      wcSSDMethod3.append(wcSSD)
      SCMethod3.append(SC)
    print 'WC-SSD ', K, ', linkage method', method, ':', wcSSD
    print 'SC ', K, ', linkage method', method, ':',SC

x = [i for i in [2, 4, 8, 16, 32]]
plt.figure()
plt.plot(x, wcSSDMethod1, label='WC-SSD Single Linkage',color = 'navy')
plt.plot(x, wcSSDMethod2, label='WC-SSD Complete Linkage',color = 'coral')
plt.plot(x, wcSSDMethod3, label='WC-SSD Average Linkage')
plt.title("K vs WC-SSD for various methods",fontsize=20,color='purple')
plt.xlabel('K',fontsize=14,color='purple')
plt.ylabel('WC-SSD',fontsize=14,color='purple')
plt.legend()
#plt.show()
plt.close()

plt.figure()
plt.plot(x, SCMethod1, label='SC Single Linkage',color = 'navy')
plt.plot(x, SCMethod2, label='SC Complete Linkage',color = 'coral')
plt.plot(x, SCMethod3, label='SC Average Linkage')
plt.title("K vs SC for various methods",fontsize=20,color='purple')
plt.xlabel('K',fontsize=14,color='purple')
plt.ylabel('SC',fontsize=14,color='purple')
plt.legend()
#plt.show()
plt.close()
