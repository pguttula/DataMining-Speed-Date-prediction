import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
import random
import time
import copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist, squareform
data = pd.read_csv('digits-embedding.csv',header=None)
itr_count = 50

def within_cluster_sum_of_squares(cluster_data):
    squared = cluster_data.distance_to_centroid * cluster_data.distance_to_centroid
    return squared.sum()

def silhouetteCoefficient(cluster_data):
    Features = data.iloc[:, [2, 3]]
    distbtwnallpoints = squareform(pdist(Features))
    A = np.zeros(len(cluster_data))
    B = np.zeros(len(cluster_data))
    Si = np.zeros(len(cluster_data))
    for i in range(len(cluster_data)):
        cluster = cluster_data.iloc[i]['min_index']
        samecluster = cluster_data[cluster_data['min_index'] == cluster].index.tolist()
        othercluster = cluster_data[cluster_data['min_index'] != cluster].index.tolist()
        A[i] = np.mean(distbtwnallpoints[i][samecluster])
        B[i] = np.mean(distbtwnallpoints[i][othercluster])
        Si[i] = (B[i]-A[i])/max(A[i], B[i])
    return np.mean(Si)

def nmi(cluster_data):
    C = len(cluster_data['label'].unique())
    G = len(cluster_data['min_index'].unique())
    H_C = 0
    H_G = 0
    I_CG = 0
    for i in range(C):
        P_C = float(len(cluster_data[cluster_data['label'] == i]))/float(len(cluster_data))
        H_C += (-1*P_C*np.log(P_C+0.00001))
        for cluster in range(G):
            P_G = float(len(cluster_data[cluster_data['min_index'] == cluster]))/float(len(cluster_data))
            if i == 0:
                H_G += (-1*P_G*np.log(P_G+0.00001))
            P_CG = float(len(cluster_data[(cluster_data['label'] == i) & \
                                          (cluster_data['min_index'] == cluster)]))/ float(len(cluster_data))
            I_CG += (P_CG*np.log((P_CG/(P_C*P_G))+0.00001))
    nmi = I_CG/(H_C+H_G)
    return nmi
def kmeans(data,itr_count,K,seed_val):
    np.random.seed(seed_val)
    cluster_data = data.iloc[:, [2, 3]]
    random_points =  np.random.randint(0,cluster_data.shape[0], size=K)
    centroids = np.array(cluster_data.ix[random_points])
    centroids_prev = np.zeros((K,2))
    euclidean = np.zeros((len(cluster_data),len(centroids)))
    arr_data = np.array(cluster_data)
    itr = 0
    for i in range(itr_count):
        for i in range(len(cluster_data)):
            for j in range(len(centroids)):
                euclidean[i][j] =  np.linalg.norm(arr_data[i]- centroids[j])
        min_index = np.argmin(euclidean,axis=1)
        cluster_data['min_index'] = min_index
        cluster_data['distance_to_centroid'] = euclidean.min(axis=1)
        if itr < 49:
            for i in range(K):
                temp = cluster_data[cluster_data['min_index']== i]
                temp1 =  temp[[2,3]]
                centroids[i] = np.mean(temp1)   
        if np.array_equal(centroids_prev,centroids) is True:
            itr = itr +1
            break
        else:
            centroids_prev = copy.deepcopy(centroids) 
            itr = itr +1
    cluster_data['label'] = data[1]
    cluster_data['image_id'] = data[0]
    WC_SSD = within_cluster_sum_of_squares(cluster_data)
    SC = silhouetteCoefficient(cluster_data)
    NMI = nmi(cluster_data)
    return WC_SSD,SC, NMI

def kloop(data,seed_val):
    for i in [2,4,8,16,32]:
        print "K-VAL",i
        WC_SSD,SC, NMI = kmeans(data,itr_count,i,seed_val)
        print "WC-SSD:",WC_SSD
        print "SC:",SC
        print "NMI:",NMI
print "Total dataset"
for i in range(10):
    print i

    kloop(data,i) 
