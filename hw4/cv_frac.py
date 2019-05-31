import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
import random
from timeit import default_timer as timer
train = pd.read_csv('trainingSet.csv')
test = pd.read_csv('testSet.csv')
train = train.sample(random_state=18,frac=1)
S1,S2,S3,S4,S5,S6,S7,S8,S9,S10 = train[:520],train[520:1040],train[1040:1560],train[1560:2080],\
train[2080:2600],train[2600:3120],train[3120:3640],train[3640:4160],train[4160:4680],train[4680:5200]
sets = [[]]
t_fracset = [0.05, 0.075, 0.1, 0.15, 0.2]
idx_sets = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10]
depth = 8

def gini_gain(df):
    dict_gini={}
    features = list(df.keys()[:-1]) 
    valcounts= df.apply(pd.Series.value_counts)/len(df)
    main_gini = 1- (((float(df.query('decision == 0').decision.count()))/float(len(df['decision'])))**2+ \
                     (float(df.query('decision == 1').decision.count())/float(len(df['decision'])))**2)
    values = [0,1]
    valcounts= df.apply(pd.Series.value_counts)/len(df)
    for feature in features:
        df_feaval0 = df[df[feature]==0]
        df_feaval1 = df[df[feature]==1]
        if(len(df_feaval0) ==0):
            gini_feature0 = 0
            #continue
        else:
            gini_feature0 = 1- (((float(df_feaval0.query('decision == 0').decision.count()))/float(len(df_feaval0)))**2+ \
                         (float(df_feaval0.query('decision == 1').decision.count())/float(len(df_feaval0)))**2)
        if(len(df_feaval1) ==0):
            gini_feature1 = 0
        else:
            gini_feature1 = 1- (((float(df_feaval1.query('decision == 0').decision.count()))/float(len(df_feaval1)))**2+ \
                         (float(df_feaval1.query('decision == 1').decision.count())/float(len(df_feaval1)))**2)
        fea_gain = main_gini - (valcounts[feature][0]*gini_feature0 + valcounts[feature][1]*gini_feature1)
        dict_gini[feature] = fea_gain
    root = max(dict_gini, key=dict_gini.get)
    ginival = dict_gini[root]
    return root,ginival
def get_split(df):
    root,ginival = gini_gain(df)
    lchild, rchild = df[df[root]==0],df[df[root]==1]
    return {'lchild':lchild, 'rchild':rchild,'gini':ginival,'root':root}
def topvalues(df):
    yes = float(df.query('decision == 1').decision.count())
    no = float(df.query('decision == 0').decision.count())
    if max(yes,no) == yes:
        return 1
    else:
        return 0
def build_tree(df,max_depth,min_elements,curr_depth):
    treedata = get_split(df)
    if curr_depth <8: 
        if len(treedata['lchild']) > min_elements:
            remtree = treedata['lchild']
            remtree = remtree.drop(treedata['root'], 1)
            treedata['lchild'] = build_tree(remtree,max_depth,min_elements,curr_depth+1)
        else:
            treedata['lchild'] = topvalues(treedata['lchild'])
        if len(treedata['rchild']) > min_elements:
            remtree1 = treedata['rchild']
            remtree1 = remtree1.drop(treedata['root'], 1)
            treedata['rchild'] = build_tree(remtree1,max_depth,min_elements,curr_depth+1)
        else:
            treedata['rchild'] = topvalues(treedata['rchild'])
    else:
        treedata['lchild'] = topvalues(treedata['lchild'])
        treedata['rchild'] = topvalues(treedata['rchild'])
    return treedata
def traverse_dict(instance,tree_dup,predict): 
    if (instance[tree_dup[tree_dup.keys()[3]]]) == 0:
        if type(tree_dup[tree_dup.keys()[0]]) is dict:
            predict = traverse_dict(instance,tree_dup[tree_dup.keys()[0]],predict)
        elif type(tree_dup[tree_dup.keys()[0]]) is not dict:
            #predict = tree_dup[tree_dup.keys()[0]]
            predict =  tree_dup['lchild']
            return predict
    elif (instance[tree_dup[tree_dup.keys()[3]]]) == 1:
        if type(tree_dup[tree_dup.keys()[1]]) is dict:
            predict = traverse_dict(instance,tree_dup[tree_dup.keys()[1]],predict)
        elif type(tree_dup[tree_dup.keys()[1]]) is not dict:
            predict =  tree_dup['rchild']
           # print tree_dup
            return predict
    #print predict
    return predict
def getfinal_pred(dict_pred,num_of_trees):
    predict = 0
    df = pd.DataFrame().astype(float)
    for i in range(num_of_trees):
        df[str(i)+ "_predict"] = dict_pred[i]
    predict = df.mode(axis=1)
    return predict[0]

def bagging(train, test):
    tree_bag ={}
    train_pred={}
    test_pred ={}
    predict_train = 0
    predict_test = 0
    training_final = train
    test_final = test
    num_of_trees = 30
    for i in range(num_of_trees):
        train_bag = train.sample(frac = 1.0, replace = True)
        train_bag.reset_index(drop = True)
        tree = build_tree(train_bag,8,50,1)
        train_pred[i] = train.apply(predict_dec,axis=1,args = (tree,))
        test_pred[i] = test.apply(predict_dec,axis=1,args = (tree,))
    return train_pred,test_pred
def gini_gain_RF(df):
    dict_gini={}
    full_features = list(df.keys()[:-1]) 
    choices = random.sample(range(len(full_features)), int(np.sqrt(len(df.keys()))))
    features = [full_features[i] for i in choices]
    valcounts= df.apply(pd.Series.value_counts)/len(df)
    main_gini = 1- (((float(df.query('decision == 0').decision.count()))/float(len(df['decision'])))**2+ \
                     (float(df.query('decision == 1').decision.count())/float(len(df['decision'])))**2)
    values = [0,1]
    valcounts= df.apply(pd.Series.value_counts)/len(df)
    for feature in features:
        df_feaval0 = df[df[feature]==0]
        df_feaval1 = df[df[feature]==1]
        if(len(df_feaval0) ==0):
            #gini_feature0 = 1
            continue
        else:
            gini_feature0 = 1- (((float(df_feaval0.query('decision == 0').decision.count()))/float(len(df_feaval0)))**2+ \
                         (float(df_feaval0.query('decision == 1').decision.count())/float(len(df_feaval0)))**2)
        if(len(df_feaval1) ==0):
            continue
            #gini_feature1 = 1
        else:
            gini_feature1 = 1- (((float(df_feaval1.query('decision == 0').decision.count()))/float(len(df_feaval1)))**2+ \
                         (float(df_feaval1.query('decision == 1').decision.count())/float(len(df_feaval1)))**2)
        fea_gain = main_gini - (valcounts[feature][0]*gini_feature0 + valcounts[feature][1]*gini_feature1)
        dict_gini[feature] = fea_gain
    root = max(dict_gini, key=dict_gini.get)
    ginival = dict_gini[root]
    return root,ginival
def get_split_RF(df):
    root,ginival = gini_gain_RF(df)
    lchild, rchild = df[df[root]==0],df[df[root]==1]
    return {'lchild':lchild, 'rchild':rchild,'gini':ginival,'root':root}

def build_tree_RF(df,max_depth,min_elements,curr_depth):
    treedata = get_split_RF(df)
    if curr_depth <8: 
        if len(treedata['lchild']) > min_elements:
            remtree = treedata['lchild']
            remtree = remtree.drop(treedata['root'], 1)
            treedata['lchild'] = build_tree_RF(remtree,max_depth,min_elements,curr_depth+1)
        else:
            treedata['lchild'] = topvalues(treedata['lchild'])
        if len(treedata['rchild']) > min_elements:
            remtree1 = treedata['rchild']
            remtree1 = remtree1.drop(treedata['root'], 1)
            treedata['rchild'] = build_tree_RF(remtree1,max_depth,min_elements,curr_depth+1)
        else:
            treedata['rchild'] = topvalues(treedata['rchild'])
    else:
        treedata['lchild'] = topvalues(treedata['lchild'])
        treedata['rchild'] = topvalues(treedata['rchild'])
    return treedata
def randomforest(train, test):
    tree_bag ={}
    train_pred={}
    test_pred ={}
    predict_train = 0
    predict_test = 0
    training_final = train
    test_final = test
    num_of_trees = 30
    for i in range(num_of_trees):
        train_bag = train.sample(frac = 1.0, replace = True)
        train_bag.reset_index(drop = True)
        tree = build_tree_RF(train_bag,8,50,1)
        train_pred[i] = train.apply(predict_dec,axis=1,args = (tree,))
        test_pred[i] = test.apply(predict_dec,axis=1,args = (tree,))
    return train_pred,test_pred

def predict_dec(df,tree_dup):
    predict = 0
    predict = traverse_dict(df,tree_dup,0)
    return predict
def printaccuracy(decision,predict):
    accuracy = abs(decision-predict)
    accuracy =  round(float((accuracy==0).sum())/((accuracy==0).sum()+(accuracy==1).sum()),2)
    return accuracy 

accuracy_DT={}
standard_error_DT =[]
accuracy_BT ={}
standard_error_BT =[]
accuracy_RF ={}
standard_error_RF = []
for t_frac in t_fracset:
    idx_itr =0
    print t_frac 
    cross_accuracy_DT=[]
    cross_accuracy_BT =[]
    cross_accuracy_RF=[]
    for set in idx_sets:
        test = set
        if(idx_itr>0 and idx_itr<9):
            set1 = pd.concat(idx_sets[0:idx_itr],ignore_index=True)
            set2 = pd.concat(idx_sets[idx_itr+1:],ignore_index=True)
            S_C = pd.concat([set1, set2],ignore_index=True)
        elif(idx_itr==0):
            set2 = pd.concat(idx_sets[idx_itr+1:],ignore_index=True)
            S_C = set2
        elif(idx_itr==len(idx_sets)-1):
            set1 = pd.concat(idx_sets[0:idx_itr],ignore_index=True)
            S_C = set1
        else:
            print "Debug your code!"
        idx_itr +=1
        train = S_C.sample(random_state=32,frac=t_frac)
        tree = build_tree(train,depth,50,1)
        predict_DT = test.apply(predict_dec,axis=1,args = (tree,))
        test_accuracy_DT = printaccuracy(test['decision'],predict_DT)
        cross_accuracy_DT.append(test_accuracy_DT)
        train_pred_BT,test_pred_BT = bagging(train, test)
        predict_test_BT = getfinal_pred(test_pred_BT,30)
        test_accuracy_BT = printaccuracy(test['decision'],predict_test_BT)
        cross_accuracy_BT.append(test_accuracy_BT)
        train_pred_RF,test_pred_RF = randomforest(train, test)
        predict_test_RF = getfinal_pred(test_pred_RF,30)
        test_accuracy_RF = printaccuracy(test['decision'],predict_test_RF)
        cross_accuracy_RF.append(test_accuracy_RF)
        
    accuracy_DT[t_frac] = round(np.float(np.mean(cross_accuracy_DT)),2)
    standard_error_DT.append(round(np.std(cross_accuracy_DT,dtype=np.float32)/np.sqrt(len(cross_accuracy_DT)),3))
    accuracy_BT[t_frac] = round(np.float(np.mean(cross_accuracy_BT)),2)
    standard_error_BT.append(round(np.std(cross_accuracy_BT,dtype=np.float32)/np.sqrt(len(cross_accuracy_BT)),3))
    accuracy_RF[t_frac] = round(np.float(np.mean(cross_accuracy_RF)),2)
    standard_error_RF.append(round(np.std(cross_accuracy_RF,dtype=np.float32)/np.sqrt(len(cross_accuracy_RF)),3))
print "DT accuracies: ", accuracy_DT
print "DT standard errors",standard_error_DT
print "BT accuracies: ", accuracy_BT
print "BT standard errors",standard_error_BT
print "RF accuracies: ", accuracy_RF
print "RF standard errors",standard_error_RF
