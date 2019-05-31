import scipy as sp
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import random
import os
data_train = sys.argv[1]
data_test = sys.argv[2]
train = pd.read_csv(data_train)
test = pd.read_csv(data_test)
modelIdx = sys.argv[-1]

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
def getfinal_pred(dict_pred,num_of_trees):
    predict = 0
    df = pd.DataFrame().astype(float)
    for i in range(num_of_trees):
        df[str(i)+ "_predict"] = dict_pred[i]
    predict = df.mode(axis=1)
    return predict[0]

def build_tree(df,max_depth,min_elements,curr_depth):
    treedata = get_split(df)
    if curr_depth < max_depth: 
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
    #print instance[tree_dup[tree_dup.keys()[3]]]
    if (instance[tree_dup[tree_dup.keys()[3]]]) == 0:
        if type(tree_dup[tree_dup.keys()[0]]) is dict:
            predict = traverse_dict(instance,tree_dup[tree_dup.keys()[0]],predict)
        elif type(tree_dup[tree_dup.keys()[0]]) is not dict:
            predict = tree_dup[tree_dup.keys()[0]]
            #print "dup", tree_dup[tree_dup.keys()[0]]
            #predict =  tree_dup['lchild']
            return predict
    elif (instance[tree_dup[tree_dup.keys()[3]]]) == 1:
        if type(tree_dup[tree_dup.keys()[1]]) is dict:
            predict = traverse_dict(instance,tree_dup[tree_dup.keys()[1]],predict)
        elif type(tree_dup[tree_dup.keys()[1]]) is not dict:
            #predict =  tree_dup['rchild']
            predict = tree_dup[tree_dup.keys()[1]]
           # print "dup", tree_dup[tree_dup.keys()[1]]
            return predict
    #print predict
    return predict
def printaccuracy(decision,predict):
    accuracy = abs(decision-predict)
    accuracy =  round(float((accuracy==0).sum())/((accuracy==0).sum()+(accuracy==1).sum()),2)
    return accuracy   
def predict_dec(df,tree_dup):
    #predict = 1
    predict = traverse_dict(df,tree_dup,1)
    return predict
def gini_gain_RF(df):
    dict_gini={}
    main_gini = 1- (((float(df.query('decision == 0').decision.count()))/float(len(df)))**2+ \
                     (float(df.query('decision == 1').decision.count())/float(len(df)))**2)
    full_features = list(df.keys()[:-1]) 
    choices = random.sample(range(len(full_features)), int(np.sqrt(len(df.keys()))))
    features = [full_features[i] for i in choices]
    valcounts= df.apply(pd.Series.value_counts)/len(df)
    #main_gini = 1- (((float(df.query('decision == 0').decision.count()))/float(len(df)))**2+ \
    #                 (float(df.query('decision == 1').decision.count())/float(len(df)))**2)
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
            #continue
            gini_feature1 = 0
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

def bagging(train, test,max_depth):
    print "BT", max_depth
    tree_bag ={}
    train_pred={}
    test_pred ={}
    predict_train = 0
    predict_test = 0
    training_final = train
    test_final = test
    num_of_trees = 30
    for i in range(30):
        print "BT",i
        train_bag = train.sample(frac = 1.0, replace = True)
        train_bag.reset_index(drop = True)
        tree = build_tree(train_bag,max_depth,50,1)
        train_pred[i] = train.apply(predict_dec,axis=1,args = (tree,))
        test_pred[i] = test.apply(predict_dec,axis=1,args = (tree,))
    return train_pred,test_pred

if int(modelIdx) == 1:
    tree_dup = build_tree(train,8,50,1)
    t = test.apply(predict_dec,axis=1,args = (tree_dup,))
    accuracy_test = printaccuracy(test['decision'],t)
    t1 = train.apply(predict_dec,axis=1,args = (tree_dup,))
    accuracy_train = printaccuracy(train['decision'],t1)
    print "Training Accuracy DT",accuracy_train
    print "Test Accuracy DT:",accuracy_test
elif int(modelIdx) == 2:
    train_pred,test_pred = bagging(train, test,8)
    predict_train = getfinal_pred(train_pred,30)
    predict_test = getfinal_pred(test_pred,30)
    accuracy_test = printaccuracy(test['decision'],predict_test)
    accuracy_train = printaccuracy(train['decision'],predict_train)
    print "Training Accuracy BT:",accuracy_train
    print "Test Accuracy BT:",accuracy_test
elif int(modelIdx) == 3:
    train_pred,test_pred = randomforest(train, test)
    predict_train = getfinal_pred(train_pred,30)
    predict_test = getfinal_pred(test_pred,30)
    accuracy_test = printaccuracy(test['decision'],predict_test)
    accuracy_train = printaccuracy(train['decision'],predict_train)
    print "Training Accuracy RF:",accuracy_train
    print "Test Accuracy RF:",accuracy_test

