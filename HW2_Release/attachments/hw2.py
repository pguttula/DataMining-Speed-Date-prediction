import pandas as pd
import string
import numpy as np

def nbc_bow(words,training_set,test_set):
    #create one long vector that has all the unique words:
    allWords = []
    for i in range(0,len(training_set)):
        allWords.extend(set(training_set.review_text.iloc[i].split()))
    allWords = pd.DataFrame.from_dict( {'words' : allWords})
    del i
    allWords['count'] = allWords.groupby('words')['words'].transform('count')
    unique_words = allWords.drop_duplicates()
    unique_words.sort_values(by = 'count',ascending = False , inplace = True)
    unique_words.reset_index(drop = True,inplace = True)
    #Create the table of feature sets:
    feature_set = unique_words.iloc[100:100+words]
    
    for i in range(0,len(feature_set)):
        col = np.zeros(len(training_set))
        current_word = feature_set.words.iloc[i]
        for k in range(0,len(training_set)):
            if training_set.review_text.iloc[k].split().count(current_word) > 0:
                    col[k] = 1
            else:
                    col[k] = 0
        training_set[current_word] = col
    
    del col,current_word,i,k
    
    probClass1 = float(training_set.rev_class.sum()) / float(len(training_set.rev_class))
    data_positive_reviews = training_set[training_set.rev_class == 1]
    
    for i in range(0,len(feature_set)):
        col = np.zeros(len(data_positive_reviews))
        current_word = feature_set.words.iloc[i]
        for k in range(0,len(data_positive_reviews)):
            if data_positive_reviews.review_text.iloc[k].split().count(current_word) > 0:
                    col[k] = 1
            else:
                    col[k] = 0
        data_positive_reviews[current_word] = col
    
    del col,current_word,i
    
    sum_allWords_positive = data_positive_reviews.drop(labels = ['ind','rev_class','review_text'] , axis = 1).sum().sum()
    prob_given_positive = np.zeros(len(feature_set))
    
    for i in range(0,len(feature_set)):
        current_word = feature_set.words.iloc[i]
        current_word_sum = data_positive_reviews[current_word].sum()
        prob_given_positive[i] = (1 + current_word_sum) / (words + sum_allWords_positive) #Laplace smoothing

    probClass0 = 1 - probClass1
    data_negative_reviews = training_set[training_set.rev_class == 0]
    
    for i in range(0,len(feature_set)):
        col = np.zeros(len(data_negative_reviews))
        current_word = feature_set.words.iloc[i]
        for k in range(0,len(data_negative_reviews)):
            if data_negative_reviews.review_text.iloc[k].split().count(current_word) > 0:
                    col[k] = 1
            else:
                    col[k] = 0
        data_negative_reviews[current_word] = col
    
    del col,current_word,i
    
    sum_allWords_negative = data_negative_reviews.drop(labels = ['ind','rev_class','review_text'] , axis = 1).sum().sum()
    prob_given_negative = np.zeros(len(feature_set))
    
    for i in range(0,len(feature_set)):
        current_word = feature_set.words.iloc[i]
        current_word_sum = data_negative_reviews[current_word].sum()
        prob_given_negative[i] = (1 + current_word_sum) / (words + sum_allWords_negative) #Laplace smoothing

    for i in range(0,len(feature_set)):
        col = np.zeros(len(test))
        current_word = feature_set.words.iloc[i]
        for k in range(0,len(test)):
            if test.review_text.iloc[k].split().count(current_word) > 0:
                    col[k] = 1
            else:
                    col[k] = 0
        test[current_word] = col

    data_positive_probabilities = test_set.drop(labels = ['ind','rev_class','review_text'] , axis = 1).multiply(prob_given_positive,axis = 1)   
    data_positive_probabilities.replace(to_replace = 0, value = 1,inplace = True)
    positive_probability = probClass1 * data_positive_probabilities.product(axis=1)
    data_negative_probabilities = test_set.drop(labels = ['ind','rev_class','review_text'] , axis = 1).multiply(prob_given_negative,axis = 1)   
    data_negative_probabilities.replace(to_replace = 0, value = 1,inplace = True)
    negative_probability = probClass0 * data_negative_probabilities.product(axis=1)
    test_set['positive_probability'] = positive_probability
    test_set['negative_probability'] = negative_probability
    test_set['prediction'] = test_set.apply(lambda x: 1 if x.positive_probability > x.negative_probability else 0, axis = 1)
    test_set['loss'] = test_set.apply(lambda x: 0 if x.prediction == x.rev_class else 1, axis = 1)
    loss_function = float(test_set.loss.sum())*(1.0/float(len(test_set)))
    return loss_function
    

read = []
with open('C:\Users\Vandith P S R\Desktop\yelp_data.csv') as f:
   for l in f:
       read.append(l.strip().split("\t"))  

read = np.asarray(read)        
px2 = read.reshape((-1,3))
rawdata = pd.DataFrame({'ind':px2[:,0],'rev_class':px2[:,1],'review_text':px2[:,2]})

rawdata.review_text = rawdata.review_text.str.lower()
rawdata.review_text = rawdata['review_text'].str.replace('[^\w\s]','')

feature_set_size = 500

training = rawdata.sample(frac = .01)
test = rawdata.drop(training.index)
x = nbc_bow(feature_set_size,training,test)
print x