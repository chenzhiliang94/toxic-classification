'''
Created on 27 Mar 2018
@author: zhi liang
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.classification import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_curve, auc

class_names = ['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate']

all_Comments_Data = pd.read_csv('train.csv', encoding='latin-1').fillna(' ') #

all_Comments = all_Comments_Data['comment_text']

# This is used to create single word/word sequence tf-idf score
# Right now this is only an initialized function object
# See documentation: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word', #this can be changed between word or char
    token_pattern=r'\w{1,}',
    stop_words='english', #this removes all stop words i.e pointless words such as a, the, to etc
    ngram_range=(1, 2), #this indicates we take single words, and all two words sequences - can be changed too
    max_features=10000) #Only considers top 10000 features in the final step, can be changed

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)

c_values = [0.1,10,20,30,40,50,60,70,80,90,100]


for class_name in class_names:
    index = 0
    result = np.zeros(shape=(len(c_values),3))
    for c_value in c_values:
        print(class_name)
        classification = all_Comments_Data[class_name]

        #initialize a logistic regression classifier
        #this part can either be changed into another model OR we can tune the model's parameters
        #see the link to check what parameters can be tuned
        logistic_Classifier = LogisticRegression(C=c_value,class_weight='balanced',solver='sag') #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        #we use 1% as testing comments, and 99% as training comments
        X_train, X_test, y_train, y_test = train_test_split(all_Comments, classification, test_size = 0.1, random_state = 100)

        #transform training set using fit_transform
        X_train = word_vectorizer.fit_transform(X_train)
        #transform the testing comments USING only the training comments' fittings
        X_test  = word_vectorizer.transform(X_test)

        #Pass the attributes and classification into a logistic regression model
        logistic_Classifier.fit(X_train, y_train)


        #predict the outcome
        y_pred = logistic_Classifier.predict(X_test)

        #confusion matrix outcome
        #can someone check what does the matrix mean? I forgot which entry corresponds to TP, TN, FN, FP
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        false_positive = cm[0,1]
        true_positive = cm[1,1]
        false_negative = cm[1,0]
        true_negative = cm[0,0]

        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        
        result[index,:] = [c_value, sensitivity, specificity]
        print(result)
        index += 1
    plt.title(class_name)
    sens, = plt.plot(result[:,0], result[:,1])
    plt.xlabel('c value')
    spec, = plt.plot(result[:,0], result[:,2])
    plt.legend([sens, spec],['sensitivity', 'specificity'], loc='lower right')
    plt.show

