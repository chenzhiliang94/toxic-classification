'''
Created on Apr 6, 2018

@author: zhi liang
'''

'''
Created on 27 Mar 2018
@author: zhi liang
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import confusion_matrix
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
ros = RandomOverSampler(random_state=None)
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
    ngram_range=(2, 4),
    max_features=2500)

countVectorizer_word = CountVectorizer(
    analyzer='word', 
    strip_accents='unicode', 
    ngram_range=(1,1),
    stop_words = 'english',
    max_features = 50000 ,
    token_pattern=r'\w{1,}',
    binary = True
   )
#c_values = [0.1,10,20,30,40,50,60,70,80,90,100]
c_values = [500,1000,1500,2000,4000,6000,8000,10000, 12000,14000,16000,18000,20000]

#for class_name in class_names:

result = np.zeros(shape=(len(c_values),3))
class_names = ['toxic', 'severe_toxic','obscene', 'threat', 'insult', 'identity_hate']
for class_name in class_names:
        
    print(class_name)
    classification = all_Comments_Data[class_name]

    #initialize a logistic regression classifier
    #this part can either be changed into another model OR we can tune the model's parameters
    #see the link to check what parameters can be tuned
    
    X_train, X_test, y_train, y_test = train_test_split(all_Comments, classification, test_size = 0.1, random_state = 100)
    X_train = char_vectorizer.fit_transform(X_train)
    X_test  = char_vectorizer.transform(X_test)
    #X_train, y_train = ros.fit_sample(X_train, y_train)
    model = LogisticRegression(C=1.0,class_weight='balanced')

    #Pass the attributes and classification into a logistic regression model
    model.fit(X_train, y_train)
    
    #predict the outcome
    y_pred = model.predict(X_test)
    y_pred_training = model.predict(X_train)
    #confusion matrix outcome
    #can someone check what does the matrix mean? I forgot which entry corresponds to TP, TN, FN, FP
    cm = confusion_matrix(y_test, y_pred)
    print('test set')
    print(cm)

    false_positive = cm[0,1]
    true_positive = cm[1,1]
    false_negative = cm[1,0]
    true_negative = cm[0,0]

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    cm = confusion_matrix(y_train, y_pred_training)
    print('train set')
    print(cm)

    false_positive = cm[0,1]
    true_positive = cm[1,1]
    false_negative = cm[1,0]
    true_negative = cm[0,0]
    sensitivity_training = true_positive / (true_positive + false_negative)
    

    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    
    # method I: plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    
