'''
Created on 27 Mar 2018

@author: zhi liang
'''

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.classification import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

all_Comments_Data = pd.read_csv('train.csv').fillna(' ') #

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
    max_features=1000) #Only considers top 10000 features in the final step, can be changed

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=5000)

class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        #y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C,class_weight='balanced', dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


ros = RandomOverSampler(random_state=0)
for class_name in class_names:
    print(class_name)
    classification = all_Comments_Data[class_name]
    
    #we use 1% as testing comments, and 99% as training comments
    X_train, X_test, y_train, y_test = train_test_split(all_Comments, classification, test_size = 0.1, random_state = 100)

   
    #transform training set using fit_transform
    X_train = word_vectorizer.fit_transform(X_train)
    #transform the testing comments USING only the training comments' fittings
    X_test  = word_vectorizer.transform(X_test)
    X_train, y_train = ros.fit_sample(X_train, y_train)


    model = NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #confusion matrix outcome
    #can someone check what does the matrix mean? I forgot which entry corresponds to TP, TN, FN, FP
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
    

    
