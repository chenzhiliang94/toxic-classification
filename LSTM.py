# LSTM with dropout for sequence classification 
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence,text
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


# fix random seed for reproducibility
numpy.random.seed(7)

#fetching data
all_Comments_Data = pd.read_csv('train.csv').fillna(' ') #
all_Comments = all_Comments_Data['comment_text']
class_names = ['severe_toxic','obscene', 'threat', 'insult', 'identity_hate']
    
for class_name in class_names:
    classification = all_Comments_Data[class_name]
    X_train, X_test, y_train, y_test = train_test_split(all_Comments, classification, test_size = 0.1, random_state = 100)
     
    
    
    ###################################
    tk = text.Tokenizer(num_words=200, lower=True)
    tk.fit_on_texts(X_train)
    
    X_train = tk.texts_to_sequences(X_train)
    X_test = tk.texts_to_sequences(X_test)
    print (len(tk.word_counts))

    ###################################
    max_len = 80
    print ("max_len ", max_len)
    print('Pad sequences (samples x time)')
    
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    
    
    max_features = 200
    model = Sequential()
    print('Build model...')
    
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    model.fit(X_train, y=y_train, batch_size=500, epochs=1, verbose=1, validation_split=0.2,  shuffle=True)
    
    y_pred = model.predict(X_test)
        
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
    
    del X_train
    del model
    del y_train
    del X_test