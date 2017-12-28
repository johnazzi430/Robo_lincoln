from __future__ import division
import nltk
import pandas as pd
import numpy as np
import theano
import keras
import os
import pickle
import pandas as pd
import scipy

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D , MaxPool1D
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import class_weight
from keras.optimizers import SGD
from keras.losses import sparse_categorical_crossentropy
from sklearn.preprocessing import OneHotEncoder

import random


def context():
    return
    ### ----------------- starting parameters ------------------------- ##


##################### ------------------------------------------------------------------------ #####################
filehandle = 'Lincoln_1.txt'
data = [word.lower() for line in open(filehandle, 'r') for word in line.split()]
data = np.asarray(nltk.pos_tag(data))[:5000]

syntax_unique = np.unique(data[:, 1], axis=0)
syntax_unique_n = len(syntax_unique)


def create_index (data ):
    word_index = []
    word_num =[]
    for word in data:
        if word not in word_index:
            word_index.append(word)
        word_num.append(np.where(np.asarray(word_index) == word)[0][0])
    return np.asarray(word_index) ,  np.asarray(word_num)

word_index , word_num = create_index(data[:,0])
syn_index , syn_num = create_index(data[:,1])

enc = OneHotEncoder()
enc.fit(word_num.reshape(-1,1))
wrd_enc = enc.transform(word_num.reshape(-1,1)).toarray()
enc.fit(syn_num.reshape(-1,1))
syn_enc = enc.transform(syn_num.reshape(-1,1)).toarray()

batch_size = 1
variables = 1
sequence_l = 20



def prepareforConv(datain , lookback):
    data_out =[]
    for rw in range(0+lookback , datain.shape[0]):
        data_row = []
        for it in range(1,lookback):
            data_row.append(datain[rw-it])
        data_out.append(data_row)
    return np.asarray(data_out)

lookback = 20
X_in = np.asarray([word_num , ])
X_in_enc = np.append(wrd_enc , syn_enc , axis=1)
X = prepareforConv( wrd_enc , lookback)

Y = word_num[lookback:]
X_ = X
#X_ = X / np.max(Y)
#Y_ = Y / np.max(Y)

#X_ = X_.reshape(X.shape[0] ,1 ,  X.shape[1])
#Y_ = Y_.reshape(Y.shape[0],1)
Y_ = wrd_enc[lookback:]

Y_size = wrd_enc[lookback:].shape
class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

batch_size = 1000
model = Sequential()
model.add(Conv1D(64, 3, padding='same', input_shape=(X.shape[1], X_.shape[2])))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(Y_size[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy' ,  optimizer='adam' ,  metrics=['accuracy'])
model.fit(X_, Y_, epochs=2, batch_size=batch_size, verbose=True, shuffle = True, class_weight = class_weight )

for x in model.predict(X_[10:30]):
    print(np.argsort(x)[::-1][:5])

#filepath="Robo-Lincoln//Data Dump//weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
#pickle(filepath)
###---------------------------


seed = 595

##'like x'
out = X[0]
out = X[seed]
n=0
length = 50
wrds=[]
while n < length:
    input = out[-lookback+1:].reshape(1 , X.shape[1] , X.shape[2])
    predict = model.predict(input)
    out_enc = np.zeros(X.shape[2])
    wrds.append(np.argmax(predict))
    out_enc[np.argmax(predict)] = 1
    out = np.append(out,out_enc.reshape(1,X.shape[2]),axis=0)
    n = n+1


L_speech = ' '.join(word_index[wrds])

print(L_speech)

context.save = False