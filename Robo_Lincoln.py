
#Goal: Build an algorithm that writes speaches in the tone of Abraham Lincoln,
#secondary goals: Build an algorithm that mixes the speach of Abraham lincoln and Donald Trump
#Make an algorithm that translates Lincoln's speaches into trumps languge

### concept
#using recurrant nerual networks and NLTK

### input



## concept for task 2
## build a  -translating- algorithm by taking the input and translating it into
## its simplest form using thesorus lookup. Then back converting it into the languge
## that Donald Trump would use


from __future__ import division
import nltk
import pandas as pd
import numpy as np
import theano
import keras
import os
import pickle
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder

import random


def context():
      return
                ### ----------------- starting parameters ------------------------- ##

##################### ------------------------------------------------------------------------ #####################
filehandle = 'C:\Users\John\Desktop\Investing\RNN\Lincoln_1.txt'
data =  [word for line in open(filehandle, 'r') for word in line.split()]
data = np.asarray(nltk.pos_tag(data))

syntax_unique = np.unique(data[:,1], axis=0)
syntax_unique_n = len(syntax_unique)

mydict_in = {}
mydict_wd = {}

for i in range(0,len(syntax_unique)):
      in_syntax = data[np.where(data[:,1] == syntax_unique[i])[0],0]
      word_unique = np.unique(in_syntax, axis=0)
      word_dict ={}
      word_dict_in = {}
      for j in range(0,len(word_unique)):
            k = len(word_unique)
            word_dict_in[j] = word_unique[j]
            word_dict[ word_unique[j]] = j
      mydict_in[i ] = syntax_unique[i] , word_dict_in , k
      mydict_wd[syntax_unique[i] ] = i , word_dict , k

data_indexed =[]
for word , syntax in data:
      data_indexed.append([mydict_wd[syntax][0] , mydict_wd[syntax][1][word] , mydict_wd[syntax][2]] )
      
def GetXY(data , sequence):
      X=[]
      Y=[]
      X2=[]
      for i in range(1,len(data)):
            seq = []
            seq_n =[]
            for n in range(1,sequence):
                  seq.append(data[i-n])
                  A = data[i-n][0]/syntax_unique_n
                  C = data[i-n][2] / data[i-n][2]
                  B = data[i-n][1] / data[i-n][2]
                  seq_n.append([A,B,C])
            X.append(seq)
            X2.append(seq_n)
            Y.append(data[i])
      return np.asarray(X) , np.asarray(Y) , np.asarray(X2)


batch_size = 1
variables = 1
sequence_l = 10
X,Y,X2 = GetXY(data_indexed, sequence_l)

enc = OneHotEncoder()
enc.fit(Y)
Y2 = enc.transform(Y).toarray()

#X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))


###---------------------------

model = Sequential()
model.add(LSTM(250, input_shape=(X2.shape[1], X2.shape[2]) , return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y2.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X2, Y2, epochs=2, batch_size=100)# , callbacks =callbacks_list)

###---------------------------
syntax_unique_n = len(syntax_unique)

length = 200
seed = list(X2[9])
indexed = seed
out = []
n = sequence_l-1
while n < length :
      input = np.asarray(indexed[n-(sequence_l-1):n])
      input = np.reshape(input,(1,input.shape[0],input.shape[1]))
      predict = model.predict(np.asarray(input))
      syntax = np.argmax(predict[0][:syntax_unique_n])
      word_k = mydict_in[syntax][2]
      word_int = np.argmax(predict[0][syntax_unique_n:syntax_unique_n+word_k])
      word = mydict_in[syntax][1][word_int]
      syntax_word = [syntax , word_int , word_k]
      syntax_word_n = [syntax / syntax_unique_n , word_int / word_k  , word_k / word_k]
      indexed.append(np.asarray(syntax_word_n))
      out.append(syntax_word)
      n = n+1

L_speech =[]
for i , j , k  in out:
      L_speech.append(mydict_in[i][1][j])

print L_speech

###_________________________

### Simplification algorithm

#wn.synsets('small'):


