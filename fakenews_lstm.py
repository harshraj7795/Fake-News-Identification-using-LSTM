# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:14:34 2021

@author: HSingh
"""

import pandas as pd

#loading the dataframe
df = pd.read_csv('train.csv')

#extracting the dependent and independent features
df = df.dropna()
x = df.drop('label',axis = 1)
y = df['label']

#importing libraries for LSTM modelling
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

#initializing vocabulary size
vocab = 5000

#importing libraries for text processing
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

msg = x.copy()
msg.reset_index(inplace=True)

#text preprocessing
ps = PorterStemmer()
corpus = []

for i in range(len(msg)):
    newmsg = re.sub('[^a-zA-Z]',' ',msg['title'][i])
    newmsg = newmsg.lower()
    newmsg = newmsg.split()
    newmsg = [ps.stem(word) for word in newmsg if word not in stopwords.words('english')]
    newmsg = ' '.join(newmsg)
    corpus.append(newmsg)

#onehot representation
onehot = [one_hot(word,vocab) for word in corpus]

#embedded representations
sent_len = 20
emb_doc = pad_sequences(onehot,padding = 'pre', maxlen = sent_len)

#creating the LSTM model
emb_feat_vectors = 40
model = Sequential()
model.add(Embedding(vocab,emb_feat_vectors,input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print(model.summary())

#preparing the model for training the model
import numpy as np
x_new = np.array(emb_doc)
y_new = np.array(y)

from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts = train_test_split(x_new,y_new,test_size = 0.3, random_state = 42)

#training the model
model.fit(x_tr,y_tr, validation_data=(x_ts,y_ts),epochs=10,batch_size=64)

#evaluating the model
from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(x_ts)
print(accuracy_score(y_ts,y_pred))







