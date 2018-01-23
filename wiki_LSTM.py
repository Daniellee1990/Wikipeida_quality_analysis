#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:48:11 2018

@author: lixiaodan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.utils import np_utils

def stacked_LSTMs_with_dropout(X_train, y_train, y_test, batch_size, epochs, dropout_rate):
    ## stacked LSTM
    model = Sequential()
    model.add(LSTM(33, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_test.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model

def getAccuracy(prediction, y_test): ### prediction and y_test are both encoded.
    sample_size = prediction.shape[0]
    col_num = prediction.shape[1]
    correct_num = 0
    wrong_num = 0
    for i in range(sample_size):
        cur_row = prediction[i,:]
        max = 0
        max_id = 0
        res_id = 0 
        for j in range(col_num):
            if cur_row[j] > max:
                max = cur_row[j]
                max_id = j
        for k in range(col_num):
            if y_test[i, k] == 1:
                res_id = k
                break
        if res_id == max_id:
            correct_num = correct_num + 1
        else:
            wrong_num = wrong_num + 1
    accuracy = float(correct_num) / sample_size
    return accuracy

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')

labels = wikidata["page_class"]
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = '0'
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = '1'
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '2'

labels = labels.convert_objects(convert_numeric=True)
print(labels)
onehotlabels = np_utils.to_categorical(labels)

### preprocess features
features = wikidata.iloc[:, 0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.3, random_state=42)

### create the deep learning models
epochs = 50
batch_size = 100
dropoutRate = 0.2

# reshape X to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

## stacked LSTM
model = stacked_LSTMs_with_dropout(X_train, y_train, y_test, batch_size, epochs, 0.2)
prediction = model.predict(X_test)
accuracy = getAccuracy(prediction, y_test)
print(accuracy)