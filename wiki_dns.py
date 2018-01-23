#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 22:10:37 2018

@author: lixiaodan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
import deep_learning_models

#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')


labels = wikidata["page_class"]
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = '0'
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = '1'
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '2'

labels = labels.convert_objects(convert_numeric=True)
onehotlabels = np_utils.to_categorical(labels)

### preprocess features
features = wikidata.iloc[:, 0:-1]
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.4, random_state=42)

### adjust the dataset dimension
# reshape X to be [samples, time steps, features]
X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

### input for CNN
X_train_CNN = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_CNN = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

### create the deep learning models
"""
epochs = 30
batch_size = 100
dropoutRate = 0.2
"""
epochs = 50
batch_size = 50
dropoutRate = 0.2

## stacked LSTM with dropout
model = deep_learning_models.stacked_LSTMs_with_dropout(X_train_LSTM, y_train, y_test, batch_size, epochs, dropoutRate)
prediction = model.predict(X_test_LSTM)
LSTM_accuracy = deep_learning_models.getAccuracy(prediction, y_test)
print("Precision for stacked LSTM with dropout")
print(LSTM_accuracy)

## CNN LSTM
model = deep_learning_models.CNN_LSTM(X_train_CNN, y_train, y_test, batch_size, epochs, dropoutRate)
prediction = model.predict(X_test_CNN)
CNN_LSTM_precision = deep_learning_models.getAccuracy(prediction, y_test)
print("Precision for CNN LSTM")
print(CNN_LSTM_precision)

## CNN
model = deep_learning_models.CNN(X_train_CNN, y_train, y_test, batch_size, epochs)
prediction = model.predict(X_test_CNN)
CNN_precision = deep_learning_models.getAccuracy(prediction, y_test)
print("Precision for CNN")
print(CNN_precision)

## DNN
model = deep_learning_models.DNN(X_train, y_train, batch_size, epochs, dropoutRate)
predictions = model.predict(X_test)
DNN_precision = deep_learning_models.getAccuracy(predictions, y_test)
print("precision for DNN")
print(DNN_precision)

## stacked LSTM
model = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction = model.predict(X_test_LSTM)
stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction, y_test)
print("Precision for stacked LSTM")
print(stacked_LSTM_precision)