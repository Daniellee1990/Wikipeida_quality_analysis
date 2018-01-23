#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:23:54 2018

@author: lixiaodan
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten

def stacked_LSTM(X_train, y_train, X_test, y_test, batch_size, epochs):
    ## CNN LSTM
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(1, X_test.shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(y_test.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model
    
def CNN(X_train, y_train, y_test, batch_size, epochs):
    ## CNN
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=5, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model

def CNN_LSTM(X_train, y_train, y_test, batch_size, epochs, dropoutRate):
    ## CNN LSTM
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(256))
    model.add(Dropout(dropoutRate))
    model.add(Dense(y_test.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model

def stacked_LSTMs_with_dropout(X_train, y_train, y_test, batch_size, epochs, dropout_rate):
    ## stacked LSTM
    model = Sequential()
    model.add(LSTM(33, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_test.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #print(model.summary())
    model.fit(X_train, y_train, batch_size, epochs)
    return model

# define baseline model
def DNN(X_train, Y_train, batch_size, epochs, dropout):
	# create model
    model = Sequential() 
    model.add(Dense(20, input_dim=X_train.shape[1], init='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(dropout))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(Y_train.shape[1], init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    model.fit(X_train, Y_train, batch_size, epochs)
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