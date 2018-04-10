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
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import log_loss

def stacked_LSTM(X_train, y_train, X_test, y_test, batch_size, epochs):
    ## CNN LSTM
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(1, X_test.shape[2]))) # 32 # 50
    model.add(LSTM(200, return_sequences=True)) #32
    model.add(LSTM(200)) # 32 # 200
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history
    
def CNN(X_train, y_train, y_test, batch_size, epochs):
    ## CNN
    model = Sequential()
    #model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=5, kernel_size=1, activation='relu'))
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=8, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history

def CNN_LSTM(X_train, y_train, y_test, batch_size, epochs, dropoutRate):
    ## CNN LSTM
    model = Sequential()
    model.add(Conv1D(input_shape=(X_train.shape[1], 1), filters=8, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(200)) # 256
    model.add(Dropout(dropoutRate))
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history

def LSTM_with_dropout(X_train, y_train, x_test, y_test, batch_size, epochs, dropout_rate):
    ## stacked LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) #33
    #model.add(LSTM(32, return_sequences=True, input_shape=(1, x_test.shape[2])))
    model.add(Dropout(dropout_rate))
    #model.add(LSTM(100))
    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history

def basic_LSTM(X_train, y_train, x_test, y_test, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True)) # 33
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history

def Bidirectional_LSTM(X_train, y_train, x_test, y_test, batch_size, epochs):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(y_test.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, y_train, batch_size, epochs)
    return model, history

# define baseline model
def DNN(X_train, Y_train, batch_size, epochs, dropout):
	# create model
    model = Sequential() 
    model.add(Dense(20, input_dim=X_train.shape[1], init='normal', activation='relu')) # 20
    model.add(Dropout(dropout))
    model.add(Dense(100, init='normal', activation='relu')) # 100
    model.add(Dropout(dropout))
    model.add(Dense(100, init='normal', activation='relu')) # 100
    model.add(Dropout(dropout))
    #model.add(Dense(200, init='normal', activation='relu'))
    #model.add(Dropout(dropout))
    #model.add(Dense(200, init='normal', activation='relu'))
    #model.add(Dropout(dropout))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.add(Dense(Y_train.shape[1], init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())
    history = model.fit(X_train, Y_train, batch_size, epochs)
    return model, history

def getAccuracy(prediction, y_test): ### prediction and y_test are both encoded.
    sample_size = prediction.shape[0]
    col_num = prediction.shape[1]
    correct_num = 0
    wrong_num = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
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
            #print("result id")
            #print(res_id)
            correct_num = correct_num + 1
            if res_id == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            wrong_num = wrong_num + 1
            if res_id == 0:
                fn = fn + 1
            else:
                fp = fp + 1
    accuracy = float(correct_num) / sample_size
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    TNR = float(tn) / (tn + fp)
    F = 2 * float(precision * recall) / (precision + recall) 
    return accuracy, precision, recall, TNR, F

def getAccuracyMulti(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    F1 = metrics.f1_score(y_true, y_pred, average='weighted') 
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    fbeta = metrics.fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    return accuracy, precision, recall, F1, fbeta

def transformResult(result):
    sample_size = result.shape[0]
    col_num = result.shape[1]
    result_list = list()
    for i in range(sample_size):
        cur_row = result[i,:]
        max = 0
        max_id = 0
        for j in range(col_num):
            if cur_row[j] > max:
                max = cur_row[j]
                max_id = j
        result_list.append(max_id)
    return result_list

def plotTrainingAccuracy(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def plotTrainingLoss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def plotRoc(y_predict, y_label):
    # roc for unigram
    fprUni, tprUni, _ = roc_curve(y_predict, y_label)
    roc_aucUni = auc(fprUni, tprUni)
    
    plt.figure()
    lw = 2
    plt.plot(fprUni, tprUni, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_aucUni)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')