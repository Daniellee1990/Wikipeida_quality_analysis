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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')
colnames = list(wikidata)
#print(colnames)

labels = wikidata["page_class"]
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC' or labels[i] == 'GA':
        labels.loc[i] = '0'
    elif labels[i] == 'BC' or labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '1'
    """
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = '0'
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = '1'
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '2'
    """

labels = labels.convert_objects(convert_numeric=True)
onehotlabels = np_utils.to_categorical(labels)

### preprocess features
features = wikidata.iloc[:, 1:]
min_max_scaler = preprocessing.MinMaxScaler()
features_minmax = min_max_scaler.fit_transform(features)

### split data into training set and label set
X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.4, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(features_minmax, labels, test_size=0.4, random_state=42)

### adjust the dataset dimension
# reshape X to be [samples, time steps, features]
X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

### input for CNN
X_train_CNN = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_CNN = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

### create the deep learning models
epochs = 15
batch_size = 195 #195 best now #190 # 100 # 250
dropoutRate = 0.2
"""
epochs = 200
batch_size = 20
dropoutRate = 0.3
"""
y_test_re = np.argmax(y_test, axis=1)

model1, hist1 = deep_learning_models.basic_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction1 = model1.predict(X_test_LSTM)
prediction1_re = np.argmax(prediction1, axis=1)
basic_LSTM_accuracy = deep_learning_models.getAccuracy(prediction1, y_test)
keys = hist1.history.keys()
print(keys)
print("Precision for basic LSTM")
print(basic_LSTM_accuracy)
#deep_learning_models.plotRoc(prediction1_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist1)
#deep_learning_models.plotTrainingLoss(hist1)

## stacked LSTM with dropout
model2, hist2 = deep_learning_models.LSTM_with_dropout(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs, dropoutRate)
prediction2 = model2.predict(X_test_LSTM)
prediction2_re = np.argmax(prediction2, axis=1)
LSTM_with_dropout_accuracy = deep_learning_models.getAccuracy(prediction2, y_test)
print("Precision for LSTM with dropout")
print(LSTM_with_dropout_accuracy)
#deep_learning_models.plotRoc(prediction2_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist2)
#deep_learning_models.plotTrainingLoss(hist2)

## CNN LSTM
model3, hist3 = deep_learning_models.CNN_LSTM(X_train_CNN, y_train, y_test, batch_size, epochs, dropoutRate)
prediction3 = model3.predict(X_test_CNN)
prediction3_re = np.argmax(prediction3, axis=1)
CNN_LSTM_precision = deep_learning_models.getAccuracy(prediction3, y_test)
print("Precision for CNN LSTM")
print(CNN_LSTM_precision)
#deep_learning_models.plotRoc(prediction3_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist3)
#deep_learning_models.plotTrainingLoss(hist3)

## CNN
model4, hist4 = deep_learning_models.CNN(X_train_CNN, y_train, y_test, batch_size, epochs)
prediction4 = model4.predict(X_test_CNN)
prediction4_re = np.argmax(prediction4, axis=1)
CNN_precision = deep_learning_models.getAccuracy(prediction4, y_test)
print("Precision for CNN")
print(CNN_precision)
#deep_learning_models.plotRoc(prediction4_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist4)
#deep_learning_models.plotTrainingLoss(hist4)

## DNN
model5, hist5 = deep_learning_models.DNN(X_train, y_train, batch_size, epochs, dropoutRate)
prediction5 = model5.predict(X_test)
prediction5_re = np.argmax(prediction5, axis=1)
DNN_precision = deep_learning_models.getAccuracy(prediction5, y_test)
print("precision for DNN")
print(DNN_precision)
#deep_learning_models.plotRoc(prediction5_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist5)
#deep_learning_models.plotTrainingLoss(hist5)

## stacked LSTM
model6, hist6 = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction6 = model6.predict(X_test_LSTM)
prediction6_re = np.argmax(prediction6, axis=1)
stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction6, y_test)
print("Precision for stacked LSTM")
print(stacked_LSTM_precision)
#deep_learning_models.plotRoc(prediction6_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist6)
#deep_learning_models.plotTrainingLoss(hist6)

print("Model accuracy")
plt.plot(hist1.history['acc'], marker = 'o', label='Basic LSTM', markersize=10)
plt.plot(hist2.history['acc'], marker = ',', label = 'LSTM with dropout', markersize=10)
plt.plot(hist3.history['acc'], marker = 's', label = 'CNN_LSTM', markersize=10)
plt.plot(hist4.history['acc'], marker = '*', label = 'CNN', markersize=10)
plt.plot(hist5.history['acc'], marker = '<', label = 'DNN', markersize=10)
plt.plot(hist6.history['acc'], marker = '.', label = 'Stacked LSTM', markersize=10)
plt.title('Model training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(shadow=False, fontsize='small', loc='lower right')
plt.savefig("Model_accuracy.png")
plt.show()

print("Model loss")
plt.plot(hist1.history['loss'], marker = 'o', label='Basic LSTM', markersize=10)
plt.plot(hist2.history['loss'], marker = ',', label = 'LSTM with dropout', markersize=10)
plt.plot(hist3.history['loss'], marker = 's', label = 'CNN_LSTM', markersize=10)
plt.plot(hist4.history['loss'], marker = '*', label = 'CNN', markersize=10)
plt.plot(hist5.history['loss'], marker = '<', label = 'DNN', markersize=10)
plt.plot(hist6.history['loss'], marker = '.', label = 'Stacked LSTM', markersize=10)
plt.title('Model training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(shadow=False, fontsize='small', loc='upper right')
plt.savefig("Model_loss.png")
plt.show()

print("ROC")
fprUni1, tprUni1, _ = roc_curve(prediction1_re, y_test_re)
roc_aucUni1 = auc(fprUni1, tprUni1)

fprUni2, tprUni2, _ = roc_curve(prediction2_re, y_test_re)
roc_aucUni2 = auc(fprUni2, tprUni2)

fprUni3, tprUni3, _ = roc_curve(prediction3_re, y_test_re)
roc_aucUni3 = auc(fprUni3, tprUni3)

fprUni4, tprUni4, _ = roc_curve(prediction4_re, y_test_re)
roc_aucUni4 = auc(fprUni4, tprUni4)

fprUni5, tprUni5, _ = roc_curve(prediction5_re, y_test_re)
roc_aucUni5 = auc(fprUni5, tprUni5)

fprUni6, tprUni6, _ = roc_curve(prediction6_re, y_test_re)
roc_aucUni6 = auc(fprUni6, tprUni6)

plt.figure()
lw = 2
plt.plot(fprUni1, tprUni1, marker = 'o', markersize=5, linestyle='--',
         lw=lw, label='Basic LSTM (AUC = %0.2f)' % roc_aucUni1)
plt.plot(fprUni2, tprUni2, marker = ',', markersize=5, linestyle='-.',
         lw=lw, label='LSTM with dropout (AUC = %0.2f)' % roc_aucUni2)
plt.plot(fprUni3, tprUni3, marker = 's', markersize=5, linestyle=':',
         lw=lw, label='CNN_LSTM (AUC = %0.2f)' % roc_aucUni3)
plt.plot(fprUni4, tprUni4, marker = '*', markersize=5, linestyle='-',
         lw=lw, label='CNN (AUC = %0.2f)' % roc_aucUni4)
plt.plot(fprUni5, tprUni5, marker = '<', markersize=5, linestyle='-.',
         lw=lw, label='DNN (AUC = %0.2f)' % roc_aucUni5)
plt.plot(fprUni6, tprUni6, marker = 's', markersize=5, linestyle=':',
         lw=lw, label='Stacked LSTM (AUC = %0.2f)' % roc_aucUni6)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()