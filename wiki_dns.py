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
import time

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/research/result_3rd_paper/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')
colnames = list(wikidata)
#print(colnames)

labels = wikidata["page_class"]
### good is possitive while bad is negative
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC' or labels[i] == 'GA':
        labels.loc[i] = '1'
    elif labels[i] == 'BC' or labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = '0'
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
accuracies = list()
precisions = list()
Fs = list()
TNRs = list()
recalls = list()

### Bidirectional LSTM 
#start_time0 = time.clock()
model0, hist0 = deep_learning_models.Bidirectional_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction0 = model0.predict(X_test_LSTM)
#end_time0 = time.clock()
#Bi_LSTM_performance = end_time0 - start_time0
prediction0_re = np.argmax(prediction0, axis=1)
Bidirectional_LSTM_accuracy, bi_precision, bi_recall, bi_TNR, bi_F = deep_learning_models.getAccuracy(prediction0, y_test)
keys = hist0.history.keys()
print(keys)
print("Precision for bidirectional LSTM")
print(Bidirectional_LSTM_accuracy)
print(bi_precision)
print(bi_recall)
print(bi_TNR)
print(bi_F)
accuracy_pair = list()
accuracy_pair.append("bidirectional LSTM")
accuracy_pair.append(Bidirectional_LSTM_accuracy)
accuracies.append(accuracy_pair)

#start_time1 = time.clock()
model1, hist1 = deep_learning_models.basic_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction1 = model1.predict(X_test_LSTM)
#end_time1 = time.clock()
#basic_LSTM_performance = end_time1 - start_time1
prediction1_re = np.argmax(prediction1, axis=1)
basic_LSTM_accuracy, LSTM_precision, LSTM_recall, LSTM_TNR, LSTM_F= deep_learning_models.getAccuracy(prediction1, y_test)
keys = hist1.history.keys()
print(keys)
print("Precision for basic LSTM")
print(basic_LSTM_accuracy)
print(LSTM_precision)
print(LSTM_recall)
print(LSTM_TNR)
print(LSTM_F)
accuracy_pair = list()
accuracy_pair.append("basic LSTM")
accuracy_pair.append(basic_LSTM_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction1_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist1)
#deep_learning_models.plotTrainingLoss(hist1)

## stacked LSTM with dropout
#start_time2 = time.clock()
model2, hist2 = deep_learning_models.LSTM_with_dropout(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs, dropoutRate)
prediction2 = model2.predict(X_test_LSTM)
#end_time2 = time.clock()
#LSTM_with_dropout_performance = end_time2 - start_time2
prediction2_re = np.argmax(prediction2, axis=1)
LSTM_with_dropout_accuracy, dropout_precision, dropout_recall, dropout_TNR, dropout_F = deep_learning_models.getAccuracy(prediction2, y_test)
print("Precision for LSTM with dropout")
print(LSTM_with_dropout_accuracy)
print(dropout_precision)
print(dropout_recall)
print(dropout_TNR)
print(dropout_F)
accuracy_pair = list()
accuracy_pair.append("LSTM with dropout")
accuracy_pair.append(LSTM_with_dropout_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction2_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist2)
#deep_learning_models.plotTrainingLoss(hist2)

## CNN LSTM
#start_time3 = time.clock()
model3, hist3 = deep_learning_models.CNN_LSTM(X_train_CNN, y_train, y_test, batch_size, epochs, dropoutRate)
prediction3 = model3.predict(X_test_CNN)
#end_time3 = time.clock()
#CNN_LSTM_performance = end_time3 - start_time3
prediction3_re = np.argmax(prediction3, axis=1)
CNN_LSTM_accuracy, CNN_LSTM_precision, CNN_LSTM_recall, CNN_LSTM_TNR, CNN_LSTM_F= deep_learning_models.getAccuracy(prediction3, y_test)
print("Precision for CNN LSTM")
print(CNN_LSTM_accuracy)
print(CNN_LSTM_precision)
print(CNN_LSTM_recall)
print(CNN_LSTM_TNR)
print(CNN_LSTM_F)
accuracy_pair = list()
accuracy_pair.append("CNN LSTM")
accuracy_pair.append(CNN_LSTM_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction3_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist3)
#deep_learning_models.plotTrainingLoss(hist3)

## CNN
#start_time4 = time.clock()
model4, hist4 = deep_learning_models.CNN(X_train_CNN, y_train, y_test, batch_size, epochs)
prediction4 = model4.predict(X_test_CNN)
#end_time4 = time.clock()
#CNN_performance = end_time4 - start_time4
prediction4_re = np.argmax(prediction4, axis=1)
CNN_accuracy, CNN_precision, CNN_recall, CNN_TNR, CNN_F = deep_learning_models.getAccuracy(prediction4, y_test)
print("Precision for CNN")
print(CNN_accuracy)
print(CNN_precision)
print(CNN_recall)
print(CNN_TNR)
print(CNN_F)
accuracy_pair = list()
accuracy_pair.append("CNN")
accuracy_pair.append(CNN_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction4_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist4)
#deep_learning_models.plotTrainingLoss(hist4)

## DNN
#start_time5 = time.clock()
model5, hist5 = deep_learning_models.DNN(X_train, y_train, batch_size, epochs, dropoutRate)
prediction5 = model5.predict(X_test)
#end_time5 = time.clock()
#DNN_performance = end_time5 - start_time5
prediction5_re = np.argmax(prediction5, axis=1)
DNN_accuracy, DNN_precision, DNN_recall, DNN_TNR, DNN_F = deep_learning_models.getAccuracy(prediction5, y_test)
print("precision for DNN")
print(DNN_accuracy)
print(DNN_precision)
print(DNN_recall)
print(DNN_TNR)
print(DNN_F)
accuracy_pair = list()
accuracy_pair.append("DNN")
accuracy_pair.append(DNN_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction5_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist5)
#deep_learning_models.plotTrainingLoss(hist5)

## stacked LSTM
#start_time6 = time.clock()
model6, hist6 = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction6 = model6.predict(X_test_LSTM)
#end_time6 = time.clock()
#stacked_LSTM_performance = end_time6 - start_time6 
prediction6_re = np.argmax(prediction6, axis=1)
stacked_LSTM_accuracy, stacked_LSTM_precision, stacked_LSTM_recall, stacked_LSTM_TNR, stacked_LSTM_F = deep_learning_models.getAccuracy(prediction6, y_test)
print("Precision for stacked LSTM")
print(stacked_LSTM_accuracy)
print(stacked_LSTM_precision)
print(stacked_LSTM_recall)
print(stacked_LSTM_TNR)
print(stacked_LSTM_F)
accuracy_pair = list()
accuracy_pair.append("stacked LSTM")
accuracy_pair.append(stacked_LSTM_accuracy)
accuracies.append(accuracy_pair)

#deep_learning_models.plotRoc(prediction6_re, y_test_re)
#deep_learning_models.plotTrainingAccuracy(hist6)
#deep_learning_models.plotTrainingLoss(hist6)
"""
print("Model accuracy")
plt.plot(hist0.history['acc'], marker = 'v', label = 'Bidirectional LSTM', markersize=10)
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
plt.savefig("Model_accuracy.png", dpi=600)
plt.show()

print("Model loss")
plt.plot(hist0.history['loss'], marker = 'v', label='Bidirectional LSTM', markersize=10)
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
plt.savefig("Model_loss.png", dpi=600)
plt.show()
"""

print("ROC")
fprUni0, tprUni0, _ = roc_curve(prediction0_re, y_test_re)
roc_aucUni0 = auc(fprUni0, tprUni0)

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
plt.plot(fprUni0, tprUni0, marker = 'v', markersize=5, linestyle=':',
         lw=lw, label='Bidirectional LSTM (AUC = %0.2f)' % roc_aucUni1)
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
plt.savefig('Mean_roc.png', dpi=600)
plt.show()
print("Accuracies")
print(accuracies)