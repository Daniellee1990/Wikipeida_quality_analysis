#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 09:28:30 2018

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
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import plot_precision_recall

wikidata = pd.read_csv('/Users/lixiaodan/Desktop/research/result_3rd_paper/wikipedia_project/dataset/wikipedia_with_all_features.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')
#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_hist_net.csv')
colnames = list(wikidata)
class_names = np.array([['Good'], ['Medium'], ['Low']])
labels = wikidata["page_class"]

### good is possitive while bad is negative
for i in range(labels.shape[0]):
    if labels[i] == 'FA' or labels[i] == 'AC':
        labels.loc[i] = 0
    elif labels[i] == 'GA' or labels[i] == 'BC':
        labels.loc[i] = 1
    elif labels[i] == 'ST' or labels[i] == 'SB':
        labels.loc[i] = 2

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
transformed_y = deep_learning_models.transformResult(y_test)
transformed_y_train = deep_learning_models.transformResult(y_train) ### one dimention list

### Bidirectional LSTM 
#start_time0 = time.clock()
model0, hist0 = deep_learning_models.Bidirectional_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction0 = model0.predict(X_test_LSTM)
transformed_pre0 = deep_learning_models.transformResult(prediction0)
#end_time0 = time.clock()
#Bi_LSTM_performance = end_time0 - start_time0
prediction0_re = np.argmax(prediction0, axis=1)
Bidirectional_LSTM_accuracy, bi_precision, bi_recall, bi_F1, bi_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre0, transformed_y)
keys = hist0.history.keys()
#print(keys)
print("Precision for bidirectional LSTM")
print(Bidirectional_LSTM_accuracy)
print(bi_precision)
print(bi_recall)
print(bi_fbeta)
print(bi_F1)
accuracy_pair = list()
accuracy_pair.append("bidirectional LSTM")
accuracy_pair.append(Bidirectional_LSTM_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix"
cnf_matrix_0 = confusion_matrix(transformed_y, transformed_pre0)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_0, classes=class_names, normalize=True,
                      title='Confusion matrix for bidirectional LSTM accuracy')
plt.savefig('Confusion matrix for bidirectional LSTM accuracy.png', dpi=300)
plt.show()

plot_precision_recall.plot_average_precision(y_test, prediction0, 'bi-LSTM')

#start_time1 = time.clock()
model1, hist1 = deep_learning_models.basic_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction1 = model1.predict(X_test_LSTM)
transformed_pre1 = deep_learning_models.transformResult(prediction1)
#end_time1 = time.clock()
#basic_LSTM_performance = end_time1 - start_time1
prediction1_re = np.argmax(prediction1, axis=1)
basic_LSTM_accuracy, LSTM_precision, LSTM_recall, LSTM_F1, LSTM_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre1, transformed_y)
keys = hist1.history.keys()
#print(keys)
print("Precision for basic LSTM")
print(basic_LSTM_accuracy)
print(LSTM_precision)
print(LSTM_recall)
print(LSTM_fbeta)
print(LSTM_F1)
accuracy_pair = list()
accuracy_pair.append("basic LSTM")
accuracy_pair.append(basic_LSTM_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix" 
cnf_matrix_1 = confusion_matrix(transformed_y, transformed_pre1)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_1, classes=class_names, normalize=True,
                      title='Confusion matrix for basic LSTM accuracy')
plt.savefig('Confusion matrix for basic LSTM accuracy.png', dpi=300)
plt.show()

plot_precision_recall.plot_average_precision(y_test, prediction1, 'basic LSTM')

## stacked LSTM with dropout
#start_time2 = time.clock()
model2, hist2 = deep_learning_models.LSTM_with_dropout(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs, dropoutRate)
prediction2 = model2.predict(X_test_LSTM)
transformed_pre2 = deep_learning_models.transformResult(prediction2)
#end_time2 = time.clock()
#LSTM_with_dropout_performance = end_time2 - start_time2
prediction2_re = np.argmax(prediction2, axis=1)
LSTM_with_dropout_accuracy, dropout_precision, dropout_recall, dropout_F1, dropout_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre2, transformed_y)
print("Precision for LSTM with dropout")
print(LSTM_with_dropout_accuracy)
print(dropout_precision)
print(dropout_recall)
print(dropout_fbeta)
print(dropout_F1)

accuracy_pair = list()
accuracy_pair.append("LSTM with dropout")
accuracy_pair.append(LSTM_with_dropout_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix" 
cnf_matrix_2 = confusion_matrix(transformed_y, transformed_pre2)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_2, classes=class_names, normalize=True,
                      title='Confusion matrix for LSTM with dropout accuracy')
plt.savefig('Confusion matrix for LSTM with dropout accuracy.png', dpi=300)
plt.show()

plot_precision_recall.plot_average_precision(y_test, prediction2, 'LSTM with dropout')

## CNN LSTM
#start_time3 = time.clock()
model3, hist3 = deep_learning_models.CNN_LSTM(X_train_CNN, y_train, y_test, batch_size, epochs, dropoutRate)
prediction3 = model3.predict(X_test_CNN)
transformed_pre3 = deep_learning_models.transformResult(prediction3)
#end_time3 = time.clock()
#CNN_LSTM_performance = end_time3 - start_time3
prediction3_re = np.argmax(prediction3, axis=1)
CNN_LSTM_accuracy, CNN_LSTM_precision, CNN_LSTM_recall, CNN_LSTM_F1, CNN_LSTM_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre3, transformed_y)
print("Precision for CNN LSTM")
print(CNN_LSTM_accuracy)
print(CNN_LSTM_precision)
print(CNN_LSTM_recall)
print(CNN_LSTM_fbeta)
print(CNN_LSTM_F1)

accuracy_pair = list()
accuracy_pair.append("CNN LSTM")
accuracy_pair.append(CNN_LSTM_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix" 
cnf_matrix_3 = confusion_matrix(transformed_y, transformed_pre3)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_3, classes=class_names, normalize=True,
                      title='Confusion matrix for CNN LSTM accuracy')
plt.savefig('Confusion matrix for CNN LSTM accuracy.png', dpi=300)
plt.show()
plot_precision_recall.plot_average_precision(y_test, prediction3, 'CNN_LSTM')

## CNN
#start_time4 = time.clock()
model4, hist4 = deep_learning_models.CNN(X_train_CNN, y_train, y_test, batch_size, epochs)
prediction4 = model4.predict(X_test_CNN)
transformed_pre4 = deep_learning_models.transformResult(prediction4)
#end_time4 = time.clock()
#CNN_performance = end_time4 - start_time4
prediction4_re = np.argmax(prediction4, axis=1)
CNN_accuracy, CNN_precision, CNN_recall, CNN_F1, CNN_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre4, transformed_y)
print("Precision for CNN")
print(CNN_accuracy)
print(CNN_precision)
print(CNN_recall)
print(CNN_fbeta)
print(CNN_F1)

accuracy_pair = list()
accuracy_pair.append("CNN")
accuracy_pair.append(CNN_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix"
cnf_matrix_4 = confusion_matrix(transformed_y, transformed_pre4)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_4, classes=class_names, normalize=True,
                      title='Confusion matrix for CNN accuracy')
plt.savefig('Confusion matrix for CNN accuracy.png', dpi=300)
plt.show()
plot_precision_recall.plot_average_precision(y_test, prediction4, 'CNN')

## DNN
model5, hist5 = deep_learning_models.DNN(X_train, y_train, batch_size, epochs, dropoutRate)
prediction5 = model5.predict(X_test)
transformed_pre5 = deep_learning_models.transformResult(prediction5)
prediction5_re = np.argmax(prediction5, axis=1)
DNN_accuracy, DNN_precision, DNN_recall, DNN_F1, DNN_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre5, transformed_y)
print("precision for DNN")
print(DNN_accuracy)
print(DNN_precision)
print(DNN_recall)
print(DNN_fbeta)
print(DNN_F1)

accuracy_pair = list()
accuracy_pair.append("DNN")
accuracy_pair.append(DNN_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix" 
cnf_matrix_5 = confusion_matrix(transformed_y, transformed_pre5)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_5, classes=class_names, normalize=True,
                      title='Confusion matrix for DNN accuracy')
plt.savefig('Confusion matrix for DNN accuracy.png', dpi=300)
plt.show()
plot_precision_recall.plot_average_precision(y_test, prediction5, 'DNN')

## stacked LSTM
#start_time6 = time.clock()
model6, hist6 = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
prediction6 = model6.predict(X_test_LSTM)
transformed_pre6 = deep_learning_models.transformResult(prediction6)
#end_time6 = time.clock()
#stacked_LSTM_performance = end_time6 - start_time6 
prediction6_re = np.argmax(prediction6, axis=1)
stacked_LSTM_accuracy, stacked_LSTM_precision, stacked_LSTM_recall, stacked_LSTM_F1, stacked_LSTM_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre6, transformed_y)
print("Precision for stacked LSTM")
print(stacked_LSTM_accuracy)
print(stacked_LSTM_precision)
print(stacked_LSTM_recall)
print(stacked_LSTM_fbeta)
print(stacked_LSTM_F1)

accuracy_pair = list()
accuracy_pair.append("stacked LSTM")
accuracy_pair.append(stacked_LSTM_accuracy)
accuracies.append(accuracy_pair)
## get Confusion matrix" 
cnf_matrix_6 = confusion_matrix(transformed_y, transformed_pre6)
plt.figure()
deep_learning_models.plot_confusion_matrix(cnf_matrix_6, classes=class_names, normalize=True,
                      title='Confusion matrix for stacked LSTM accuracy')
plt.savefig('Confusion matrix for stacked LSTM accuracy.png', dpi=300)
plt.show()
plot_precision_recall.plot_average_precision(y_test, prediction6, 'stacked LSTM')

###### Traditional machine learning algorithm #############
### Decision tree ##### predict_proba
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, transformed_y_train)
dtree_predictions = dtree_model.predict(X_test)
transformed_pre7 = dtree_predictions
# creating a confusion matrix
cm_dt = confusion_matrix(transformed_y, transformed_pre7)
dt_accuracy, dt_precision, dt_recall, dt_F1, dt_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre7, transformed_y)
accuracy_pair = list()
accuracy_pair.append("Decision tree")
accuracy_pair.append(dt_accuracy)
accuracies.append(accuracy_pair)
print("precision for decision tree")
print(dt_accuracy)
print(dt_precision)
print(dt_recall)
print(dt_fbeta)
print(dt_F1)
plt.figure()
deep_learning_models.plot_confusion_matrix(cm_dt, classes=class_names, normalize=True,
                      title='Confusion matrix for decision tree')
plt.savefig('Confusion matrix for decision tree.png', dpi=300)
plt.show()
y_score_dt = dtree_model.predict_proba(X_test)
plot_precision_recall.plot_average_precision(y_test, y_score_dt, 'Decision tree')

### svm ### decision function
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, transformed_y_train)
svm_predictions = svm_model_linear.predict(X_test)
transformed_pre8 = svm_predictions
# creating a confusion matrix
cm_svm = confusion_matrix(transformed_y, transformed_pre8)
svm_accuracy, svm_precision, svm_recall, svm_F1, svm_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre8, transformed_y)
accuracy_pair = list()
accuracy_pair.append("Support vector machine")
accuracy_pair.append(svm_accuracy)
accuracies.append(accuracy_pair)
print("precision for SVM")
print(svm_accuracy)
print(svm_precision)
print(svm_recall)
print(svm_fbeta)
print(svm_F1)
plt.figure()
deep_learning_models.plot_confusion_matrix(cm_svm, classes=class_names, normalize=True,
                      title='Confusion matrix for SVM')
plt.savefig('Confusion matrix for SVM.png', dpi=300)
plt.show()
y_score_svm = svm_model_linear.decision_function(X_test)
plot_precision_recall.plot_average_precision(y_test, y_score_svm, 'SVM')

### KNN ###
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, transformed_y_train) 
# creating a confusion matrix
knn_predictions = knn.predict(X_test) 
transformed_pre9 = knn_predictions
cm_knn = confusion_matrix(transformed_y, transformed_pre9)
knn_accuracy, knn_precision, knn_recall, knn_F1, knn_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre9, transformed_y) 
accuracy_pair = list()
accuracy_pair.append("KNN")
accuracy_pair.append(knn_accuracy)
accuracies.append(accuracy_pair)
print("precision for KNN")
print(knn_accuracy)
print(knn_precision)
print(knn_recall)
print(knn_fbeta)
print(knn_F1)
plt.figure()
deep_learning_models.plot_confusion_matrix(cm_knn, classes=class_names, normalize=True,
                      title='Confusion matrix for KNN')
plt.savefig('Confusion matrix for KNN.png', dpi=300)
plt.show()

y_score_knn = knn.predict_proba(X_test)
plot_precision_recall.plot_average_precision(y_test, y_score_knn, 'KNN')


# training a Naive Bayes classifier
gnb = GaussianNB().fit(X_train, transformed_y_train)
gnb_predictions = gnb.predict(X_test)
transformed_pre10 = gnb_predictions
# creating a confusion matrix
gnb_cm = confusion_matrix(transformed_y, transformed_pre10)
gnb_accuracy, gnb_precision, gnb_recall, gnb_F1, gnb_fbeta = deep_learning_models.getAccuracyMulti(transformed_pre10, transformed_y) 
accuracy_pair = list()
accuracy_pair.append("Naive Bayes")
accuracy_pair.append (gnb_accuracy)
accuracies.append(accuracy_pair)
print("precision for Naive Bayes")
print(gnb_accuracy)
print(gnb_precision)
print(gnb_recall)
print(gnb_fbeta)
print(gnb_F1)
plt.figure()
deep_learning_models.plot_confusion_matrix(gnb_cm, classes=class_names, normalize=True,
                      title='Confusion matrix for Naive Bayes')
plt.savefig('Confusion matrix for Naive Bayes.png', dpi=300)
plt.show()
y_score_nb = gnb.predict_proba(X_test)
plot_precision_recall.plot_average_precision(y_test, y_score_nb, 'Naive Bayes')