#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:16:53 2018

@author: lixiaodan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:22:28 2018

@author: lixiaodan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
import deep_learning_models
import matplotlib.pyplot as plt

datasets = list()
wikidata_writing_style = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_writing_style.csv')
wikidata_text_statiscs = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_text_statiscs.csv')
wikidata_structure = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_structure.csv')
wikidata_read_scores = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_read_scores.csv')
wikidata_network_features = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_network_features.csv')
wikidata_edit_hist = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_edit_hist.csv')
datasets.append(wikidata_writing_style)
datasets.append(wikidata_text_statiscs)
datasets.append(wikidata_structure)
datasets.append(wikidata_read_scores)
datasets.append(wikidata_network_features)
datasets.append(wikidata_edit_hist)

precisions = list()
names = ["Writing style", "Text statistics", "Structure feature", "Readability scores", "Network features", "Edit history"]
markers = ['o', ',', 's', '*', '<', '.']

for index, wikidata in enumerate(datasets):
    labels = wikidata["page_class"]
    for i in range(labels.shape[0]):
        if labels[i] == 'FA' or labels[i] == 'AC' or labels[i] == 'GA':
            labels.loc[i] = '0'
        elif labels[i] == 'BC' or labels[i] == 'ST' or labels[i] == 'SB':
            labels.loc[i] = '1'
    
    labels = labels.convert_objects(convert_numeric=True)
    onehotlabels = np_utils.to_categorical(labels)
    
    ### preprocess features
    features = wikidata.iloc[:, 1:]
    feature_names = list(features)
    print(feature_names)
    min_max_scaler = preprocessing.MinMaxScaler()
    features_minmax = min_max_scaler.fit_transform(features)
    
    ### split data into training set and label set
    X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.4, random_state=42)
    
    ### adjust the dataset dimension
    # reshape X to be [samples, time steps, features]
    X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    ### create the deep learning models
    epochs = 15
    batch_size = 195 #195 best now #190 # 100 # 250
    dropoutRate = 0.2
    
    ## stacked LSTM
    model, hist = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
    prediction = model.predict(X_test_LSTM)
    prediction_re = np.argmax(prediction, axis=1)
    stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction, y_test)
    print("Precision for stacked LSTM")
    print(stacked_LSTM_precision)
    precisions.append(stacked_LSTM_precision)
    
    cur_label = names[index]
    cur_marker = markers[index]
    print("Model accuracy")
    plt.plot(hist.history['acc'], marker = cur_marker, label= cur_label, markersize=10)
    
plt.title('Model training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(shadow=False, fontsize='xx-small', loc='lower right', prop={'size': 7})
plt.savefig("Model_accuracy.png")
plt.show()

for index, wikidata in enumerate(datasets):
    labels = wikidata["page_class"]
    for i in range(labels.shape[0]):
        if labels[i] == 'FA' or labels[i] == 'AC' or labels[i] == 'GA':
            labels.loc[i] = '0'
        elif labels[i] == 'BC' or labels[i] == 'ST' or labels[i] == 'SB':
            labels.loc[i] = '1'
    
    labels = labels.convert_objects(convert_numeric=True)
    onehotlabels = np_utils.to_categorical(labels)
    
    ### preprocess features
    features = wikidata.iloc[:, 1:]
    feature_names = list(features)
    print(feature_names)
    min_max_scaler = preprocessing.MinMaxScaler()
    features_minmax = min_max_scaler.fit_transform(features)
    
    ### split data into training set and label set
    X_train, X_test, y_train, y_test = train_test_split(features_minmax, onehotlabels, test_size=0.4, random_state=42)
    
    ### adjust the dataset dimension
    # reshape X to be [samples, time steps, features]
    X_train_LSTM = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_LSTM = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    ### create the deep learning models
    epochs = 15
    batch_size = 195 #195 best now #190 # 100 # 250
    dropoutRate = 0.2
    
    ## stacked LSTM
    model, hist = deep_learning_models.stacked_LSTM(X_train_LSTM, y_train, X_test_LSTM, y_test, batch_size, epochs)
    prediction = model.predict(X_test_LSTM)
    prediction_re = np.argmax(prediction, axis=1)
    stacked_LSTM_precision = deep_learning_models.getAccuracy(prediction, y_test)
    print("Precision for stacked LSTM")
    print(stacked_LSTM_precision)
    #precisions.append(stacked_LSTM_precision)
    
    cur_label = names[index]
    cur_marker = markers[index]
    print("Model loss")
    plt.plot(hist.history['loss'], marker = cur_marker, label= cur_label, markersize=10)
    
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(shadow=False, fontsize='xx-small', loc='lower left', prop={'size': 7})
plt.savefig("Model_loss.png")
plt.show() 
print(precisions)   