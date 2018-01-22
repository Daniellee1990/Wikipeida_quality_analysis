#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:10:15 2018

@author: lixiaodan
"""
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.utils import np_utils
from keras.layers import Dropout
from keras.optimizers import SGD


seed = 7
np.random.seed(seed)

# define baseline model
def DNN():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=33, init='normal', activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_with_all_features.csv')
wikidata = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/wikipedia_without_network.csv')

### process one hot shot 
#labels = wikidata.select_dtypes(include=[object])
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

X_train, X_test, Y_train, Y_test = train_test_split(features_minmax, onehotlabels, test_size=0.33, random_state=seed)

epochs = 50
batch_size = 30
dropoutRate = 0.2
numHidden = 40

##  neural network
"""
estimator = KerasClassifier(build_fn=DNN, nb_epoch=100, batch_size=5, verbose=0)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
print(predictions)
"""

### DNN
model = Sequential() 
model.add(Dense(20, input_dim=X_train.shape[1], init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, init='normal', activation='relu'))
model.add(Dropout(0.2))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.add(Dense(Y_train.shape[1], init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, batch_size, epochs)
predictions = model.predict(X_test)

test_size = X_test.shape[0]
col_size = Y_test.shape[1]
correct_num = 0
wrong_num = 0

for i in range(test_size):
    cur_row = Y_test[i,:]
    cur_index = 0
    for j in range(col_size):
        if cur_row[j] == 1:
            cur_index = j
            break
    pre_id = 0
    pre_max = 0
    for k in range(col_size):
        if predictions[i, k] > pre_max:
            pre_max = predictions[i, k]
            pre_id = k
    if cur_index == k:
        correct_num = correct_num + 1
    else:
        wrong_num = wrong_num + 1

accuracy = float(correct_num) / test_size
print(accuracy)