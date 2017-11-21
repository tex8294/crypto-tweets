# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:14:57 2017

@author: LeBonAT430
"""
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import TFOptimizer
import numpy as np

coinarrays=[]
coinarrayp=[]
with open('sentarray.csv', newline='') as coinfile:
    coinreader=csv.reader(coinfile, delimiter='\t', quotechar='|')
    for row in coinreader:
        coinarrays.append(float(row[0]))
        coinarrayp.append(float(row[1]))

coinlen=len(coinarrays)-801
       

x_train = np.transpose(coinarrays[801:])
y_train = np.transpose(coinarrayp[801:])
x_test = coinarrays[0:800]
y_test = coinarrayp[0:800]

model = Sequential()

model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['mean_squared_error'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=800)
score = model.evaluate(x_test, y_test, batch_size=800)