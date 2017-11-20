# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:14:57 2017

@author: LeBonAT430
"""
#import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import TFOptimizer

coinarray=[]

with open('bitcoin3.csv', newline='') as coinfile:
    coinreader=csv.reader(coinfile, delimiter='\t', quotechar='|')
    for row in coinreader:
        coinarray.append(','.join(row))

coinsize=len(coinarray)

        
(x_train, y_train), (x_test, y_test) = #data input here training data will be tweets(x) and the labels are the previous price(y). 
#possibly need to pre-process data depending on formatting also consider time shifts

linear_model_keras = Sequential()

linear_model_keras.add(Dense(10, input_shape=('input formatting here'), kernel_initializer='zeroes', bias_initializer='zeros'))
linear_model_keras.add(Activation('softmax'))

GradientDescent = TFOptimizer(tf.train.GradientDescentOptimizer(0.5))

linear_model_keras.compile(loss='binary_crossentropy', optimizer=GradientDescent, metrics=['acc'])
#we should maybe use a different error function depending on how we want our output

linear_model_keras.fit(x_train, y_train, batch_size=100, epochs=10)
#begins the training
loss_and_metrics = linear_model_keras.evaluate(x_test, y_test, batch_size=128)

print('\n Loss: {}, Accuracy: {}'.format(loss_and_metrics[0],loss_and_metrics[1]))