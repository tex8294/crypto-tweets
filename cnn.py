from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import string
import gensim
from gensim import corpora
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from six.moves import xrange  # pylint: disable=redefined-builtin
import csv
import numpy as np
import re



dim_e=100 #dimension of the embedding space

n_m=20000     #number of mails 

v_dim=1000  #validation set dimension


d={'ham':1, 'spam':0}  #dictionary


#read csv file

def read_csv(filename,n_m):
  with open(filename) as csvfile:         
      readCSV = csv.reader(csvfile, delimiter='\t')
      values = []
      text = []
      for row in readCSV:
        
          a=float(row[1])
          b = row[3]
        
          values.append(a)
          text.append(b)

  return values[0:n_m],text[0:n_m]



values,mails=read_csv('sent3.csv',n_m)

labels=np.asarray(values)




mails0=mails[0:n_m-v_dim]  #we don't take the validation set for training 

text_mails=[s for sentence in mails0 for s in sent_tokenize(sentence)]





#from urllib import request
#url = "http://www.gutenberg.org/files/2554/2554-0.txt"
#response = request.urlopen(url)
#text0 = response.read().decode('utf8')
f = open('sample.txt')  #read the txt file
text0=f.read()



text1=text_mails +sent_tokenize(text0)
#text1=text_mails 







#tokenize, remove punctuation, common words and very rare words
stoplist = set('for a of the and to in ! " # $ % & ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~'.split())
texts = [[word.lower() for word in word_tokenize(document) if word not in stoplist] for document in text1]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]




#train the model

model = gensim.models.Word2Vec(texts, min_count=5, iter=20, size=dim_e, sorted_vocab=1)



#use the model to calculate similiarities between words

words1=['three', 'take', 'man', 'young', 'am']
words2=['two', 'get', 'dog', 'old', 'is']

for n in range(0,len(words1)):
    try:
         print('similarity(',words1[n],',',words2[n],')=',model.similarity(words1[n], words2[n]))
         
    except KeyError:
        print("Oops!", words1[n], 'or', words2[n], 'is not in the vocabulary.')

print('doesnt_match(dinner we breakfast lunch)=',model.doesnt_match("dinner we breakfast lunch".split()))

print('most_similar(take)=',model.most_similar('take'))

print('most_similar(am)=',model.most_similar('am'))



#sort vectors by frequency

l_k=list(model.wv.vocab.keys())

def f_key(x):
    return model.wv.vocab[x].count

X = model[model.wv.vocab]

   
label=list(reversed(sorted(l_k,key=f_key)))





X_sort=X
for j in range(0,len(label)):
    X_sort[j,:]=model[label[j]]


    
#tsne visualize


plot_i=0
plot_f=500

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_sort[plot_i:plot_f,:])

def plot_with_labels(low_dim_embs, labels):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')


  plt.show()



plot_with_labels(X_tsne, label[plot_i:plot_f])




# SVM

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

text_mails0=[[s  for s in sent_tokenize(mail)] for mail in mails]
text_mails=text_mails0[0:n_m]

words = [[word.lower() for sentence in mail for word in word_tokenize(sentence) if word not in stoplist] for mail in text_mails]

frequency = defaultdict(int)
for text in words:
    for token in text:
        frequency[token] += 1

words = [[token for token in text if frequency[token] > 1] for text in words]




#sum of the word vectors for each mail

mail_vectors=np.zeros((n_m,dim_e)); 

for j in range(0,n_m):
    m=words[j]
    for k in range(0,len(m)):
      try:
        mail_vectors[j,:]=mail_vectors[j,:]+np.asarray(model[m[k]])
      except KeyError:
        pass
        #print(m[k],'not in vocabulary')


#clf = SVC(kernel='linear', C = 1.0)
#clf=SVC(gamma=2, C=1)      #worst
#clf=LogisticRegression()   #best
#clf= DecisionTreeClassifier(max_depth=5)
#clf= KNeighborsClassifier(3)
#clf= GaussianProcessClassifier(1.0 * RBF(1.0))
#clf=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) 
#clf=AdaBoostClassifier()    #0.39
clf=MLPClassifier(alpha=1)    #0.258 with sample

#to train the model we only take the first n_m-v_dim mails (the training set)
print('training')
clf.fit(mail_vectors[0:n_m-v_dim],labels[0:n_m-v_dim])



#validation

print("validation error=",np.sum(np.abs(labels[n_m-v_dim:n_m]-clf.predict(mail_vectors[n_m-v_dim:n_m,:])))/v_dim)


print("s=",clf.predict(mail_vectors[1:100,:]))
print("l=",labels[1:100])



#read csv file

def read_csv1(filename):
  with open(filename) as csvfile:         
      readCSV = csv.reader(csvfile, delimiter='\t')
      text = []
      price = []
      for row in readCSV:
        a=row[2]
        b=row[3]
        text.append(a)
        price.append(b)
        
  return text, price

tweets, price =read_csv1('tweet-price.csv') #time is an array with the day for each tweet 

text_mails=[[s  for s in sent_tokenize(mail)] for mail in tweets]


#calculate the sentiments and word vectors for each tweet

words = [[word.lower() for sentence in mail for word in word_tokenize(sentence) if word not in stoplist] for mail in text_mails]

frequency = defaultdict(int)
for text in words:
    for token in text:
        frequency[token] += 1

words = [[token for token in text if frequency[token] > 1] for text in words]

tweet_vectors=np.zeros((len(words),dim_e));

tweet_sentiment=np.zeros(len(words))



for j in range(0,len(words)):
    m=words[j]                  #word vectors for each tweet
    for k in range(0,len(m)):
      try:                    
        tweet_vectors[j,:]=tweet_vectors[j,:]+np.asarray(model[m[k]])
      except KeyError:
        pass


tweet_sentiment=clf.predict(tweet_vectors);   #sentiments

print('sentiments=',tweet_sentiment)

X_train=np.asarray(tweet_sentiment)

Y_train=np.asarray(price)
  


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import TFOptimizer

coinarray=[]

with open('bitcoin3.csv', newline='') as coinfile:
    coinreader=csv.reader(coinfile, delimiter='\t', quotechar='|')
    for row in coinreader:
        coinarray.append(','.join(row))

coinsize=len(coinarray)

        
#(x_train, y_train), (x_test, y_test) = #data input here training data will be tweets(x) and the labels are the previous price(y). 
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





#another possibility: use word vectors as X_train



