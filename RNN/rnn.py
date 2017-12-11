
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import AveragePooling2D
from keras.layers import SimpleRNN, LSTM
from keras.models import model_from_json

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *

import numpy as np
import random
import sys

import gensim
import time

nc = 5

stop_words = set(stopwords.words("english"))
operators = set(('not','never'))
stop_words = stop_words - operators

stemmer = PorterStemmer()

def preprocess(sentences):
	for i in range(len(sentences)):
		sentences[i] = word_tokenize(sentences[i])
		# sentences[i] = [word for word in sentences[i] if word not in stop_words]
		# sentences[i] = [stemmer.stem(word) for word in sentences[i]]
	return sentences

def max_length(sentences):
	max_len=-1
	for i in range(len(sentences)):
		if len(sentences[i])>max_len:
			max_len=len(sentences[i])
	return max_len

# References for using word2vec

# https://rare-technologies.com/word2vec-tutorial/
# https://radimrehurek.com/gensim/models/word2vec.html
# http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python

start_time = time.time()

print "Training Word2Vec...."
# f = open("stanfordSentimentTreebank/original_rt_snippets.txt")
f = open("sentence_labels.txt")
# text = f.read().decode("utf-8")
sentences= []
for sentence in f:
	sentence = sentence.split("\t")[0]
	sentences.append(sentence)
# sentences = sent_tokenize(text)
sentences = preprocess(sentences)
max_len = max_length(sentences)
model = gensim.models.Word2Vec(sentences, min_count=1)
f.close()

word2vec_len = 100

def get_file_data(filename):
	sentences = []
	labels = []
	sentence_labels = open(filename)
	for sentence_label in sentence_labels:
		sentence = sentence_label.split("\t")[0]
		sentences.append(sentence)
		label = sentence_label.split("\t")[2]
		labels.append(int(label))
	sentences = preprocess(sentences)
	sentence_labels.close

	X = np.zeros((len(sentences),max_len, word2vec_len))
	y = np.zeros((len(sentences),nc))

	for i in range(len(sentences)):
		for j in range(len(sentences[i])):
			X[i,j,:] = model[sentences[i][j]]
		# print labels[i],
		y[i,labels[i]]=1

	return X,y

print "Getting Data..."
X_train, y_train = get_file_data("train.txt")
X_valid, y_valid = get_file_data("valid.txt")
X_test, y_test = get_file_data("test.txt")



print "Defining model..."
# a sequential model in keras is being used, sequential means we are building layers in a sequential manner
model = Sequential()

# now we add the layers, we dont have to specify the input layer since it can be inferred from the input data
# below step adds a hidden layer of 512 nodes 
#(SimpleRNN is to specify that we need normal recurrent cells not LSTMs or GRU)
#
## in the below layer we specify that our sequence_length=maxlen and input_Dimesnion=word2vec dimension
print "Adding Layer 1..."
# model.add(SimpleRNN(512, return_sequences=True, input_shape=(max_len, word2vec_len)))
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, word2vec_len)))


# in the second hidden layer we dont need to specify the dimensions since it will be fully connected to the one below
#return_sequences=False as our training is in many-to-one style fashion 
print "Adding Layer 2..."
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))

#OUTPUT LAYER contais just the sentiment value
print "Adding Output Layer..."
model.add(Dense(nc))

# print "Adding Mean pooling layer.."
# model.add(AveragePooling1D())

# apply softmax activation on top of the output layer
model.add(Activation('softmax'))

#specify the loss function and optimizer we want to use


print model.summary()

batch_sizes = [32,64,128]
nb_epochs = [1,2,3]
optimizers = ['rmsprop','adam']

for batch_size in batch_sizes:
	for nb_epoch in nb_epochs:
		for optimizer in optimizers:

			print "Compiling Model..."
			model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

			print "Fitting Model..."
			history=model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=0)

			print "Saving Model...."
			# serialize model to JSON
			model_json = model.to_json()
			with open("model.json", "w") as json_file:
			   json_file.write(model_json)
			#serialize weights to HDF5
			model.save_weights("model_lstm_"+str(batch_size)+"_"+str(nb_epoch)+"_"+optimizer+".h5")


			f = open("results_lstm.txt",'a')
			print "batch_size : "+str(batch_size)+"  nb_epoch : "+str(nb_epoch)+"   "+"optimizer : "+optimizer
			f.write("batch_size : "+str(batch_size)+"  nb_epoch : "+str(nb_epoch)+"   "+"optimizer : "+optimizer+"\n")
			def get_accuracy(X,y,stri):
				scores = model.evaluate(X,y, verbose=0)
				print scores

				f.write(stri+" Accuracy : "+ str(scores[1]*100)+"\n")
				#print stri," Accuracy : ",str(scores[1]*100)

			print "Testing model..."
			get_accuracy(X_train, y_train,"Training")
			get_accuracy(X_valid, y_valid, "Validation")
			get_accuracy(X_test, y_test,"Test")


			print "Time taken : ",time.time()-start_time
			f.close()


# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")

# model = loaded_model

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')