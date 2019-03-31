#
# Copyright 2018
# Ubiquitous Knowledge Processing (UKP) Lab
# Technische Universitat Darmstadt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.



# This is an adapted demo test case from the Keras project (https://keras.io)
#!usr/bin/env python
# *-- coding: utf-8 --*

from sys import argv
import argparse
from keras.preprocessing.text import Tokenizer

import numpy as np

# Embedding
max_features = 20000
#maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 3
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 128

# Training
batch_size = 32
epochs = 10

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

def numpyizeDataVector(vec):
	trainVecNump=[]
	file = open(vec, 'r')
	for l in file.readlines():
		l = l.strip()
		trainVecNump.append(np.fromstring(l, dtype=int, sep=' '))
	file.close()
	return trainVecNump
	
def numpyizeOutcomeVector(vec):
	file = open(vec, 'r')
	v=""
	for l in file.readlines():
		l = l.strip()
		v=np.fromstring(l, dtype=int, sep=' ')
	file.close()
	return v
	
def loadEmbeddings(emb):
	matrix = {}	
	f = open(emb, 'r')
	embData = f.readlines()
	f.close()
	dim = len(embData[0].split())-1
	matrix = np.zeros((len(embData)+1, dim))	
	for e in embData:
		e = e.strip()
		if not e:
			continue
		idx = e.find(" ")
		id = e[:idx]
		vector = e[idx+1:]
		matrix[int(id)]=np.asarray(vector.split(" "), dtype='float32')
	return matrix, dim

def runExperiment( inputData, embedding, maximumLength, predictionOut):

	from keras.preprocessing import sequence
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Embedding
	from keras.layers import LSTM
	from keras.layers import Conv1D, MaxPooling1D
	from keras.datasets import imdb
	from sklearn.preprocessing import LabelEncoder
	from sklearn.model_selection import train_test_split

	# trainVecNump = numpyizeDataVector(trainVec)
	# trainOutcome = numpyizeOutcomeVector(trainOutcome)
	
	# testVecNump = numpyizeDataVector(testVec)
	# testOutcome = numpyizeOutcomeVector(testOutcome)
	
	# if embedding:
	# 	print("Load pretrained embeddings")
	# 	embeddings,dim = loadEmbeddings(embedding)
	# 	EMBEDDING_DIM = dim
	# else:
	# 	print("Train embeddings on the fly")
	# 	EMBEDDING_DIM = 50

	from keras.preprocessing.text import Tokenizer
	import codecs
	import pandas as pd

	df_post = pd.read_csv(inputData, delimiter='\t', encoding='utf-8')
	df_post.drop('id',axis=1,inplace=True)
	X = df_post.tweet
	Y = df_post.subtask_a
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.15)


	tok = Tokenizer(num_words=maximumLength)
	tok.fit_on_texts(X)



	embeddings_index = dict()
	with codecs.open(embedding, 'r', 'utf-8') as emb_obj:
		for line in emb_obj:
			embeddings_index[line.split()[0]] = line.split()[1:]

	vocabulary_size = len(embeddings_index)
	embedding_dimension = len(list(embeddings_index.values())[0])
	embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))
	for word, index in tok.word_index.items():
	    if index > vocabulary_size - 1:
	        break
	    else:
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[index] = embedding_vector
	sequences = tok.texts_to_sequences(x_train)
	sequences_matrix = sequence.pad_sequences(sequences,maxlen=maximumLength)

	test_sequences = tok.texts_to_sequences(x_test)
	test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=maximumLength)

	# x_train = sequence.pad_sequences(trainVecNump, maxlen=int(maximumLength))
	# x_test = sequence.pad_sequences(testVecNump, maxlen=int(maximumLength))
	
	# y_train = trainOutcome
	# y_test = testOutcome
	combined_sequence = tok.texts_to_sequences(X)
	combined_sequences_matrix = sequence.pad_sequences(combined_sequence,maxlen=maximumLength)
	vocabSize = max(x for s in combined_sequences_matrix for x in s)
	vocabSize = embedding_matrix.shape[0]
	print(embedding_matrix.shape)
		
	print(x_train.shape)
	print(x_test.shape)	

	print('Build model...')

	model = Sequential()
	#if embedding_matrix:
	#	print("Using pre-trained embeddings")
	model.add(Embedding(output_dim=embedding_matrix.shape[1], input_dim=vocabSize, 
		input_length=maximumLength, weights=[embedding_matrix], trainable=False))
	# else:
	# 	print("Train embeddings on-the-fly")	
	# 	model.add(Embedding(vocabSize+1, EMBEDDING_DIM))	
	model.add(Dropout(0.25))
	model.add(LSTM(lstm_output_size, return_sequences=True))
	model.add(LSTM(lstm_output_size, go_backwards=True, return_sequences=True))
	model.add(Dropout(0.25))
	model.add(Conv1D(128,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	model.add(Conv1D(filters,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
	model.add(MaxPooling1D(pool_size=pool_size))
	# model.add(Flatten())
	model.add(Dense(40))
	model.add(Dense(10))
	model.add(Flatten())
	model.add(Dense(2))
	model.add(Activation('softmax'))

	model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

	# model.summary()

	print('Train...')
	model.fit(sequences_matrix,y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_sequences_matrix, y_test),
          shuffle=True)

	score, acc = model.evaluate(test_sequences_matrix, y_test, batch_size=batch_size)
	print("score:%s\naccuracy:%s\n" %(score, acc))
	
	prediction = model.predict_classes(test_sequences_matrix)

	predictionFile = open(predictionOut, 'w')
	predictionFile.write("#Gold\tPrediction\n")
	for i in range(0, len(prediction)):
		predictionFile.write(str(y_test[i]) +"\t" + str(prediction[i])+ "\n")
	predictionFile.close()

		# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")


if  __name__ =='__main__':
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--inputData", nargs=1, required=True)   
	parser.add_argument("--embedding", nargs=1, required=True)    
	parser.add_argument("--maxLen", nargs=1, required=True)
	parser.add_argument("--predictionOut", nargs=1, required=True)    
    
    
	args = parser.parse_args()
	inputData = args.inputData[0]
	if not args.embedding:
		embedding=""
	else:
		embedding = args.embedding[0]
	maxLen = args.maxLen[0]
	predictionOut = args.predictionOut[0]
	# if not args.seed:
	# 	seed=897534793	#random seed
	# else:
	# 	seed = args.seed[0]

	
	runExperiment(inputData, embedding, int(maxLen), predictionOut)