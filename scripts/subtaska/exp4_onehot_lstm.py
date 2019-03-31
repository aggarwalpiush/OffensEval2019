#! usr/bin/env python
# *--coding : utf-8 --*

from __future__ import division
import codecs
import sys
import numpy as np
import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

ps = PorterStemmer()
tknzr = TweetTokenizer()
MAX_SIZE = 40
REDUCED_SVD_DIM = 20
MAX_WORDS = 1000
EMBEDDING_DIMENSION = 300
kernel_size = 3
filters = 64
pool_size = 4


def tokenize_and_stemize(tweet_list):
	tweet_list_out = []
	for tweet in tweet_list:
		tweet_vect = []
		for word in tknzr.tokenize(tweet):
			tweet_vect.append(ps.stem(str(word).strip().lower()))
		tweet_list_out.append(' '.join(tweet_vect))
	return tweet_list_out

def get_encoding_tweet_vector(tweet, lexicon_list):
	tweet = [ps.stem(str(tweet_word).strip().lower()) for tweet_word in tweet]
	return list(map(lambda x: 1 if x in lexicon_list else 0, tweet))

def get_padding(tweet_vector):
	len_tweet_vector = len(tweet_vector)
	if len_tweet_vector > MAX_SIZE:
		return tweet_vector[:MAX_SIZE]
	elif len_tweet_vector < MAX_SIZE:
		return (tweet_vector + [0] * (MAX_SIZE - len_tweet_vector))
	else: 
		return tweet_vector

def apply_svd(tweet_vector_matrix):
	u, s, _ = np.linalg.svd(tweet_vector_matrix, full_matrices=True)
	return np.matmul(u[:,:REDUCED_SVD_DIM], np.diag(s[:REDUCED_SVD_DIM]))


def LSTM_model_seq(vocabulary_size, max_words, embedding_matrix, embedding_dimension):
    model = Sequential()
    model.add(Embedding(vocabulary_size,embedding_dimension,input_length=max_words, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.2))
    # model.add(Conv1D(64, 5, activation='relu'))
    # model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(100))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Conv1D(128,
					 kernel_size,
					 padding='valid',
					 activation='relu',
					 strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dense(80,name='FC1'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(40,name='FC2'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(20,name='FC3'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model.add(Dense(1,name='out_layer'))
    # model.add(Activation('sigmoid'))
    return model


def main():
	input_path_pretrained= sys.argv[1]
	lexicon_path = sys.argv[2]
	input_path = sys.argv[3]
	df = pd.read_csv(input_path_pretrained,encoding='utf-8', delimiter='|', engine='python')
	print(df.info)
	df.drop('id',axis=1,inplace=True)
	X = df.tweet.astype(str) 
	Y = df.subtask_a
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	print(X.shape)
	#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

	lexicon_list = []
	pos_list = []
	with codecs.open(lexicon_path, 'r', 'utf-8') as lexicon_obj:
		for line in lexicon_obj:
			lexicon_list.append([ps.stem(str(line.split('\t')[0].split('_')[0].rstrip('\r\n')).strip().lower()),
				float(line.split('\t')[1].rstrip('\r\n')), ps.stem(str(line.split('\t')[0].split('_')[1].rstrip('\r\n')).strip().lower())])
			pos_list.append(ps.stem(str(line.split('\t')[0].split('_')[1].rstrip('\r\n')).strip().lower()))
	print('get_hot_encoders..')


	cnt = Counter()
	for word in pos_list:
	    cnt[word] += 1

	# lexicon_list = [ps.stem(str(word).strip().lower()) for word in lexicon_list]
	# X = [tknzr.tokenize(tweet) for tweet in X]
	# X = [get_encoding_tweet_vector(tweet, lexicon_list) for tweet in X]
	# embedding_matrix = [get_padding(tweet) for tweet in X]
	# embedding_matrix = apply_svd(sequences_matrix)
	tok = Tokenizer(num_words=MAX_WORDS)
	tok.fit_on_texts(X)


	# create embedding matrix

	sequences = tok.texts_to_sequences(X)
	sequences_matrix = sequence.pad_sequences(sequences,maxlen=MAX_WORDS)
	vocabulary_size = max(x for s in sequences_matrix for x in s)

	maximum_pos = max(cnt.values())
	
	embeddings_index = dict()
	for word in lexicon_list:
	 	embeddings_index[word[0]] = [word[1], float(cnt[word[2]])/maximum_pos]

	count = 0

	for i,j in embeddings_index.items():
		print("%s:%s" %(i,j))
		count += 1
		if count > 10:
			break

	embedding_dimension = len(list(embeddings_index.values())[0])
	embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))
	for word, index in tok.word_index.items():
	    if index > vocabulary_size - 1:
	        break
	    else:
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[index] = embedding_vector


	print(sequences_matrix.shape)

	model = LSTM_model_seq(vocabulary_size, MAX_WORDS, embedding_matrix, embedding_dimension)
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

	model.fit(sequences_matrix,Y,batch_size=128,epochs=15,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

	df_post = pd.read_csv(input_path,delimiter='\t',encoding='utf-8')
	df_post.drop('id',axis=1,inplace=True)
	X = df_post.tweet
	Y = df_post.subtask_a
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)
	sequences = tok.texts_to_sequences(X_train)
	sequences_matrix = sequence.pad_sequences(sequences,maxlen=MAX_WORDS)
	model.fit(sequences_matrix,Y_train,batch_size=128,epochs=15,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

	test_sequences = tok.texts_to_sequences(X_test)
	test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=MAX_WORDS)
	accr = model.evaluate(test_sequences_matrix,Y_test)

	print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


if __name__ == '__main__':
	main()



