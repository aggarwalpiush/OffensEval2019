#! usr/bin/env python
# *--coding : utf-8 --*

import codecs
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

EMBEDDING_DIMENSION = 50

def RNN(max_len, vocabulary_size, embedding_matrix):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(vocabulary_size,100,input_length=max_len, weights=[embedding_matrix], trainable=False)(inputs)
    layer = Dropout(0.2)(layer)
    layer = Conv1D(64, 5, activation='relu')(layer)
    layer = MaxPooling1D(pool_size=4)(layer)
    layer = LSTM(100)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('sigmoid')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def RNN_seq(max_len, vocabulary_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocabulary_size,EMBEDDING_DIMENSION,input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(256,name='FC1'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1,name='out_layer'))
    model.add(Activation('sigmoid'))
    return model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def main():
	input_file = sys.argv[1]
	test_file = sys.argv[2]
	embedding_path = sys.argv[3]
	embeddings_index = dict()
	f = open(embedding_path)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	df = pd.read_csv(input_file,delimiter='\t',encoding='utf-8')
	df.head()
	df.drop(['id'],axis=1,inplace=True)
	df.info()
	sns.countplot(df.subtask_a)
	plt.xlabel('Label')
	plt.title('Number of offensive and non-offensive tweets')
	X = df.tweet
	Y = df.subtask_a
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	Y = Y.reshape(-1,1)
	if input_file == test_file:
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15, random_state=12)
	else:
		X_train = X
		Y_train = Y
		df_test = pd.read_csv(test_file,delimiter='\t',encoding='utf-8')
		df_test.head()
		df_test.drop(['id'],axis=1,inplace=True)
		X_test = df_test.tweet
		Y_test = df_test.subtask_a
		Y_test = le.transform(Y_test)
		Y_test = Y_test.reshape(-1,1)

	max_words = 1000
	max_len = 150
	tok = Tokenizer(num_words=max_words)
	tok.fit_on_texts(X_train)
	

	vocabulary_size = len(embeddings_index.keys())
	print(vocabulary_size)
	sequences = tok.texts_to_sequences(X_train)
	embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIMENSION))
	for word, index in tok.word_index.items():
	    if index > vocabulary_size - 1:
	        break
	    else:
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[index] = embedding_vector

	sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
	model = RNN_seq(max_len, vocabulary_size, embedding_matrix)
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
	model.fit(sequences_matrix,Y_train,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
	test_sequences = tok.texts_to_sequences(X_test)
	test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
	accr = model.evaluate(test_sequences_matrix,Y_test)
	print('\nTest set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
	plt.show()
	

if __name__ == '__main__':
	main()
