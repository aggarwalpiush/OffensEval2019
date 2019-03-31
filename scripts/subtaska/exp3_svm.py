#! usr/bin/env python
# *-- coding : utf-8 --*

import pandas as pd
import sys
import re, nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


import collections
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
count_vectorizer = CountVectorizer(ngram_range=(1,2))


def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams


def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

def offesivity2target(offesivity):
    return {
        'OFF': 0,
        'NOT': 1
    }[offesivity]



input_file = sys.argv[1]
tweets = pd.read_csv(input_file, sep='\t')
list(tweets.columns.values)
tweets.head()
tweets['normalized_tweet'] = tweets.tweet.apply(normalizer)
tweets[['tweet','normalized_tweet']].head()
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets[['grams']].head()
tweets[(tweets.subtask_a == 'OFF')][['grams']].apply(count_words)['grams'].most_common(20)
tweets[(tweets.subtask_a == 'NOT')][['grams']].apply(count_words)['grams'].most_common(20)

vectorized_data = count_vectorizer.fit_transform(tweets.tweet)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))



targets = tweets.subtask_a.apply(offesivity2target)
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]

clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)
clf.score(data_test, targets_test)
