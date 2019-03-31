#!/usr/bin/env python
# coding: utf-8

# # Text Classification Using Word Embeddings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
import Algorithmia
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from time import sleep

cv = 5
client = Algorithmia.client('simpn4Yfv9BqAZjoYoE8aaPO0PC1')
algo = client.algo('nlp/ProfanityDetection/1.0.0')


class MeanEmbeddingTransformer(TransformerMixin):
    
    def __init__(self):
        self._vocab, self._E = self._load_words()

        
    
    def _load_words(self):
        E = {}
        vocab = []

        bad_word_list = []
        with open('/Users/aggarwalpiush/github_repos/offensivetextevaluation/data/bad-words/bad_words_cycle4.txt', 'r', encoding="utf8") as bfile:
            for bad_word in bfile:
                bad_word_list.append(bad_word.split('\t')[0].split('_')[0].strip().lower())

        off_words = algo.pipe([' '.join(bad_word_list)]).result

        with open('/Users/aggarwalpiush/embeddings/train_preprocessed.vec', 'r', encoding="utf8") as file:
            for i, line in enumerate(file):
                l = line.split(' ')
                if l[0].isalpha():
                    if l[0] in bad_word_list:
                        v = [float(i) for i in l[1:]] + [float(1)]
                    else:
                        v = [float(i) for i in l[1:]] + [float(-1)]
                    if  l[0] in off_words.keys():
                        v = v + [float(1)]
                    else:
                        v = v + [float(-1)]

                    E[l[0].lower()] = np.array(v)
                    vocab.append(l[0].lower())
        return np.array(vocab), E            

    
    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None
    
    def _doc_mean(self, doc):
        word_array = []
        for w in doc:
            if  w.lower().strip() in self._E.keys():
                word_array.append(self._E[w.lower().strip()])
            else:
                word_array.append(np.zeros([len(v) for v in self._E.values()][0]))

        return np.sum(np.array(word_array), axis=0)
                
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)



def plot_roc(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
def print_scores(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Precision score: {:3f}'.format(precision_score(y_test, y_pred, average='macro') ))
    print('Recall score: {:3f}'.format(recall_score(y_test, y_pred, average='macro') ))
    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred)))
    print('AUC score: {:3f}'.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred)))



train_data = pd.read_csv('/Users/aggarwalpiush/github_repos/offensivetextevaluation/data/train_data/train.tsv_preprocessed', sep='\t',
                        dtype={'tweet': object,  'id': np.int32,
                              'subtask_a': 'category'})


X = train_data[['tweet']].as_matrix()
y = train_data['subtask_a'].as_matrix()



def tokenize_and_transform(X, sample_size):
    tweets = X[:,0]
    tok_tweet = []
    for doc in tweets[:sample_size]:
        if type(doc) == str:
            tok_tweet.append(word_tokenize(doc))
        else:
            tok_tweet.append(word_tokenize("None None None None"))
    met = MeanEmbeddingTransformer()
    X_transform = met.fit_transform(tok_tweet)
    return X_transform


X_transform = tokenize_and_transform(X, 160000)
print(X_transform[:10])


np.savetxt('X_embed.csv', X_transform, delimiter='\t')


X_transform = np.loadtxt('X_embed.csv', delimiter='\t')



X_transform = scale(X_transform)
le = LabelEncoder()
y = le.fit_transform(y)




# for i in range(cv):
#     X_train, X_test, y_train, y_test = train_test_split(X_transform, y[:X_transform.shape[0]], stratify=y[:X_transform.shape[0]], random_state=0)
#     rus = RandomOverSampler(random_state=0)
#     X_train, y_train = rus.fit_sample(X_train, y_train)

#     svc = SVC().fit(X_train, y_train)
#     print_scores(svc, X_train, y_train, X_test, y_test)
#     plot_roc(svc, X_test, y_test)





rus = RandomOverSampler(random_state=42)
X_resample, y_resample = rus.fit_sample(X_transform, y[:X_transform.shape[0]])


# X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, stratify=y_resample, test_size=0.02, random_state=42)

svc = SVC().fit(X_resample, y_resample)
# print_scores(svc, X_train, y_train, X_test, y_test)
# plot_roc(svc, X_test, y_test)

filename = 'svm_model.sav'
pickle.dump(svc, open(filename, 'wb'))


test_data = pd.read_csv('/Users/aggarwalpiush/github_repos/offensivetextevaluation/data/train_data/dev.tsv_preprocessed', sep='\t',
                        dtype={'tweet': object,  'id': np.int32,
                              'subtask_a': 'category'})


X_sub = test_data[['tweet']].as_matrix()
print(X_sub[:10])



X_transform_sub = tokenize_and_transform(X_sub, X_sub.shape[0])

X_transform_sub = scale(X_transform_sub)



loaded_model = pickle.load(open('svm_model.sav', 'rb'))
y_pred = loaded_model.predict(X_transform_sub)

out_data = np.column_stack((test_data['id'].as_matrix(), y_pred))

out_data = [[x[0],'OFF' if x[1]==1 else 'NOT'] for x in list(out_data)]

np.savetxt('submission.csv', out_data, fmt='%s,%s', delimiter=',')


y_true = test_data[['subtask_a']].as_matrix()
le = LabelEncoder()

y_true = le.fit_transform(y_true)
print('F1 score: {:3f}'.format(f1_score(y_true, y_pred)))

off_list = ['fuck yeah', 'user hit', 'holy shit', 'holder prison', 'piece shit', 'crazy liberal', 'eric holder', 'white person', 'get shit', 'ye bitch', 'person die', 'go jail', 'liberal lie', 'make look', 'liberal really', 'mentally unstable', 'fuck url', 'gun control right', 'terrible person'] 
not_off_list = ['gun control', 'user beautiful', 'conservative url', 'boycott nfl', 'join antifa', 'user adorable', 'user gorgeou', 'great day', 'please follow', 'good riddance', 'conservative conservative', 'liberal call', 'please follow back', 'god hope', 'democrat antifa', 'best player', 'president trump maga', 'thank follow', 'correct url', 'welcome url']

fn = 0
tp = 0
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        for num,j in enumerate(str(X_sub[i]).split(' ')):
            flag = 0
            for o in not_off_list:
                if num < len(str(X_sub[i]).split(' '))-1:
                    if j.strip('[').strip(']').strip("'") +' '+str(X_sub[i]).split(' ')[num+1].strip('[').strip(']').strip("'") in o:
                        flag = 1
                        break
            if flag == 1:
                if y_true[i] == 0:
                    fn += 1
                if y_true[i] == 1:
                    tp += 1
                # print(X_sub[i])
                # print(y_true[i])
                # sleep(2)
                break
print(fn)
print(tp)










