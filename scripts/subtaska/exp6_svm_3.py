#!/usr/bin/env python
# coding: utf-8

# # Text Classification Using Word Embeddings

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from time import sleep
import os
import codecs
import sys
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression


def load_and_prune_embeddings(embedding_file, cache=True):
    if cache:
        if os.path.exists(os.path.join(os.path.dirname(embedding_file), embedding_file+'_pruned')):
            embedding_file = os.path.join(os.path.dirname(embedding_file), embedding_file+'_pruned')
    vocab = []
    E = {}
    with codecs.open(embedding_file, 'r', 'utf-8') as emb_obj:
        for line in emb_obj:
            l = line.split(' ')
            if l[0].isalpha():
                v = [float(i) for i in l[1:]] 
                E[l[0].lower()] = np.array(v)
                vocab.append(l[0].lower())

    return set(vocab), E 





def emb_vectorizer(document, vocab, embeddings, mode):
    embedding_size = len(embeddings[list(embeddings.keys())[0]])
    doc_length = len(document)
    doc_matrix = np.zeros((doc_length,embedding_size), dtype=np.float)
    for i,rec in enumerate(document):
        for line in rec:
            line_vector = []
            if type(line) == str:
                for word in line.split(' '):
                    word = str(word).lower().strip()
                    if word in vocab:
                        line_vector.append(embeddings[word])
                if len(line_vector) != 0:
                    if mode == 'max':
                        doc_matrix[i] = np.max(np.array(line_vector), axis=0)
                    elif mode == 'min':
                        doc_matrix[i] = np.min(np.array(line_vector), axis=0)
                    elif mode == 'average':
                        doc_matrix[i] = np.average(np.array(line_vector), axis=0)
                    elif mode == 'mean':
                        doc_matrix[i] = np.mean(np.array(line_vector), axis=0)
                    else:
                        doc_matrix[i] = np.sum(np.array(line_vector), axis=0)
    return doc_matrix


    
def print_scores(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Precision score: {:3f}'.format(precision_score(y_test, y_pred, average='macro') ))
    print('Recall score: {:3f}'.format(recall_score(y_test, y_pred, average='macro') ))
    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred,  average='macro')))
    print('AUC score: {:3f}'.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred)))



def main():
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    embedding_file = sys.argv[3]
    mode = sys.argv[4].strip().lower()
    cache = bool(sys.argv[5])

    vocab, embeddings = load_and_prune_embeddings(embedding_file, cache)

    #if not os.path.exists('/Users/aggarwalpiush/github_repos/offensivetextevaluation/experiments/svm_model.sav'):

    train_data = pd.read_csv(train_input, sep='\t', dtype={'tweet': object,  'id': np.int32,
                              'subtask_a': 'category'})

    X = train_data[['tweet']].values
    y = train_data['subtask_a'].values


    X_transform = emb_vectorizer(X, vocab, embeddings, mode)

    X_transform = scale(X_transform)
    le = LabelEncoder()
    y = le.fit_transform(y)

    rus = RandomOverSampler(random_state=0)
    X_resample, y_resample = rus.fit_sample(X_transform, y[:X_transform.shape[0]])

    #X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, stratify=y_resample, test_size=0.2, random_state=0)

    #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #clf = SVC(kernel='rbf', C=1)
    #clf2 = LogisticRegression(C=0.001)
    #print(cross_val_score(clf, X_resample, y_resample, cv=cv, scoring='f1_macro'))

    #print(cross_val_score(clf2, X_resample, y_resample, cv=cv, scoring='f1_macro'))

    svc = SVC().fit(X_resample, y_resample)
    #print_scores(svc, X_train, y_train, X_test, y_test)

    #svc = LogisticRegression(C=0.001).fit(X_resample, y_resample)
    #print_scores(svc, X_train, y_train, X_test, y_test)


    filename = 'svm_model_lr.sav'
    pickle.dump(svc, open(filename, 'wb'))
    
    test_data = pd.read_csv(test_input, sep='\t',
                    dtype={'tweet': object,  'id': np.int32,
                          'subtask_a': 'category'})

    # test_data = pd.read_csv(test_input, sep='\t',
    #                 dtype={'tweet': object,  'id': np.int32})



    X_sub = test_data[['tweet']].values
    X_id = test_data[['id']].values
    y_sub = test_data['subtask_a'].values

    X_sub_transform = emb_vectorizer(X_sub, vocab, embeddings, mode)

    X_sub_transform = scale(X_sub_transform)
    le = LabelEncoder()
    y_true = le.fit_transform(y_sub)

    loaded_model = pickle.load(open('svm_model_lr.sav', 'rb'))
    y_pred_test = loaded_model.predict(X_sub_transform)

    # with codecs.open('testset-taska_logreg_.tsv', 'w', 'utf-8') as hate_obj:
    #     for i in range(len(y_pred_test)):
    #         hate_obj.write(str(X_id[i])+'\t'+str(X_sub[i])+'\t'+str(y_pred_test[i])+'\n')


    print('============Result on test set=======================')
    print('F1 score: {:3f}'.format(f1_score(y_true, y_pred_test)))
    print('Precision score: {:3f}'.format(precision_score(y_true, y_pred_test, average='macro') ))
    print('Recall score: {:3f}'.format(recall_score(y_true, y_pred_test, average='macro') ))
    print('F1 score: {:3f}'.format(f1_score(y_true, y_pred_test,  average='macro')))
    print('AUC score: {:3f}'.format(roc_auc_score(y_true, y_pred_test)))
    print('Confusion Metric : %s' %(confusion_matrix(y_true, y_pred_test)))
    print('Confusion Metric : %s' %(accuracy_score(y_true, y_pred_test)))






if __name__ == '__main__':
    main()












