#! usr/bin/env python
# -*- coding : utf-8 -*-

'''
This python file is designed for 3 classes classification task with skewed distribution of records
having minority class has only around 20% of majority class records
input file should be of tsv format and contain three fields as follows
1 -> record_id
2 -> text
3 -> annotation
with header as 
id\ttext\annotation

This script will assign minority class to one of the two other class label and make two classifier
record then in test set if record in both classifier belong to same class, then class is assigned to 
that record else minority class is assigned

Many different types of classifier combination approaches will be considered.  
'''


import codecs
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessor_arc import Arc_preprocessor
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

EMBEDDING_FILE = '/Users/aggarwalpiush/embeddings/offeneval_subtask_a.vec'
SAMPLE_SIZE = 16000


class MeanEmbeddingTransformer(TransformerMixin):
    
    def __init__(self):
        self._vocab, self._E = self._load_words()
        
    
    def _load_words(self):
        E = {}
        vocab = []

        with open(EMBEDDING_FILE, 'r', encoding="utf8") as file:
            for i, line in enumerate(file):
                l = line.split(' ')
                if l[0].isalpha():
                    v = [float(i) for i in l[1:]]
                    E[l[0]] = np.array(v)
                    vocab.append(l[0])
        return np.array(vocab), E            

    
    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None
    
    def _doc_mean(self, doc):
        word_array = []
        for w in doc:
            if  w.lower().strip() in self._E:
                word_array.append(self._E[w.lower().strip()])
            else:
                word_array.append(np.zeros([len(v) for v in self._E.values()][0]))

        return np.mean(np.array(word_array), axis=0)
                
        #return np.mean(np.array([self._E[w.lower().strip()] for w in doc if w.lower().strip() in self._E else np.zeros([len(v) for v in self._E.values()][0])]), axis=0)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def load_data(input_file):
	train_data = pd.read_csv(input_file, sep='\t',
                        dtype={'text': object,  'id': np.int32,
                              'annotation': 'category'})
	return train_data

def preprocess_transform(X, arc_tokenizer=True):
	text = X[:,0]
	text = [x.lower() for x in text]
	if arc_tokenizer:
		arc_obj = Arc_preprocessor()
		tok_text = [arc_obj.tokenizeRawTweetText(doc) for doc in text[:SAMPLE_SIZE]]
	else:
		tok_text = [word_tokenize(doc) for doc in text[:SAMPLE_SIZE]]
	met = MeanEmbeddingTransformer()
	X_transform = met.fit_transform(tok_text)
	return X_transform


def make_final_y_pred(y_pred_cl1, y_pred_cl2, minority_label):
	final_y_pred = []
	for i in range(len(y_pred_cl1)):
		if y_pred_cl1[i] == y_pred_cl2[i]:
			final_y_pred.append(y_pred_cl1[i])
		else:
			final_y_pred.append(minority_label)
	return  final_y_pred

def compare_use_classifier(X,y1,y2,X_test, y_test, minority_label):
	# best_f1_score = 0
	# rc = RidgeClassifier()
	#X_train, X_dev, y_train, y_dev = train_test_split(X,
     #                                                y, stratify=y, random_state=0, test_size= 0.2)
	#clf1 = SVC(kernel='rbf', C=1)
	#scores = cross_val_score(clf1, X,y, cv=10, scoring='f1_macro')
	rus = RandomUnderSampler(random_state=0)
	X_resample1, y_resample1 = rus.fit_sample(X, y1)
	X_resample2, y_resample2 = rus.fit_sample(X, y2)
	param_grid1 = {'C': [0.1, 1, 10], 'kernel' : ['rbf'], 'gamma' : [ 'scale']}
	param_grid2 = {'activation': ['relu', 'logistic', 'tanh'],
              'alpha': [0.0001, 0.001, 0.01],
              'learning_rate': ['constant', 'invscaling', 'adaptive'], 'tol': [0.01, 0.1]}
	param_grid4 = {'activation': ['logistic'],'alpha': [0.001],'learning_rate': ['adaptive'], 'tol': [ 0.1]}
	param_grid3 = {'n_estimators':[100], 'max_depth':[2]}
	gs1 = GridSearchCV(MLPClassifier(), 
             param_grid=param_grid4, scoring="f1_macro", cv=5)
	gs2 = GridSearchCV(MLPClassifier(), 
             param_grid=param_grid4, scoring="f1_macro", cv=5)
#	gs1 = GridSearchCV(RandomForestClassifier(), 
 #             param_grid=param_grid3, scoring="f1_macro", cv=5)
	# gs2 = GridSearchCV(RandomForestClassifier(), 
 #             param_grid=param_grid3, scoring="f1_macro", cv=5)
	gs1 = gs1.fit(X_resample2, y_resample2)
	gs2 = gs2.fit(X_resample1, y_resample1)
	print(gs1.best_params_)
	print(gs2.best_params_)
	y_pred_final = make_final_y_pred(gs1.predict(X_test), gs2.predict(X_test), minority_label)
	f1_value = f1_score(y_test, y_pred_final, average='macro')
	print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred_final)))
	print('Prediction Accuracy: {:3f}'.format(accuracy_score(y_test, y_pred_final)))

	return f1_value

def make_submission(X,y1,y2,X_test,minority_label):
	rus = RandomUnderSampler(random_state=0)
	X_resample1, y_resample1 = rus.fit_sample(X, y1)
	X_resample2, y_resample2 = rus.fit_sample(X, y2)
	param_grid1 = {'C': [1], 'gamma': ['scale'], 'kernel': ['rbf']}
	param_grid4 = {'activation': ['logistic'],'alpha': [0.001],'learning_rate': ['adaptive'], 'tol': [ 0.1]}
	gs1 = GridSearchCV(MLPClassifier(), 
             param_grid=param_grid4, scoring="f1_macro", cv=5)
	gs2 = GridSearchCV(SVC(), 
             param_grid=param_grid1, scoring="f1_macro", cv=5)
	gs1 = gs1.fit(X_resample1, y_resample1)
	gs2 = gs2.fit(X_resample2, y_resample2)
	print(gs1.best_params_)
	print(gs2.best_params_)
	y_pred_final = make_final_y_pred(gs1.predict(X_test), gs2.predict(X_test), minority_label)

	return y_pred_final




def main():
	input_file = sys.argv[1]
	test_file = sys.argv[2]
	cascade = sys.argv[3]
	train_data = load_data(input_file)
	X = train_data[['text']].values
	X_transform_unstd = preprocess_transform(X)
	scaler = StandardScaler()
	X_transform = scaler.fit_transform(X_transform_unstd)
	y = train_data['annotation'].values
	print(y[:10])
	le = LabelEncoder()
	y_le = le.fit_transform(y)
	print(y_le[:10])
	if test_file == '':
		X_train, X_test, y_train, y_test = train_test_split(X_transform,
                                                    y_le, stratify=y_le, random_state=0, test_size= 0.2)
	else:
		test_data = load_data(test_file)
		X_test = test_data[['text']].values
		X_test_transform_unstd = preprocess_transform(X_test)
		X_test_transform = scaler.fit_transform(X_test_transform_unstd)
		X_train, X_test, y_train = X_transform,  X_test_transform, y_le


		if cascade.lower().strip() == 'false':
			rus = RandomUnderSampler(random_state=0)
			X_resample1, y_resample1 = rus.fit_sample(X_train, y_train)
			param_grid = {'activation': ['logistic'],
	              'alpha': [0.001],
	              'learning_rate': ['adaptive'], 'tol': [0.01]}
			gs = GridSearchCV(MLPClassifier(), 
	             param_grid=param_grid, scoring="f1_macro", cv=5)
			print(len(X_train))
			print(len(y_train))
			print(len(X_test))
			gs = gs.fit(X_resample1, y_resample1)
			test_id = test_data[['id']].values
			pred = ['IND' if x==1  else ('GRP' if x == 0 else 'OTH')  for x in gs.predict(X_test)]
			pred = np.array([pred])
			pred = pred.reshape(-1,1)
			out_data = np.append(test_id, pred, axis=1)
			np.savetxt('submission_mlp.csv', out_data, fmt='%s,%s', delimiter=',')
			sys.exit(0)

	print('something_wrong')


	ct = Counter(y_train)
	minority_label = min(ct, key=ct.get)
	class_labels = np.unique(y_train)
	print(class_labels)
	majority_labels = [x for x in class_labels if x != minority_label]
	print(majority_labels)
	majority_class_1 = majority_labels[0]
	majority_class_2 = majority_labels[1]
	print( majority_class_1)
	print(majority_class_2)




	# prepare data for classifier 1,2
	y_train_le_cl1 = []
	y_train_le_cl2 = []
	for i,lab in enumerate(y_train):
		if lab == minority_label:
			y_train_le_cl1.append(majority_class_1)
			y_train_le_cl2.append(majority_class_2)
		else:
			y_train_le_cl1.append(lab)
			y_train_le_cl2.append(lab)
	print(Counter(y_train))
	print(Counter(y_train_le_cl1))
	print(Counter(y_train_le_cl2))

	if test_file == '':
		f1_score = compare_use_classifier(X_train,y_train_le_cl1,y_train_le_cl2,X_test, y_test,minority_label)
		print('F1 score: {:3f}'.format(f1_score))
	else:
		print(len(X_train))
		print(len(y_train_le_cl1))
		print(len(y_train_le_cl2))
		print(len(X_test))
		print(minority_label)
		y_pred = make_submission(X_train,y_train_le_cl1,y_train_le_cl2,X_test,minority_label)

	test_id = test_data[['id']].values
	pred = ['IND' if x==1  else ('GRP' if x == 0 else 'OTH')  for x in y_pred]
	pred = np.array([pred])
	pred = pred.reshape(-1,1)
	out_data = np.append(test_id, pred, axis=1)
	np.savetxt('submission_mlp_svc.csv', out_data, fmt='%s,%s', delimiter=',')





if __name__ == '__main__':
	main()








	

