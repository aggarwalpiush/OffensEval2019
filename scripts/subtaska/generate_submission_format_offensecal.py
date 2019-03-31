#!usr/bin/env python
# *-- coding : utf-8 --*

import codecs
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef


def load_file(input_file):
	in_data = []
	with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			in_data.append(line)

	return in_data

def submission_format(method,test_id,pred):
	test_id = np.array([test_id])
	test_id = test_id.reshape(-1,1)
	pred = ['OFF' if x==1 else 'NOT' for x in pred]
	pred = np.array([pred])
	pred = pred.reshape(-1,1)
	out_data = np.append(test_id, pred, axis=1)
	np.savetxt('submission_'+method+'.csv', out_data, fmt='%s,%s', delimiter=',')


def review_format(method,tweet,pred):
	tweet = np.array([tweet])
	tweet = tweet.reshape(-1,1)
	pred = ['OFF' if x==1 else 'NOT' for x in pred]
	pred = np.array([pred])
	pred = pred.reshape(-1,1)
	out_data = np.append(tweet, pred, axis=1)
	np.savetxt('review_'+method+'.csv', out_data, fmt='%s,%s', delimiter='\t')

def find_correlation(y1,y2):
	y1 = [1 if x==1 else -1 for x in y1]
	y2 = [1 if x==1 else -1 for x in y2]
	return matthews_corrcoef(y1,y2) 




def main():
	pred_results_bert = load_file('testset-taska_bert_.tsv')
	y_pred_bert = []
	for x in pred_results_bert:
		y_pred_bert.append(1 if float(x.split('\t')[0]) > float(x.split('\t')[1]) else 0)


	pred_results_svm = load_file('testset-taska_svm_.tsv')
	y_pred_svm = []
	for x in pred_results_svm:
		y_pred_svm.append(int(x.split('\t')[2]))


	pred_results_lr = load_file('testset-taska_logreg_.tsv')
	y_pred_lr = []
	for x in pred_results_lr:
		y_pred_lr.append(int(x.split('\t')[2]))

	pred_results_ensemble = np.sum([y_pred_bert, y_pred_svm, y_pred_lr], axis=0)
	y_pred_ensemble = []
	for x in pred_results_ensemble:
		if x == 3:
			y_pred_ensemble.append(1)
		if x == 2:
			y_pred_ensemble.append(1)
		if x == 1:
			y_pred_ensemble.append(0)
		if x == 0:
			y_pred_ensemble.append(0)


	test_file = load_file('/Users/aggarwalpiush/github_repos/offensivetextevaluation/data/test_data/testset-taska.tsv')
	test_id = []
	for i,x in enumerate(test_file):
		if i == 0:
			continue
		test_id.append(int(x.split('\t')[0]))
	test_tweet = []
	for i,x in enumerate(test_file):
		if i == 0:
			continue
		test_tweet.append(x.split('\t')[1].strip('\r\n'))


	print('Bert-SVM: %s' %find_correlation(y_pred_bert,y_pred_svm))
	print('Bert-LR: %s' %find_correlation(y_pred_bert,y_pred_lr))
	print('Bert-ensemble: %s' %find_correlation(y_pred_bert,y_pred_ensemble))
	print('SVM-LR: %s' %find_correlation(y_pred_svm,y_pred_lr))
	print('SVM-ensemble: %s' %find_correlation(y_pred_svm,y_pred_ensemble))
	print('LR-ensemble: %s' %find_correlation(y_pred_lr,y_pred_ensemble))




	# submission_format('bert',test_id, y_pred_bert)
	# submission_format('svm',test_id, y_pred_svm)
	# submission_format('lr',test_id, y_pred_lr)
	# submission_format('ensemble',test_id, y_pred_ensemble)

	# review_format('bert',test_tweet, y_pred_bert)
	# review_format('svm',test_tweet, y_pred_svm)
	# review_format('lr',test_tweet, y_pred_lr)
	# review_format('ensemble',test_tweet, y_pred_ensemble)

if __name__ == '__main__':
	main()






