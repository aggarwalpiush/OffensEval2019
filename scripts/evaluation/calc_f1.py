from __future__ import division
import codecs
import sys
import subprocess
import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def load_file(input_file):
	in_data = []
	with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			in_data.append(line)

	return in_data

def main():

	f1_scores = []
	pred_results = load_file('/tmp/aug/test_results.tsv')
	y_pred = []
	for x in pred_results:
		y_pred.append(1 if float(x.split('\t')[0]) > float(x.split('\t')[1]) else 0)

	test_results = load_file('/home/piush/github_repo/offensivetextevaluation/data/train_data/augab')
	y_test = []
	for x in test_results:
		y_test.append(0 if x.split('\t')[-1].strip('\r\n') == 'NOT' else 1)

	print(y_test[:10])
	print(y_pred[:10])

	f1_scores.append(f1_score(y_test, y_pred, average=macro))
	print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred)))


	print(f1_scores)

if __name__ == '__main__':
	main()
