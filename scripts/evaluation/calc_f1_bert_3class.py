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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_file(input_file):
	in_data = []
	with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			in_data.append(line)

	return in_data

def main():
	bert_file = sys.argv[1]
	ref_file = sys.argv[2]

	f1_scores = []
	pred_results = load_file(bert_file)
	y_pred = []
	for x in pred_results:
		if float(x.split('\t')[0]) > float(x.split('\t')[1]) and float(x.split('\t')[0]) > float(x.split('\t')[2]):
			y_pred.append(0)
		elif float(x.split('\t')[1]) > float(x.split('\t')[0]) and float(x.split('\t')[1]) > float(x.split('\t')[2]):
			y_pred.append(1)
		else:
			y_pred.append(2)

	test_results = load_file(ref_file)
	y_test = []
	for i,x in enumerate(test_results):
		# if i == 0:
		# 	continue
		if x.split('\t')[-1].strip('\r\n') == 'IND':
			y_test.append(0)
		elif x.split('\t')[-1].strip('\r\n') == 'GRP':
			y_test.append(1)
		else:
			y_test.append(2)

	print(y_test[:10])
	print(y_pred[:10])

	f1_scores.append(f1_score(y_test, y_pred, average='macro'))
	print('Prediction Accuracy: {:3f}'.format(accuracy_score(y_test, y_pred)))
	print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred)))


	print(f1_scores)

if __name__ == '__main__':
	main()