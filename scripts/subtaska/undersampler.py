#! usr/bin/env python
# *-- coding : utf-8 --*

import codecs
import sys
import numpy as np
from collections import Counter


def main():
	input_file = sys.argv[1] 
	indata = []
	with codecs.open(input_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			indata.append(line.split('\t'))

	length_infile = len(indata)

	output_train_file = sys.argv[2]
	output_test_file = sys.argv[3]
	split_percentage = sys.argv[4]
	category_field = sys.argv[5]

	categories = []
	for rec in indata:
		categories.append(rec[category_field-1])

	c = Counter(categories)
	min_cat_value = min(c.items(), key=lambda x: x[1])[1]

	unique_categories = set(categories)

	sampled_data = []
	for each_cat in unique_categories:
		count = 0
		for rec in indata:
			if rec[category_field-1] == each_cat:
				sampled_data.append(rec)
				count += 1
				if count > min_cat_value:
					break

	train_count = np.ceil(split_percentage*length_infile)
	test_count = length_infile - train_count
	with codecs.open(output_train_file, 'w', 'utf-8') as out_train_obj:
		for i in range(train_count):
			