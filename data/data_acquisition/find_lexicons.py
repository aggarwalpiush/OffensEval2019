# !usr/bin/env python
# *-- codng : utf-8 --*


import pandas as pd
import codecs
import sys
from collections import Counter
import string
import re
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import combinations 
import subprocess
import shlex
from time import sleep
import inflect
from wordsegment import load, segment
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from time import sleep
load()
p = inflect.engine()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def load_file(input_file):
	in_data = []
	with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			in_data.append(line)
	return in_data

def find_combination(letters): 
    for n in range(3, len(letters)+1): 
    	yield map(''.join, combinations(letters, n))

def find_root_pattern( word_list, freq_num):
	frequent_pattern = {}

	for word in word_list:
		for w in find_combination(word):
			if w in word:
				count = 0
				for rest_word in word_list:
					if w in rest_word:
						count += 1
						#print(w+'\t'+rest_word+'\t'+str(count))
						#sleep(1)

				if count < freq_num:
					frequent_pattern[w] = count
						
	return frequent_pattern





def generate_unigrams(document):
	unigrams = [] 
	for record in document:
		for word in record.strip('\r\n').split(' '):
			unigrams.append(processed_word)
	return unigrams


def calculate_tf_idf(document):
	vectorizer = TfidfVectorizer(ngram_range=(2, 3))
	matrix = vectorizer.fit_transform(document).todense()
	matrix = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
	tfidf_words_dict = dict(matrix.sum(axis=0).sort_values(ascending=False))
	return tfidf_words_dict

def create_lexicon_list(input_dict, compare_dict, hp_alone, hp_co):
	cat_list = [] 
	for k,v in input_dict.items():
		#if k not in compare_dict.keys():
		if abs(v) >= abs(hp_alone):
			cat_list.append('%s_%s' %(k,v))
		elif k in compare_dict.keys():
			if abs(compare_dict[k] +  float(hp_co)) <=  abs(input_dict[k]):
				cat_list.append('%s_%s' %(k,v))
	return cat_list


def save_file(input_data, filename):
	with codecs.open(filename, 'w', 'utf-8') as out_obj:
		for words in sorted(input_data):
			out_obj.write(words)


def main():
	cat1_file = sys.argv[1]
	cat2_file = sys.argv[2]

	hyperparam_alone_freq_cat1 = float(sys.argv[3])
	hyperparam_co_greater_value_cat1 = float(sys.argv[4])
	hyperparam_alone_freq_cat2 = float(sys.argv[5])
	hyperparam_co_greater_value_cat2 = float(sys.argv[6])

	cat1_records = load_file(cat1_file)
	cat2_records = load_file(cat2_file)

	# cat1_unigrams = generate_unigrams(cat1_records)
	# cat2_unigrams = generate_unigrams(cat2_records)

	# print(cat1_unigrams[:10])
	# print(cat2_unigrams[:10])

	# print(len(cat1_unigrams))
	# print(len(cat2_unigrams))

	# cat1_counter = Counter(cat1_unigrams)
	# cat2_counter = Counter(cat2_unigrams)

	cat1_tfidf = calculate_tf_idf(cat1_records)
	cat2_tfidf = calculate_tf_idf(cat2_records)

	print(len(list(cat1_tfidf.keys())))
	print(len(list(cat2_tfidf.keys())))


	cat1_list = create_lexicon_list(cat1_tfidf, cat2_tfidf, hyperparam_alone_freq_cat1, hyperparam_co_greater_value_cat1)
	cat2_list = create_lexicon_list(cat2_tfidf, cat1_tfidf, hyperparam_alone_freq_cat2, hyperparam_co_greater_value_cat2)

	print(len(cat1_list))
	print(len(cat2_list))

	print(cat1_list)
	print(cat2_list)

	# print([x.split('_')[0] for x in cat1_list])
	# print([x.split('_')[0] for x in cat2_list])
	#print(cat2_list[:40])



	# for key,value in cat1_counter.items():
	# 	if value >= int(hyperparam_freq):
	# 		continue
	# 	if key not in cat2_counter.keys():
	# 		if value > 1:
	# 			cat1_list.append(key+'_'+str(value))
	# 	elif key in cat2_counter.keys():
	# 		if value * float(hyperparam_more_coeff) > cat2_counter[key]:
	# 			if value > 1:
	# 				cat1_list.append(key+'_'+str(value))

	# for key,value in cat2_counter.items():
	# 	if value >= int(hyperparam_freq):
	# 		continue
	# 	if key not in cat1_counter.keys():
	# 		if value > 1:
	# 			cat2_list.append(key+'_'+str(value))
	# 	elif key in cat1_counter.keys():
	# 		if value * float(hyperparam_more_coeff) > cat1_counter[key]:
	# 			if value > 1:
	# 				cat2_list.append(key+'_'+str(value))

	# print(cat1_list[:10])
	# print(cat2_list[:10])
	# print(len(cat1_list))
	# print(len(cat2_list))

	# cat1_list_wo_freq = [w.split('_')[0] for w in cat1_list]
	# cat2_list_wo_freq = [w.split('_')[0] for w in cat2_list]


	# cat1_list_wo_freq_bad = []
	# for w in cat1_list:
	# 	if int(w.split('_')[1]) > 3:
	# 		cat1_list_wo_freq_bad.append(w.split('_')[0])
	# cat1_freq_word = find_root_pattern(cat1_list_wo_freq, 3)
	# #cat2_freq_word = find_root_pattern(cat2_list_wo_freq, 5)

	# cat1_freq_word_temp = list(cat1_freq_word.keys())

	# for key in cat1_freq_word.keys():
	# 	for word_key in cat1_freq_word.keys():
	# 		if key in word_key and word_key != key:
	# 			if key in cat1_freq_word_temp:
	# 				cat1_freq_word_temp.remove(key)
	# 			break

			

	# print(cat1_freq_word_temp[:10])
	# #print(list(cat2_freq_word.keys())[:10])
	# save_file([str(x)+'_'for x in cat1_freq_word_temp], cat1_file+'_pattern')
	# save_file(cat1_list, cat1_file+'_out')
	# save_file(cat2_list, cat2_file+'_out')



if __name__ == '__main__':
	main()







