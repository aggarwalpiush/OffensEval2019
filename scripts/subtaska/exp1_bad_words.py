#! usr/bin/env python
# * -- coding : utf-8 --*

import codecs
import sys
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer


def main():
	ctl_path = sys.argv[1]
	ctl_path_pos = sys.argv[2]
	input_path = sys.argv[3]
	with_stem = sys.argv[4]
	with_tokenizer = sys.argv[5]
	bad_words = []
	good_words = []
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	ps = PorterStemmer()
	tknzr = TweetTokenizer()
	with codecs.open(ctl_path, 'r', 'utf-8') as ctl_obj: 
		for line in ctl_obj:
			if with_stem.strip().lower() == 'true':
				bad_words.append(ps.stem(str(line).strip().lower()))
			else:
				bad_words.append(str(line).strip().lower())
	bad_words = set(bad_words)

	with codecs.open(ctl_path_pos, 'r', 'utf-8') as ctl_obj: 
		for line in ctl_obj:
			if with_stem.strip().lower() == 'true':
				good_words.append(ps.stem(str(line).strip().lower()))
			else:
				good_words.append(str(line).strip().lower())
	good_words = set(good_words)

	with codecs.open(input_path+'_pruned', 'w', 'utf-8') as out_obj:
		with codecs.open(input_path, 'r', 'utf-8') as in_obj:
			for off_line in in_obj:
				tweet = str(off_line.split('\t')[1]).strip().lower()
				label = off_line.split('\t')[2].strip()
				if with_tokenizer.strip().lower() == 'true':
					tweet_words = tknzr.tokenize(tweet)
				else:
					tweet_words = tweet.split(' ')
				word_len = 0
				for word in tweet_words:
					if with_stem.strip().lower() == 'true':
						word = ps.stem(word)
					word_len += 1
					if word in bad_words:
						if label == 'OFF':
							tp += 1
							out_obj.write(off_line)
						elif label == 'NOT':
							fn += 1
						break
					elif word_len == len(tweet_words):
						word_len2 = 0
						for word in tweet_words:
							if with_stem.strip().lower() == 'true':
								word = ps.stem(word)
							word_len2 += 1
							if word in good_words:
								if label == 'NOT':
									tn += 1
									out_obj.write(off_line)
								elif label == 'OFF':
									fp += 1
								break
							elif word_len2 == len(tweet_words):
								if label == 'OFF':
									fp += 1
								elif label == 'NOT':
									tn += 1
									out_obj.write(off_line)
	prec = 	tp/(tp+fp)
	rec  = 	tp/(tp+fn)
	f1 = 2 * (prec * rec) / (prec + rec)

	print('%s\t%s\n%s\t%s' %(tp, fp, fn, tn))
	print('Precision : %.3f' %prec)
	print('Recall : %.3f' %rec)
	print('F1 : %.3f' %f1)
	print('Total count in file : %s' %( tp + fp + fn + tn))


if __name__ == '__main__':
	main()
