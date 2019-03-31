# !usr/bin/env python
# *-- codng : utf-8 --*


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
load()
p = inflect.engine()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def main():
	input_file = sys.argv[1]
	text_column = int(sys.argv[2])
	table = str.maketrans('', '', string.punctuation)
	# ps = PorterStemmer()
	emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
	
	with codecs.open(input_file+'_preprocessed', 'w', 'utf-8') as processed_obj:
		with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
			for line_rec in in_file_obj:
				line = line_rec.split('\t')[text_column-1]
				line = segment(line)
				line = [w for w in line if not w in stop_words] 
				line = [w for w in line if len(w) > 2]
				line = [w for w in line if not w[0].isdigit()]
				line = [w for w in line if w != 'user']
				line = [emoji_pattern.sub(r'',w.translate(table)) for w in line]
				
				new_line = []
				for word in line:
					if  type(p.singular_noun(word)) == bool:
						new_line.append(lemmatizer.lemmatize(word, 'v'))			
					else:
						new_line.append(p.singular_noun(word))
				processed_obj.write('\t'.join(line_rec.split('\t')[:text_column-1])+'\t'+' '.join(new_line)+'\t'+'\t'.join(line_rec.split('\t')[text_column:]))



if __name__ == '__main__':
	main()
