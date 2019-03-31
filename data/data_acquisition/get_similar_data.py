#! usr/bin/env pythin
# *-- coding : utf-8 --*

import codecs
import sys
import subprocess
import pandas as pd
import numpy as np
from scipy import spatial

def load_file(input_file):
	in_data = []
	with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			in_data.append(line)
	return in_data


def tokenize_data(infile):
	call_arc_preprocessor = 'python /Users/aggarwalpiush/github_repos/offensivetextevaluation/data/train_data_taska/preprocessor_arc.py ' + str(infile)
	subprocess.call(call_arc_preprocessor, shell = True)
	return infile + '_arc_preprocessed'


def load_embeddings(embedding_file):
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

def emb_vectorizer(document, vocab, embeddings):
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
                        doc_matrix[i] = np.sum(np.array(line_vector), axis=0)
    return doc_matrix


def main():
	relevant_file = sys.argv[1]
	irrelevant_file = sys.argv[2]
	multple = sys.argv[3]

	tokenized_rel_file = tokenize_data(relevant_file)
	tokenized_irrel_file = tokenize_data(irrelevant_file)

	vocab, embeddings = load_embeddings('/Users/aggarwalpiush/embeddings/offeneval_subtask_a.vec')

	relevant_data = pd.read_csv(tokenized_rel_file, sep='\t', dtype={'tweet': object,  'id': np.int32,
                                  'subtask_b': 'category'})
	irrelevant_data = pd.read_csv(tokenized_irrel_file, sep='\t', dtype={'tweet': object,  'id': np.int32,
                                  'subtask_b': 'category'})

	relevant_data_text = relevant_data[['tweet']].values

	irrelevant_data_text = irrelevant_data[['tweet']].values

	rel_doc_matrix = emb_vectorizer(relevant_data_text, vocab, embeddings)
	irrel_doc_matrix = emb_vectorizer(irrelevant_data_text, vocab, embeddings)

	irrelevant_similar_tweets = []
	for emb_vector in rel_doc_matrix:
		sim_scores_list = []
		for compare_vector in irrel_doc_matrix:
			sim_scores_list.append(1 - spatial.distance.cosine(emb_vector, compare_vector))


		irrelevant_similar_tweets.append(np.argsort(sim_scores_list)[:-multple])

	rel_data = []
	for i, index_vec in enumerate(irrelevant_similar_tweets):
		rel_data.append(relevant_data[i])
		for j in index_vec:
			rel_data.append(irrelevant_data[j])

	rel_data = np.array([rel_data])
	rel_data = rel_data.reshape(-1,1)
	np.savetxt(relevant_file+'_augmented', rel_data, delimiter='\t')


if __name__ == '__main__':
	main()





















