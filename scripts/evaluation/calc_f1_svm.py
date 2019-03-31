import pandas as pd
from sklearn.preprocessing import scale



def main():

	test_data = pd.read_csv('/Users/aggarwalpiush/github_repos/offensivetextevaluation/data/train_data/dev.tsv_preprocessed', sep='\t',
	                        dtype={'tweet': object,  'id': np.int32,
	                              'subtask_a': 'category'})


	X_sub = test_data[['tweet']].as_matrix()
	print(X_sub[:10])



	X_transform_sub = tokenize_and_transform(X_sub, X_sub.shape[0])

	X_transform_sub = scale(X_transform_sub)



	loaded_model = pickle.load(open('../train_data/svm_model.sav', 'rb'))
	y_pred = loaded_model.predict(X_transform_sub)

	out_data = np.column_stack((test_data['id'].as_matrix(), y_pred))

	out_data = [[x[0],'OFF' if x[1]==1 else 'NOT'] for x in list(out_data)]

	np.savetxt('submission.csv', out_data, fmt='%s,%s', delimiter=',')


	y_true = test_data[['subtask_a']].as_matrix()
	le = LabelEncoder()

	y_true = le.fit_transform(y_true)
	print('F1 score: {:3f}'.format(f1_score(y_true, y_pred)))