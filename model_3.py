import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin

class Model3(BaseEstimator, TransformerMixin):

	def __init__(self, disable=False):
		self.name = 'Stacked Model 2'
		self.disable = disable

	def fit(self, train, y_train):
		if self.disable:
			print(self.name, 'disabled')
			return
		print('Training', self.name)

		train = train.copy()
		y_train = y_train.copy()

		random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)
		model_elastic = ElasticNetCV(l1_ratio=[.1, .4, .5, .6, .7, .8, .9, .95, .99, 1], cv=5)
		model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 0.0005], cv=5, verbose=False)
		tree_model = ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=5)

		self.model = StackingRegressor(regressors=[random_forest, model_lasso, model_elastic], 
		                           meta_regressor=tree_model)
		
		self.model.fit(train, y_train)

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		return self.model.predict(test)

def predict(name, train, test, disable=False):
	if disable:
		print(name, 'disabled')
		return (np.zeros((train.shape[0],)), np.zeros((test.shape[0],)))
	print('Training', name)

	y_train = train['y'].values
	train.drop(['ID', 'y'], axis=1, inplace=True)
	test.drop(['ID'], axis=1, inplace=True)

	# # drop 30 lowest scored form XGB
	# lowest_scored_thirty = ['X344', 'X20','X117','X109','X378','X45','X362','X161','X164','X61',
 # 'X65','X380','X154', 'X300','X77', 'X114', 'X85', 'X321', 'X195','X209', 'X206', 'X283', 'X343', 'X340', 'X376',
 # 'X36', 'X375', 'X264', 'X250', 'X329']
	# train = train.drop(lowest_scored_thirty, axis=1)
	# test = test.drop(lowest_scored_thirty, axis=1)

	# # drop LassoCV eliminated features
	# lasso_eliminated_features = ['X3', 'X0', 'X314', 'X350', 'X315', 'X180', 'X27', 'X261', 
 #                             'X220', 'X321', 'X355', 'X29', 'X136']

	# to_eliminate = list(set(lasso_eliminated_features) - set(lowest_scored_thirty))

	# train = train.drop(to_eliminate, axis=1)
	# test = test.drop(to_eliminate, axis=1)

	# drop unnecessary numeric columns
	num_train = train.shape[0]
	df_all = pd.concat([train, test])
	print(df_all.shape)
	df_numeric = df_all.select_dtypes(exclude=['object'])
	df_obj = df_all.select_dtypes(include=['object']).copy()
	print(df_numeric.shape, " ", df_obj.shape)

	for col in df_numeric:
	    cardinality = len(np.unique(train[col]))
	    if cardinality == 1:
	        df_numeric = df_numeric.drop(col, axis=1)
	        
	for col in df_obj:
	    df_obj[col] = pd.factorize(df_obj[col])[0]

	df_values = pd.concat([df_numeric, df_obj], axis=1)
	print(df_values.shape)

	train = df_values.values[:num_train]
	test = df_values.values[num_train:]

	random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)
	model_elastic = ElasticNetCV(l1_ratio=[.1, .4, .5, .6, .7, .8, .9, .95, .99, 1], cv=5)
	model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 0.0005], cv=5, verbose=False)
	tree_model = ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=5)

	stregr = StackingRegressor(regressors=[random_forest, model_lasso, model_elastic], 
	                           meta_regressor=tree_model)
	
	stregr.fit(train, y_train)

	y_pred_test = stregr.predict(test)
	y_pred_train = stregr.predict(train)

	print(name, 'R2 score on train data:')
	print(r2_score(y_train, y_pred_train))

	return (y_pred_train, y_pred_test)
