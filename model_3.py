import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import r2_score


def predict(train, test, disable=False):
	if disable:
		print('Stacked Model 2 disabled')
		return (np.zeros((train.shape[0],)), np.zeros((test.shape[0],)))
	print('Training Stacked Model 2')

	random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)
	model_elastic = ElasticNetCV(l1_ratio=[.1, .4, .5, .6, .7, .8, .9, .95, .99, 1], cv=5)
	model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 0.0005], cv=5, verbose=False)
	tree_model = ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=5)

	stregr = StackingRegressor(regressors=[random_forest, model_lasso, model_elastic], 
	                           meta_regressor=tree_model)
	
	y_train = train['y'].values
	stregr.fit(train.drop('y', axis=1), y_train)

	y_pred_test = stregr.predict(test)
	y_pred_train = stregr.predict(train.drop('y', axis=1))

	print('Stacked Model 2 R2 score on train data:')
	print(r2_score(y_train, y_pred_train))

	return (y_pred_train, y_pred_test)
