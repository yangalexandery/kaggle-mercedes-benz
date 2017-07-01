import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

def predict(train, test, disable=False):
	if disable:
		print('XGB Model disabled')
		return (np.zeros((train.shape[0],)), np.zeros((test.shape[0],)))
	print('Training XGB Model')
	n_comp = 12

	# tSVD
	tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
	tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
	tsvd_results_test = tsvd.transform(test)

	# PCA
	pca = PCA(n_components=n_comp, random_state=420)
	pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
	pca2_results_test = pca.transform(test)

	# ICA
	ica = FastICA(n_components=n_comp, random_state=420)
	ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
	ica2_results_test = ica.transform(test)

	# GRP
	grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
	grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
	grp_results_test = grp.transform(test)

	# SRP
	srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
	srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
	srp_results_test = srp.transform(test)

	for i in range(1, n_comp + 1):
	    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
	    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

	    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
	    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

	    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
	    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

	    train['grp_' + str(i)] = grp_results_train[:, i - 1]
	    test['grp_' + str(i)] = grp_results_test[:, i - 1]

	    train['srp_' + str(i)] = srp_results_train[:, i - 1]
	    test['srp_' + str(i)] = srp_results_test[:, i - 1]

	y_train = train['y'].values
	y_mean = np.mean(y_train)
	id_test = test['ID'].values

	xgb_params = {
	    'n_trees': 520, 
	    'eta': 0.0045,
	    'max_depth': 4,
	    'subsample': 0.93,
	    'objective': 'reg:linear',
	    'eval_metric': 'rmse',
	    'base_score': y_mean, # base prediction = mean(target)
	    'silent': 1
	}
	dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
	dtest = xgb.DMatrix(test)

	num_boost_rounds = 1250

	model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
	y_pred_test = model.predict(dtest)
	y_pred_train = model.predict(dtrain)

	print('XGB R2 score on train data:')
	print(r2_score(y_train, y_pred_train))

	return (y_pred_train, y_pred_test)