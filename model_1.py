import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, ClassifierMixin

class Model1(BaseEstimator, ClassifierMixin):

	def __init__(self, disable=False):
		self.name = 'XGB Model'
		self.disable = disable

	def transform(self, data):
		n_comp = 12

		# TSVD
		tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
		tsvd_results = tsvd.fit_transform(data)

		# PCA
		pca = PCA(n_components=n_comp, random_state=420)
		pca2_results = pca.fit_transform(data)

		# ICA
		ica = FastICA(n_components=n_comp, random_state=420)
		ica2_results = ica.fit_transform(data)

		# GRP
		grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
		grp_results = grp.fit_transform(data)

		# SRP
		srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
		srp_results = srp.fit_transform(data)

		# for i in range(1, n_comp + 1):
		#     data['pca_' + str(i)] = pca2_results[:, i - 1]
		#     data['ica_' + str(i)] = ica2_results[:, i - 1]
		    # data['tsvd_' + str(i)] = tsvd_results[:, i - 1]
		    # data['grp_' + str(i)] = grp_results[:, i - 1]
		    # data['srp_' + str(i)] = srp_results[:, i - 1]

		return data

	def fit(self, train, y_train):
		if self.disable:
			print(self.name, 'disabled')
			return
		print('Training', self.name)
		train = self.transform(train.copy())
		y_train = y_train.copy()
		y_mean = np.mean(y_train)

		xgb_params = {
		    'n_trees': 520, 
		    'eta': 0.0025,
		    'max_depth': 3,
		    'subsample': 0.7,
		    'objective': 'reg:linear',
		    'eval_metric': 'rmse',
		    'base_score': y_mean, # base prediction = mean(target)
		    'silent': 1
		}
		dtrain = xgb.DMatrix(train, y_train)

		num_boost_rounds = 1250

		self.model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(xgb.DMatrix(test))
