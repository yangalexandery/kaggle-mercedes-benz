import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.preprocessing import MaxAbsScaler
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
		if not hasattr(self, 'scaler'):
			self.scaler = MaxAbsScaler().fit(data)
		data = pd.DataFrame(self.scaler.transform(data))			

		features = [426, 398, 223, 0, 87, 305, 193, 408, 436, 18, 7, 146, 265, 306, 189, 36, 314, 92, 27]
		# features = [398, 223, 0, 87, 305, 193, 408, 146, 265, 306, 189, 36, 314, 27]
		data = data[features]            
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
		    'n_trees': 500, 
		    'eta': 0.0025,
		    'max_depth': 4,
		    'subsample': 0.6,
		    'objective': 'reg:linear',
		    'eval_metric': 'rmse',
		    'base_score': y_mean, # base prediction = mean(target)
		    'silent': 1,
		    # 'alpha': 0.001,
		    # 'lambda': 2,
		    'gamma': 40
		    # 'alpha': 0.1,
		    # 'lambda': 2,
		    # 'gamma': 50
		}
		dtrain = xgb.DMatrix(train, y_train)

		num_boost_rounds = 1250

		self.model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
		vals = self.model.get_score()
		# print(vals)
		# print(self.model.get_score())
		# for key in vals:
			# if vals[key] > 70:
			# print(key, vals[key])

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(xgb.DMatrix(test))
