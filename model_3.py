import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from model_mix import ModelMix

class Model3(BaseEstimator, TransformerMixin):

	def __init__(self, disable=False):
		self.name = 'RidgeCV Model'
		self.disable = disable

	def transform(self, data):
		# lowest_scored_thirty = []
		# lowest_scored_thirty = ['X344', 'X20','X117','X109','X378','X45','X362','X161','X164','X61',
		#  'X65','X380','X154', 'X300','X77', 'X114', 'X85', 'X321', 'X195','X209', 'X206', 'X283', 'X343', 'X340', 'X376',
		#  'X36', 'X375', 'X264', 'X250', 'X329']
		# lowest_scored_thirty = [ele for ele in lowest_scored_thirty if ele in data.columns]
		# data = data.drop(lowest_scored_thirty, axis=1)

		# lasso_eliminated_features = ['X3', 'X0', 'X314', 'X350', 'X315', 'X180', 'X27', 'X261', 
  #                            'X220', 'X321', 'X355', 'X29', 'X136']

		# to_eliminate = [ele for ele in (set(lasso_eliminated_features)) if ele in data.columns]

		# data = data.drop(to_eliminate, axis=1)

		if not hasattr(self, 'scaler'):
			self.scaler = MaxAbsScaler().fit(data)
		data = pd.DataFrame(self.scaler.transform(data))	

		# features = [14, 16, 18, 19, 20, 31, 87, 129, 146, 147, 148, 151, 185, 190, 193, 226, 240, 250, 251, 252, 265, 306, 310, 316, 349, 377, 389, 398, 410, 411]
		features = [16, 18, 19, 20, 31, 87, 129, 146, 147, 148, 151, 190, 193, 240, 250, 251, 252, 265, 306, 316, 389, 398, 410, 411]
		

		# features = [19, 20, 51, 87, 125, 127, 133, 146, 147, 149, 151, 192, 193, 216, 250, 257, 265, 306, 314, 398, 410, 411]
		# features = [19, 20, 51, 87, 125, 133, 146, 147, 149, 151, 192, 193, 216, 257, 265, 306, 314, 398, 410, 411]
# 
		# features = [125, 146, 147, 151, 192, 193, 257, 265, 306, 314, 398, 410, 411]
		data = data[features]            

		return data

	def fit(self, train, y_train): 
		if self.disable:
			print(self.name, 'disabled')
			return
		print('Training', self.name)

		train = self.transform(train.copy())
		y_train = y_train.copy()

		# random_forest = RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=3)
		# model_elastic = ElasticNetCV(l1_ratio=[.1, .4, .5, .6, .7, .8, .9, .95, .99, 1], cv=5)
		# model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0001, 0.0005], cv=5, verbose=False)
		# tree_model = ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)

		# self.model = StackingRegressor(regressors=[random_forest, model_lasso, model_elastic], 
		#                            meta_regressor=tree_model)

		# self.model = ModelMix([random_forest, model_elastic, model_lasso], [0.4, 0.3, 0.3])		
		self.model = RidgeCV(alphas = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02], cv=5)
		self.model.fit(train, y_train)
		# print(self.model.alpha_)
		# print(self.model.coef_)
		# print(len(self.model.coef_))
		# asdf = []
		# for i in range(len(train.columns)):
		# 	if self.model.coef_[i] < 0.4:
				# print("    ", train.columns.values[i], self.model.coef_[i])
		# 		asdf.append(train.columns.values[i])
		# print(asdf)

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(test)
