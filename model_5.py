import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNetCV, LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# large standard dev in CV
class Model5(BaseEstimator, ClassifierMixin):

	def __init__(self, disable=False):
		self.name = 'DecisionTreeRegressor Model'
		self.disable = disable

	def transform(self, data):
		# remove = []
		# data = data['X0']
		# for col in data.columns.values.tolist():
		# 	if not col.find('_') == -1:
		# 	# if not col.startswith('X8_'):
		# 		remove.append(col)
		# data = data.drop(remove, axis=1)
		if not hasattr(self, 'scaler'):
			self.scaler = MaxAbsScaler().fit(data)
		data = pd.DataFrame(self.scaler.transform(data))			
		# print(data.shape)
		# features = [0, 20, 27, 33, 36, 45, 55,
		# 99, 112, 118, 130, 146,
		# 151, 157, 166, 168, 189,
		# 206, 223, 231, 232, 265,
		# 266, 269, 270, 302, 303, 305, 306,
		# 314, 320, 326, 330, 386, 393,
		# 398, 408, 413, 432, 433,
		# 452] # 0.57338
		features = [0, 20, 27, 36, 45, 55,
		99, 118, 130, 146,
		151, 157, 166, 189,
		206, 232, 265,
		270, 302, 303, 305, 306,
		314, 320, 326, 330, 386, 393,
		398, 408, 432, 433,
		452] # 0.57338
		# features = [0, 20, 27, 36, 45, 55,
		# 99, 118, 130, 146,
		# 151, 166, 189,
		# 206, 265,
		# 270, 302, 303, 305, 306,
		# 314, 320, 326, 330, 386, 393,
		# 398, 408, 433,
		# 452] # 0.57338
		# features = [0, 6, 7, 14, 18, 20, 27, 30, 32, 36, 39, 42, 50, 51, 53, 55, 61, 68, 75, 76, 87,
		# 99, 118, 127, 147, 151, 166, 170, 172, 189, 211, 216, 220, 223, 232, 235, 239,
		# 257, 265, 268, 269, 270, 274, 277, 285, 287, 290, 298, 302, 303, 305, 306, 310,
		# 311, 314, 320, 322, 326, 335, 376, 381, 389, 398, 406, 411, 414, 424, 426, 428,
		# 430, 434, 436, 437, 440, 441, 443, 446, 451, 455] #0.57125
		# features = [0, 6, 7, 14, 19, 27, 36, 42, 51, 61, 68, 74, 87, 125, 130, 147, 151, 166, 170,
		# 189, 192, 201, 216, 219, 220, 223, 250, 257, 265, 268, 270, 285, 302, 303, 305,
		# 306, 314, 326, 330, 335, 339, 381, 386, 398, 408, 411, 413, 414, 428, 433, 436,
		# 441, 453, 455, 458]
		data = data[features]
		return data

	def fit(self, train, y_train):
		if self.disable:
			print(self.name, 'disabled')
			return
		print('Training', self.name)
		train = self.transform(train.copy())
		y_train = y_train.copy()

		self.model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=4),
                           n_estimators=20,
                           bootstrap=True,
                           oob_score=True)
		# self.model = ElasticNetCV(l1_ratio=1.0, tol=0.0001)

		# self.model = make_pipeline(
		# 	# MaxAbsScaler(), 
		# 	ElasticNetCV(l1_ratio=1.0, tol=0.0001)
		# )
		self.model.fit(train, y_train)
		tot = np.zeros((train.shape[1],))
		for tree in self.model.estimators_:
			tot += tree.feature_importances_
		# print(tot)
		# asdf = []
		# for i in range(len(train.columns)):
			# if tot[i] > 0.1:
				# print("    ", train.columns.values[i], self.model.coef_[i])
				# asdf.append(train.columns.values[i])
		# print(asdf)
		# print(self.model.estimators_[0].feature_importances_)
                                       
	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(test)