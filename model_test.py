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
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin


class ModelTest(BaseEstimator, ClassifierMixin):

	def __init__(self, disable=False):
		self.name = 'ElasticNetCV Model'
		self.disable = disable

	def transform(self, data):
		if not hasattr(self, 'scaler'):
			self.scaler = MaxAbsScaler().fit(data)
		data = pd.DataFrame(self.scaler.transform(data))			

		return data

	def fit(self, train, y_train):
		if self.disable:
			print(self.name, 'disabled')
			return
		print('Training', self.name)
		train = self.transform(train.copy())
		y_train = y_train.copy()

		self.model = make_pipeline(
			# MaxAbsScaler(), 
			ElasticNetCV(l1_ratio=1.0, tol=0.0001)
		)
		self.model.fit(train, y_train)

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(test)