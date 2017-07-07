import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.linear_model import LassoLarsCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.metrics import r2_score


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


class Model2(BaseEstimator, TransformerMixin):

	def __init__(self, disable=False):
		self.name = 'Stacked Model 1'
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
		train = self.transform(train.copy())
		y_train = y_train.copy()
		print('Training', self.name)

		self.model = make_pipeline(
		    # StackingEstimator(estimator=RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)),
		    # StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
		    StackingEstimator(estimator=ElasticNetCV(l1_ratio=1.0, tol=0.0001)),
		    # MaxAbsScaler(),
		    # StackingEstimator(estimator=LassoLarsCV(normalize=True, verbose=False)),
		    # LassoLarsCV()
		    RandomForestRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=3)
			# ExtraTreesRegressor(n_estimators=20, oob_score=True, bootstrap=True, max_depth=4)
		)

		self.model.fit(train, y_train)

	def predict(self, test):
		if self.disable or not self.model:
			return np.zeros((test.shape[0],))
		test = self.transform(test.copy())
		return self.model.predict(test)

