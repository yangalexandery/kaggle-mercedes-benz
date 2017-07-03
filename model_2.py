import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
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

	def fit(self, train, y_train):
		if self.disable:
			print(self.name, 'disabled')
			return
		train = train.copy()
		y_train = y_train.copy()
		print('Training', self.name)

		self.model = make_pipeline(
		    StackingEstimator(estimator=LassoLarsCV(normalize=True, verbose=False)),
		    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=4, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
		    LassoLarsCV(verbose=False)
		)

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

	stacked_pipeline = make_pipeline(
	    StackingEstimator(estimator=LassoLarsCV(normalize=True, verbose=False)),
	    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=4, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
	    LassoLarsCV(verbose=False)
	)

	y_train = train['y'].values
	stacked_pipeline.fit(train.drop('y', axis=1), y_train)
	y_pred_test = stacked_pipeline.predict(test)
	y_pred_train = stacked_pipeline.predict(train.drop('y', axis=1))

	print(name, 'R2 score on train data:')
	print(r2_score(y_train, y_pred_train))

	return (y_pred_train, y_pred_test)
