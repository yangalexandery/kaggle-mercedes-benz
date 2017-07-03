import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

class ModelMix(BaseEstimator, ClassifierMixin):

	def __init__(self, models, proportions):
		self.models = models
		self.proportions = proportions

	def fit(self, train, y_train):
		for model in self.models:
			model.fit(train.copy(), y_train.copy())

	def predict(self, test):
		y_pred = np.zeros((test.shape[0],))
		for i in range(len(self.models)):
			y_pred += models[i].predict(test.copy()) * proportions[i]
		return y_pred