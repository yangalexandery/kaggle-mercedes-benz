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


class Model4(BaseEstimator, ClassifierMixin):

    def __init__(self, disable=False):
        self.name = 'ElasticNetCV Model'
        self.disable = disable

    def transform(self, data):
        if not hasattr(self, 'scaler'):
            self.scaler = MaxAbsScaler().fit(data)
        data = pd.DataFrame(self.scaler.transform(data))            

        # if hasattr(self, 'drop_list'):
        #     data = data.drop(self.drop_list, axis=1)
        return data

    def fit(self, train, y_train):
        if self.disable:
            print(self.name, 'disabled')
            return
        print('Training', self.name)
        train = self.transform(train.copy())
        y_train = y_train.copy()

        # self.model = make_pipeline(
            # ElasticNetCV(l1_ratio=1.0, tol=0.0001)
        # )
        # self.model = ElasticNetCV(l1_ratio=1.0, tol=0.0001)
        # self.model.fit(train, y_train)

        # self.drop_list = []
        # print(self.model.coef_)
        # print(len(self.model.coef_))
        # features = train.columns.values
        # for i in range(len(features)):
        #     if self.model.coef_[i] < 0.00001:
        #         # print(train.columns[i])
        #         self.drop_list.append(features[i])
        # train = train.drop(self.drop_list, axis=1)
        # print(train.shape)


        self.model = ElasticNetCV(l1_ratio=1.0, tol=0.0001)
        self.model.fit(train, y_train)

        # print(self.model.coef_)
        # print(len(self.model.coef_))
        # for i in range(len(train.columns)):
        #     if self.model.coef_[i] < 0.00001:
        #         print(train.columns[i])

    def predict(self, test):
        if self.disable or not self.model:
            return np.zeros((test.shape[0],))
        test = self.transform(test.copy())
        return self.model.predict(test)