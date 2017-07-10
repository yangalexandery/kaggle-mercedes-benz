import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.base import BaseEstimator, ClassifierMixin

class Model4(BaseEstimator, ClassifierMixin):

    def __init__(self, disable=False):
        self.name = 'ElasticNetCV Model'
        self.disable = disable

    def transform(self, data):
        # features = ['X314', 'X0_6', 'X261']
        # data = data[features]
        if not hasattr(self, 'scaler'):
            self.scaler = MaxAbsScaler().fit(data)
        data = pd.DataFrame(self.scaler.transform(data))
        # features = [ 20, 146, 147, 183, 193, 265, 306, 398, 411]
        # features = [0, 2, 3, 5, 15, 16, 21, 27, 28, 29, 33, 38, 40, 41, 43, 45, 47, 
        #             56, 61, 62, 63, 67, 71, 77, 82, 88, 99, 104, 107, 113, 115, 117, 119, 121, 124, 139, 
        #             143, 153, 168, 172, 178, 179, 181, 187, 189, 194, 206, 213,
        #             225, 227, 228, 236, 238, 250, 255, 262, 264, 266, 267, 270, 274, 276, 281, 288, 
        #             289, 300, 302, 303, 308, 330, 343, 387, 390, 392, 
        #             399, 409, 415, 416, 419, 420, 427, 432, 
        #             435, 439, 442, 444, 453, 454, 458, 459]
        # data = data.drop(features, axis=1)
        # features = [
        #     20, 133, 146, 147, 183,
        #     193, 199, 265, 306, 398,
        #     410, 411
        # ]
        # features = [
        #     20, 133, 146, 147, 183,
        #     193, 199, 265, 306, 398,
        #     410, 411
        # ]
        features = [18, 20, 133, 146, 147, 159, 183, 192, 193, 199, 202, 232, 249, 265, 269, 306, 398, 410, 411, 413, 445]
        data = data[features]            
        return data

    def fit(self, train, y_train):
        if self.disable:
            print(self.name, 'disabled')
            return
        print('Training', self.name)
        # self.outlier = ModelTest()
        # self.outlier.fit(train, y_train)
        train = self.transform(train.copy())
        y_train = y_train.copy()


        # train['outlier'] = m_test.predict()

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
        #     if self.model.coef_[i] > 0.4:
        #         print("    ", train.columns.values[i], self.model.coef_[i])

    def predict(self, test):
        if self.disable or not self.model:
            return np.zeros((test.shape[0],))
        test = self.transform(test.copy())
        return self.model.predict(test)