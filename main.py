# Author: Alexander Yang

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


import dimred
import LinearModel

train = pd.read_csv('./data/mercedes_benz_train.csv')
test = pd.read_csv('./data/mercedes_benz_test.csv')


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))



target = train['y']
train = train.drop(['ID', 'y'], axis=1)
test_ID = test['ID']
test = test.drop(['ID'], axis=1)

n_components = 10
pca = PCA(n_components=n_components)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

ica = FastICA(n_components=n_components)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

# tSVD
tsvd = TruncatedSVD(n_components=n_components)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

for i in range(n_components):
	train['pca' + str(i)] = pca_results_train[:,i]
	test['pca' + str(i)] = pca_results_test[:,i]
	train['ica' + str(i)] = ica_results_train[:,i]
	test['ica' + str(i)] = ica_results_test[:,i]
	train['tsvd' + str(i)] = tsvd_results_train[:,i]
	test['tsvd' + str(i)] = tsvd_results_test[:,i]

y_mean = np.mean(target)

# code from Fred Navruzov
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1,
    'gamma': 100
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train, target)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1000, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

print(r2_score(dtrain.get_label(), model.predict(dtrain)))

y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test_ID.astype(np.int32), 'y': y_pred})
output.to_csv('data/xgboost-depth4-pca-ica-6.csv'.format(xgb_params['max_depth']), index=False)