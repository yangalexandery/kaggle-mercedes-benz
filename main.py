import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# import model_1, model_2, model_3
from model_1 import Model1
from model_2 import Model2
from model_3 import Model3
from model_4 import Model4
from model_mix import ModelMix
from model_test import ModelTest


train = pd.read_csv('./data/mercedes_benz_train.csv').iloc[:-32].sample(frac=1).reset_index(drop=True)
test = pd.read_csv('./data/mercedes_benz_test.csv')

# train = train[:-32]

def transformX0(cat):
    groups = [
        ['bc', 'az'],
        ['ac','am','l','b','aq','u','ad','e','al','s',
            'n','y','t','ai','k','f','z','o','ba','m','q'],
        ['d','ay','h','aj','v','ao','aw'],
        ['c','ax','x','j','w','i','ak','g','at','ab',
            'af','r','as','a','ap','au','aa']
    ]
    for i in range(len(groups)):
        if cat in groups[i]:
            return str(i * 2)
    # print(cat)
    return '3'

train_untouched = train.copy()
test_untouched = test.copy()

train['X0'] = train['X0'].transform(transformX0)
test['X0'] = test['X0'].transform(transformX0)
# untouched_y = train['y']
# train['y'] = np.log(train['y'])

hold_y = train['y']
df_all = pd.concat([train, test]).drop('y', axis=1)
df_numeric = df_all.select_dtypes(exclude=['object']).copy()
df_obj = df_all.select_dtypes(include=['object']).copy()
print(df_obj.shape)
print(df_numeric.shape)

# drop the numeric features where the column contains only one unique value
for col in df_numeric:
    cardinality = len(np.unique(train[col]))
    if cardinality == 1:
        df_numeric = df_numeric.drop(col, axis=1)

cols = df_numeric.columns.tolist()
remove = []
for i in range(len(cols)-1):
    v = df_all[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,df_all[cols[j]].values):
            remove.append(cols[j])
            # print(' Column %s is identical to %s. Removing %s' % (str(cols[i]), str(cols[j]), str(cols[j])))
df_numeric.drop(remove, axis=1, inplace=True)

cols = df_numeric.columns.tolist()
remove = []
print('\n Converting categorical features:')
categorical_feats = df_all.dtypes[df_all.dtypes == 'object'].index
for i, col_name in enumerate(categorical_feats):
    # print(' Converting %s' % col_name)
    temp_df = pd.get_dummies(df_all[col_name])
    new_features = temp_df.columns.tolist()
    new_features = [col_name + '_' + w for w in new_features]
    temp_df.columns = new_features
    for j in range(len(new_features)):
        v = temp_df[new_features[j]].values
        for k in range(len(cols)):
            if np.array_equal(v, df_all[cols[k]].values):
                remove.append(cols[k])
                # print(' Column %s is identical to %s. Removing %s' % (str(new_features[j]), str(cols[k]), str(cols[k])))
df_numeric.drop(remove, axis=1, inplace=True)

cols = df_obj.columns.values.tolist()
for col in cols:
    print(col)
    print(pd.get_dummies(df_obj[col], prefix=col).shape)
    # print(pd.get_dummies(df_obj[col], prefix=col).head())
    df_obj = pd.concat([df_obj, pd.get_dummies(df_obj[col], prefix=col)], axis=1)
    # df_obj[col] = pd.factorize(df_obj[col])[0]
    df_obj = df_obj.drop(col, axis=1)
print(df_obj.head())

df_values = pd.concat([df_numeric, df_obj], axis=1)
print(df_values.shape)
train = df_values.iloc[:train.shape[0]].copy()
train['y'] = hold_y
test = df_values.iloc[train.shape[0]:].copy()
print("Blah", train.shape, " ", test.shape)

# for c in train.columns:
#     if train[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(train[c].values) + list(test[c].values))
#         train[c] = lbl.transform(list(train[c].values))
#         test[c] = lbl.transform(list(test[c].values))

# train a model without X0, for the rows with X0 unclustered
# train_noX0 = train.copy().drop('X0', axis=1)
# test_noX0 = test.copy().drop('X0', axis=1)

# train_ = train[train.y < 200].copy()
# print(train_.shape)
ys = train['y'].values
y_mean = np.sum(ys)/len(ys)
SS = np.sum(np.power(ys - y_mean, 2))
print(SS)
def oof_r2(scores):
    global ys
    global SS
    return 1 + sum(scores) / len(scores) * len(ys) / SS

count = 0
def custom_score(y1, y2):
    global count
    print(count % 5 + 1, ":", r2_score(y1, y2))
    count += 1
    return r2_score(y1, y2)

def custom_score_2(y1, y2):
    global count
    print(count % 5 + 1, ":", mean_squared_error(y1, y2))
    count += 1
    return mean_squared_error(y1, y2)

def two_scorer(MSE=True):
    return make_scorer(custom_score if not MSE else custom_score_2, greater_is_better=False)

# print('Model Test:')
# m_test = ModelTest()
# m_test.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_test = m_test.predict(train.drop(['y'], axis=1).copy())
# y_test_test = m_test.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_test))
# scores_test = cross_val_score(m_test, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)

# print(scores_test)
# print('Average (scores_test):', sum(scores_test)/len(scores_test))


# print('Model 1:')
# m1 = Model1()
# m1.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_1 = m1.predict(train.drop(['y'], axis=1).copy())
# y_test_1 = m1.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_1))
# scores_1 = cross_val_score(m1, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
# print('Average (scores_1):', sum(scores_1)/len(scores_1))
# print('R2 score:', oof_r2(scores_1))

# print('Model 2:')
# m2 = Model2(disable=False)
# m2.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_2 = m2.predict(train.drop(['y'], axis=1).copy())
# y_test_2 = m2.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_2))
# scores_2 = cross_val_score(m2, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
# print('Average (scores_2):', sum(scores_2)/len(scores_2))
# print('R2 score:', oof_r2(scores_2))

# print('Model 3:')
# m3 = Model3(disable=False)
# m3.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_3 = m3.predict(train.drop(['y'], axis=1).copy())
# y_test_3 = m3.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_3))
# scores_3 = cross_val_score(m3, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
# print('Average (scores_3):', sum(scores_3)/len(scores_3))
# print('R2 score:', oof_r2(scores_3))

# print('Model 4:')
# m4 = Model4(disable=False)
# m4.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_4 = m4.predict(train.drop(['y'], axis=1).copy())
# y_test_4 = m4.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_4))
# scores_4 = cross_val_score(m4, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
# print('Average (scores_4):', sum(scores_4)/len(scores_4))
# print('R2 score:', oof_r2(scores_4))

# y_diff = train['y'] - y_train_4
# m1 = Model1()
# m1.fit(train.drop(['y'], axis=1).copy(), y_diff)
# y_train_1 = m1.predict(train.drop(['y'], axis=1).copy())
# print(mean_squared_error(train['y'], y_train_4 + y_train_1))

# plt.plot(train['y'], y_train_4)
tmp = np.polyfit(train['y'], y_train_1, deg=1)
plt.plot(train['y'], tmp[0] * train['y'] + tmp[1], color='black')
plt.scatter(train['y'], y_train_1)
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.show()

# print('Model noX0:')
# m_noX0 = Model1()
# m_noX0.fit(train_noX0.drop(['y'], axis=1).copy(), train_noX0['y'])
# y_train_noX0 = m_noX0.predict(train_noX0.drop(['y'], axis=1).copy())
# y_test_noX0 = m_noX0.predict(test_noX0.copy())
# print(mean_squared_error(train_noX0['y'], y_train_noX0))
# scores_noX0 = cross_val_score(m_noX0, train_noX0.drop(['y'], axis=1).copy(), train_noX0['y'], scoring=two_scorer(), cv=5)

# print('Model Mix:')
# m_mix = ModelMix([m1, m2, m3, m4], [0.2, 0.2, 0.3, 0.3])
# # m_mix.fit(train.drop(['y'], axis=1).copy(), train['y'])
# y_train_mix = m_mix.predict(train.drop(['y'], axis=1).copy())
# y_test_mix = m_mix.predict(test.copy())
# print(mean_squared_error(train['y'], y_train_mix))
# scores_mix = cross_val_score(m_mix, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)

# print(scores_1)
# print('Average (scores_1):', sum(scores_1)/len(scores_1))
# print(scores_2)
# print('Average (scores_2):', sum(scores_2)/len(scores_2))
# print(scores_3)
# print('Average (scores_3):', sum(scores_3)/len(scores_3))
# print(scores_4)
# print('Average (scores_4):', sum(scores_4)/len(scores_4))

# print(scores_noX0)
# print('Average (scores_noX0):', sum(scores_noX0)/len(scores_noX0))
# print('R2 score:', oof_r2(scores_noX0))

# print(scores_mix)
# print('Average (scores_mix):', sum(scores_mix)/len(scores_mix))
# print('R2 score:', oof_r2(scores_mix))


# for index, row in test.iterrows():
#     if test['X0'][index] == 3:
#         print(index, " ", y_test_mix[index], " ", y_test_noX0[index])
#         y_test_mix[index] = y_test_noX0[index]


# file_path = 'data/mercedes_benz_submission_16.csv'

# sub = pd.DataFrame()
# sub['ID'] = test['ID'].values
# sub['y'] = y_test_mix
# sub.to_csv(file_path, index=False)
# print('Training predictions written to ', file_path)