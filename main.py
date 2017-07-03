import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

import model_1, model_2, model_3
from model_1 import Model1
from model_2 import Model2
from model_3 import Model3
from model_mix import ModelMix


train = pd.read_csv('./data/mercedes_benz_train.csv').sample(frac=1).reset_index(drop=True)
test = pd.read_csv('./data/mercedes_benz_test.csv')

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
            return i * 2
    # print(cat)
    return 3

train_untouched = train.copy()
test_untouched = test.copy()
train['X0'] = train['X0'].transform(transformX0)
test['X0'] = test['X0'].transform(transformX0)

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
        
# for col in df_obj:
#     df_obj[col] = pd.factorize(df_obj[col])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)
print(df_values.shape)
train = df_values.iloc[:train.shape[0]].copy()
train['y'] = hold_y
test = df_values.iloc[test.shape[0]:].copy()
print("Blah", train.shape, " ", test.shape)

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# train a model without X0, for the rows with X0 unclustered
train_noX0 = train.copy().drop('X0', axis=1)
test_noX0 = test.copy().drop('X0', axis=1)

# train_ = train[train.y < 200].copy()
# print(train_.shape)

count = 0
def custom_score(y1, y2):
    global count
    print(count % 5 + 1, ":", mean_squared_error(y1, y2))
    count += 1
    return mean_squared_error(y1, y2)

def two_scorer():
    return make_scorer(custom_score, greater_is_better=False)

print('Model 1:')
m1 = Model1()
m1.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_1 = m1.predict(train.drop(['y'], axis=1).copy())
y_test_1 = m1.predict(test.copy())
print(mean_squared_error(train['y'], y_train_1))
scores_1 = cross_val_score(m1, train.drop(['y'], axis=1), train['y'], scoring=two_scorer(), cv=5)

print('Model 2:')
m2 = Model2(disable=False)
m2.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_2 = m2.predict(train.drop(['y'], axis=1).copy())
y_test_2 = m2.predict(test.copy())
print(mean_squared_error(train['y'], y_train_2))
scores_2 = cross_val_score(m2, train.drop(['y'], axis=1), train['y'], scoring=two_scorer(), cv=5)

print('Model 3:')
m3 = Model3(disable=False)
m3.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_3 = m3.predict(train.drop(['y'], axis=1).copy())
y_test_3 = m3.predict(test.copy())
print(mean_squared_error(train['y'], y_train_3))
scores_3 = cross_val_score(m3, train.drop(['y'], axis=1), train['y'], scoring=two_scorer(), cv=5)

print('Model noX0:')
m_noX0 = Model1()
m_noX0.fit(train_noX0.drop(['y'], axis=1).copy(), train_noX0['y'])
y_train_noX0 = m_noX0.predict(train_noX0.drop(['y'], axis=1).copy())
y_test_noX0 = m_noX0.predict(test_noX0.copy())
print(mean_squared_error(train_noX0['y'], y_train_noX0))
scores_noX0 = cross_val_score(m_noX0, train_noX0.drop(['y'], axis=1), train_noX0['y'], scoring=two_scorer(), cv=5)

print('Model Mix:')
m_mix = ModelMix([m1, m2, m3], [0.3, 0.3, 0.4])
# m_mix.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_mix = m_mix.predict(train.drop(['y'], axis=1).copy())
y_test_mix = m_mix.predict(test.copy())
print(mean_squared_error(train['y'], y_train_mix))
scores_mix = cross_val_score(m_mix, train.drop(['y'], axis=1), train['y'], scoring=two_scorer(), cv=5)

print(scores_1)
print('Average (scores_1):', sum(scores_1)/len(scores_1))
print(scores_2)
print('Average (scores_2):', sum(scores_2)/len(scores_2))
print(scores_3)
print('Average (scores_3):', sum(scores_3)/len(scores_3))

print(scores_noX0)
print('Average (scores_noX0):', sum(scores_noX0)/len(scores_noX0))

print(scores_mix)
print('Average (scores_mix):', sum(scores_mix)/len(scores_mix))

print('R^2 score of mixture model:', r2_score(train['y'], y_train_mix))

for index, row in test.iterrows():
    if test['X0'][index] == 3:
        print(index, " ", y_test_mix[index], " ", y_test_noX0[index])
        y_test_mix[index] = y_test_noX0[index]


file_path = 'data/mercedes_benz_submission_9.csv'

sub = pd.DataFrame()
sub['ID'] = test['ID'].values
sub['y'] = y_test_mix
sub.to_csv(file_path, index=False)
print('Training predictions written to ', file_path)