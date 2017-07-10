import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score

from model_1 import Model1
from model_2 import Model2
from model_3 import Model3
from model_4 import Model4
from model_5 import Model5
from model_mix import ModelMix
from model_test import ModelTest


train = pd.read_csv('./data/mercedes_benz_train.csv').iloc[:-32].sample(frac=1).reset_index(drop=True)
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
            return str(i * 2)
    # print(cat)
    return '3'

train_untouched = train.copy()
test_untouched = test.copy()

train['X0'] = train['X0'].transform(transformX0)
test['X0'] = test['X0'].transform(transformX0)

hold_y = train['y']
df_all = pd.concat([train, test]).drop(['y', 'X4'], axis=1)
df_numeric = df_all.select_dtypes(exclude=['object']).copy()
df_obj = df_all.select_dtypes(include=['object']).copy()
print(df_obj.shape)
print(df_numeric.shape)

# drop the numeric features where the column contains only one unique value
for col in df_numeric:
    cardinality = len(np.unique(train[col]))
    if cardinality == 1:
        df_numeric = df_numeric.drop(col, axis=1)

cols = df_obj.columns.values.tolist()
for col in cols:
    df_obj = pd.concat([df_obj, pd.get_dummies(df_obj[col], prefix=col)], axis=1)
    # df_obj[col] = pd.factorize(df_obj[col])[0]
    df_obj = df_obj.drop(col, axis=1)


#remove duplicates in df_numeric
cols = df_numeric.columns.tolist()
remove = []
for i in range(len(cols)-1):
    v = df_numeric[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,df_numeric[cols[j]].values):
            remove.append(cols[j])
            # print(' Column %s is identical to %s. Removing %s' % (str(cols[i]), str(cols[j]), str(cols[j])))
df_numeric.drop(remove, axis=1, inplace=True)

#remove duplicates in between obj and numeric
cols = df_numeric.columns.tolist()
cols2 = df_obj.columns.tolist()
remove = []
for i in range(len(cols2)):
    v = df_obj[cols2[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,df_numeric[cols[j]].values):
            remove.append(cols[j])
            # print(' Column %s is identical to %s. Removing %s' % (str(cols[i]), str(cols[j]), str(cols[j])))
df_numeric.drop(remove, axis=1, inplace=True)

df_values = pd.concat([df_numeric, df_obj], axis=1)

print(df_values.shape)
train = df_values.iloc[:train.shape[0]].copy()
train['y'] = hold_y
test = df_values.iloc[train.shape[0]:].copy()
print("Blah", train.shape, " ", test.shape)


ys = train['y'].values
y_mean = np.sum(ys)/len(ys)
SS = np.sum(np.power(ys - y_mean, 2))
print(SS)
def oof_r2(scores):
    global ys
    global SS
    return 1 + sum(scores) / len(scores) * len(ys) / SS
    # return sum(scores) / len(scores)

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
# print('R2 score:', oof_r2(scores_test))


print('Model 1:')
m1 = Model1()
m1.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_1 = m1.predict(train.drop(['y'], axis=1).copy())
y_test_1 = m1.predict(test.copy())
print(mean_squared_error(train['y'], y_train_1))
scores_1 = cross_val_score(m1, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
print('Average (scores_1):', sum(scores_1)/len(scores_1))
print('R2 score:', oof_r2(scores_1))

print('Model 2:')
m2 = Model2(disable=False)
m2.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_2 = m2.predict(train.drop(['y'], axis=1).copy())
y_test_2 = m2.predict(test.copy())
print(mean_squared_error(train['y'], y_train_2))
scores_2 = cross_val_score(m2, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
print('Average (scores_2):', sum(scores_2)/len(scores_2))
print('R2 score:', oof_r2(scores_2))

print('Model 3:')
m3 = Model3(disable=False)
m3.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_3 = m3.predict(train.drop(['y'], axis=1).copy())
y_test_3 = m3.predict(test.copy())
print(mean_squared_error(train['y'], y_train_3))
scores_3 = cross_val_score(m3, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
print('Average (scores_3):', sum(scores_3)/len(scores_3))
print('R2 score:', oof_r2(scores_3))

print('Model 4:')
m4 = Model4(disable=False)
m4.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_4 = m4.predict(train.drop(['y'], axis=1).copy())
y_test_4 = m4.predict(test.copy())
print(mean_squared_error(train['y'], y_train_4))
scores_4 = cross_val_score(m4, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
print('Average (scores_4):', sum(scores_4)/len(scores_4))
print('R2 score:', oof_r2(scores_4))

print('Model 5:')
m5 = Model5(disable=False)
m5.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_5 = m5.predict(train.drop(['y'], axis=1).copy())
y_test_5 = m5.predict(test.copy())
print(mean_squared_error(train['y'], y_train_5))
scores_5 = cross_val_score(m5, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)
print('Average (scores_5):', sum(scores_5)/len(scores_5))
print('R2 score:', oof_r2(scores_5))


print('Model Mix:')
m_mix = ModelMix([Model1(), Model2(), Model3(), Model4(), Model5()], [0.2, 0.2, 0.2, 0.2, 0.2])
m_mix.fit(train.drop(['y'], axis=1).copy(), train['y'])
y_train_mix = m_mix.predict(train.drop(['y'], axis=1).copy())
y_test_mix = m_mix.predict(test.copy())
print(mean_squared_error(train['y'], y_train_mix))
scores_mix = cross_val_score(m_mix, train.drop(['y'], axis=1).copy(), train['y'], scoring=two_scorer(), cv=5)

print(scores_1)
print('Average (scores_1):', sum(scores_1)/len(scores_1))
print(scores_2)
print('Average (scores_2):', sum(scores_2)/len(scores_2))
print(scores_3)
print('Average (scores_3):', sum(scores_3)/len(scores_3))
print(scores_4)
print('Average (scores_4):', sum(scores_4)/len(scores_4))
print(scores_5)
print('Average (scores_5):', sum(scores_5)/len(scores_5))

print(scores_mix)
print('Average (scores_mix):', sum(scores_mix)/len(scores_mix))
print('R2 score:', oof_r2(scores_mix))


file_path = 'data/mercedes_benz_submission_22.csv'

sub = pd.DataFrame()
sub['ID'] = test['ID'].values
sub['y'] = y_test_mix
sub.to_csv(file_path, index=False)
print('Training predictions written to ', file_path)