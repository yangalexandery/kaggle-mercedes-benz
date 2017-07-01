import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import model_1, model_2, model_3, model_4

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


train = pd.read_csv('./data/mercedes_benz_train.csv')
test = pd.read_csv('./data/mercedes_benz_test.csv')

# def transformX0(cat):
#     groups = [
#         ['bc', 'az'],
#         ['ac','am','l','b','aq','u','ad','e','al','s',
#             'n','y','t','ai','k','f','z','o','ba','m','q'],
#         ['d','ay','h','aj','v','ao','aw'],
#         ['c','ax','x','j','w','i','ak','g','at','ab',
#             'af','r','as','a','ap','au','aa']
#     ]
#     for i in range(len(groups)):
#         if cat in groups[i]:
#             return i
#     print(cat)
#     return 5

train_untouched = train.copy()
test_untouched = test.copy()
# train['X0'] = train['X0'].transform(transformX0)
# test['X0'] = test['X0'].transform(transformX0)
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# train a model without X0, for the rows with X0 unclustered
train_noX0 = train.copy().drop('X0', axis=1)
test_noX0 = test.copy().drop('X0', axis=1)

y_train_1, y_test_1 = model_1.predict('XGB Model', train.copy(), test.copy(), disable=False)
y_train_2, y_test_2 = model_2.predict('Stacked Model 1', train.copy(), test.copy(), disable=False)
y_train_3, y_test_3 = model_3.predict('Stacked Model 2', train_untouched.copy(), test_untouched.copy(), disable=False)
y_train_4, y_test_4 = model_1.predict('XGB no X0 Model', train_noX0.copy(), test_noX0.copy(), disable=False)

avg = [0.60, 0.05, 0.20, 0.15]
y_train = [y_train_1, y_train_2, y_train_3, y_train_4]
y_test = [y_test_1, y_test_2, y_test_3, y_test_4]

y_train_comp = np.zeros((train.shape[0],))
y_test_comp = np.zeros((test.shape[0],))
for i in range(4):
  y_train_comp += y_train[i] * avg[i]
  y_test_comp += y_test[i] * avg[i]

print('Composite R2 score on train data:')
print(r2_score(train['y'], y_train_comp))

for index, row in test.iterrows():
    # if test['X0'][index] == 5:
    #     print(index, " ", y_test_comp[index], " ", y_test_4[index])
    #     y_test_comp[index] = y_test_4[index]
    if test_untouched['X0'][index] in ['av', 'ag', 'an', 'ae', 'p', 'bb']:
        print(index, " ", y_test_comp[index], " ", y_test_4[index])
        y_test_comp[index] = y_test_4[index]


file_path = 'data/mercedes_benz_submission_6.csv'

sub = pd.DataFrame()
sub['ID'] = test['ID'].values
sub['y'] = y_test_comp
sub.to_csv(file_path, index=False)
print('Training predictions written to ', file_path)