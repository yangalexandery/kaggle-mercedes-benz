import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import model_1, model_2, model_3

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


train = pd.read_csv('./data/mercedes_benz_train.csv')
test = pd.read_csv('./data/mercedes_benz_test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

y_train_1, y_test_1 = model_1.predict(train.copy(), test.copy())
y_train_2, y_test_2 = model_2.predict(train.copy(), test.copy())
y_train_3, y_test_3 = model_3.predict(train.copy(), test.copy())

avg = [0.75, 0.10, 0.15]
y_train = [y_train_1, y_train_2, y_train_3]
y_test = [y_test_1, y_test_2, y_test_3]

y_train_comp = np.zeros((train.shape[0],))
y_test_comp = np.zeros((test.shape[0],))
for i in range(3):
  y_train_comp += y_train[i] * avg[i]
  y_test_comp += y_test[i] * avg[i]

print('Composite R2 score on train data:')
print(r2_score(train['y'], y_train_comp))

file_path = 'data/mercedes_benz_submission_3.csv'

sub = pd.DataFrame()
sub['ID'] = test['ID'].values
sub['y'] = y_test_comp
sub.to_csv(file_path, index=False)
print('Training predictions written to ', file_path)