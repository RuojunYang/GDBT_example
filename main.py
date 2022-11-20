import os
import math
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
# from pydotplus import graph_from_dot_data
# from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from xgboost import XGBRegressor

cereal = pd.read_csv('./cereal.csv')
# print(cereal[pd.isna(cereal)].head(5))
cereal = cereal.dropna()
# print(cereal.head(5))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(cereal['mfr'])
cereal['mfr'] = integer_encoded

integer_encoded = label_encoder.fit_transform(cereal['type'])
cereal['type'] = integer_encoded

# print(cereal.head(5))

X = cereal[cereal.columns[1:-1]]
y = cereal[cereal.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True)

clf = GradientBoostingRegressor(random_state=0, max_depth = 2, n_estimators = 15, learning_rate=1)
clf.fit(X_train, y_train)

temp = pd.DataFrame()
temp['predict'] = clf.predict(X_test)
temp['true'] = np.array(y_test)

# print(len(xgb.predict(X_test)), len(y_test))
print(temp)

print('GDBT: ' + str(clf.score(X_test, y_test)))

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
print('XGB: ' + str(xgb.score(X_test, y_test)))

temp = pd.DataFrame()
temp['predict'] = xgb.predict(X_test)
temp['true'] = np.array(y_test)

# print(len(xgb.predict(X_test)), len(y_test))
print(temp)

