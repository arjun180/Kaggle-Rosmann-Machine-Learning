import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import re
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

import xgboost as xgb

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe



def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w



def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe




def process_data(data):
	'''
	Processes data for model
		INPUT: DataFrame
		OUTPUT: DataFrame
	'''
	# Merge store data
	data = data.merge(store, on = 'Store', copy = False)

	# Break down date column
	data['year'] = data.Date.apply(lambda x: x.year)
	data['month'] = data.Date.apply(lambda x: x.month)
	#     data['dow'] = data.Date.apply(lambda x: x.dayofweek)
	data['woy'] = data.Date.apply(lambda x: x.weekofyear)
	data.drop(['Date'], axis = 1, inplace= True)

	# Calculate time competition open time in months
	data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + \
	(data.month - data.CompetitionOpenSinceMonth)
	data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
	data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis = 1, 
	         inplace = True)

	# Promo open time in months
	data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + \
	(data.woy - data.Promo2SinceWeek) / float(4)
	data['PromoOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
	data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis = 1, 
	         inplace = True)

	# Get promo months
	data['p_1'] = data.PromoInterval.apply(lambda x: x[:3] if type(x) == str else 0)
	data['p_2'] = data.PromoInterval.apply(lambda x: x[4:7] if type(x) == str else 0)
	data['p_3'] = data.PromoInterval.apply(lambda x: x[8:11] if type(x) == str else 0)
	data['p_4'] = data.PromoInterval.apply(lambda x: x[12:15] if type(x) == str else 0)


	# Get dummies for categoricals
	data = pd.get_dummies(data, columns = ['p_1', 'p_2', 'p_3', 'p_4', 
	                                       'StateHoliday' , 
	                                       'StoreType', 
	                                       'Assortment'])
	data.drop(['Store',
	           'PromoInterval', 
	           'p_1_0', 'p_2_0', 'p_3_0', 'p_4_0', 
	           'StateHoliday_0', 
	           'year'], axis=1,inplace=True)


	# Fill in missing values
	data = data.fillna(0)
	data = data.sort_index(axis=1)

	return data




## Start of main script

# Load data
data = pd.read_csv('/Users/arjun/Documents/Kaggle/Rosmann/Data/train.csv', parse_dates = ['Date'])
store = pd.read_csv('/Users/arjun/Documents/Kaggle/Rosmann/Data/store.csv') 
print('training data loaded')

# Only use stores that are open to train
data = data[data['Open'] != 0]

# Process training data
data = process_data(data)
print('training data processed')

# Set up training data
X_train = data
y_train = data.Sales



params = {"objective": "reg:linear",
          "eta": 0.05,
          "max_depth": 10,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "silent": 1,
          "nthread":4
          }
num_trees = 20


print("Train a XGBoost model")
val_size = 100000

X_train, X_test = cross_validation.train_test_split(X_train, test_size=0.01)

dtrain = xgb.DMatrix(X_train, np.log(X_train["Sales"] + 1))
dvalid = xgb.DMatrix(X_test, np.log(X_test["Sales"] + 1))


test = pd.read_csv('/Users/arjun/Documents/Kaggle/Rosmann/Data/test.csv', parse_dates = ['Date'])
test = process_data(test)
#test = test[test['Open'] != 0]

# Ensure same columns in test data as training
for col in data.columns:
    if col not in test.columns:
        test[col] = np.zeros(test.shape[0])

test = test.sort_index(axis=1).set_index('Id')
print('test data loaded and processed')

test = test.drop(['Sales', 'Customers'], axis=1).values




dtest = xgb.DMatrix(test)
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)

print("Validating")

train_probs = gbm.predict(xgb.DMatrix(X_test))
indices = train_probs < 0
train_probs[indices] = 0
error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)
print('error', error)

print("Make predictions on the test set")
test_probs = gbm.predict(xgb.DMatrix(test))
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})
submission.to_csv("/Users/Documents/Kaggle/Rosmann/Submissions/xgboost_submission.csv", index=False)


