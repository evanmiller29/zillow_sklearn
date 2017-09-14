# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:10:58 2017

@author: evanm_000
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import datetime as dt

os.chdir('C:/Users/evanm_000/Documents/GitHub/Zillow-Kaggle')

#==============================================================================
# Reading in data
#==============================================================================

train = pd.read_csv('data/train_2016_v2.csv')
properties = pd.read_csv('data/properties_2016.csv')
sample = (pd.read_csv('data/sample_submission.csv')
            .rename(columns ={'ParcelId':'parcelid'}))

submission = pd.read_csv("data/sample_submission.csv")

os.chdir('C:/Users/evanm_000/Documents/GitHub/Zillow-Kaggle/code/Python')

from featEngineering import GetFeatures, sqFtFeat
from dataPrep import LBMdata, XGBdata
from modelFuncs import MAE, TestMonthMask

df_test = sample.merge(properties, on = 'parcelid', how = 'left')

#==============================================================================
# Setting up global variables
#==============================================================================

weights = {}
weights['XGB_WEIGHT'] = 0.6840
weights['BASELINE_WEIGHT'] = 0.0056
weights['OLS_WEIGHT'] = 0.0550
weights['XGB1_WEIGHT'] = 0.8083  # Weight of first in combination of two XGB models

weights['lgb_weight'] = (1 - weights['XGB_WEIGHT'] - weights['BASELINE_WEIGHT']) / (1 - weights['OLS_WEIGHT'])
weights['xgb_weight0'] = weights['XGB_WEIGHT'] / (1 - weights['OLS_WEIGHT'])
weights['baseline_weight0'] =  weights['BASELINE_WEIGHT'] / (1 - weights['OLS_WEIGHT'])

BASELINE_PRED = 0.0115

basicReg = True # Flag for deciding whether OLS/Lasso used
FeatEngineer = True
linearComb = True

overSampleTestMonths = False
underSampleElseMonths = False

testMonths = [10, 11, 12]

minoritySampleRate = 0.10
majorityReductionRate = 0.2

modelPreds = pd.DataFrame()

#==============================================================================
# Splitting of a validation set
#==============================================================================

trainTestMonths = 1.0#0.5 # Set to 1 for no validation
trainElseMonths = 1.0#0.8 # Set to 1 for no validation

print(train.shape)

trainValid = (train
                  .assign(transactiondate = lambda x: pd.to_datetime(x['transactiondate']))
                  .assign(trans_month = lambda x: x['transactiondate'].dt.month))

trainTestMonths = (trainValid
                       .loc[trainValid['trans_month'].isin(testMonths)]
                       .sample(frac = trainTestMonths)
                       .index
                       .values)

trainElseMonths = (trainValid
                       .loc[~trainValid['trans_month'].isin(testMonths)]
                       .sample(frac = trainElseMonths)
                       .index
                       .values
                       )

trainIdx = np.append(trainTestMonths, trainElseMonths)

trainValid = trainValid.drop('trans_month', axis = 1)

train = trainValid.loc[trainIdx, :]
valid = trainValid[np.logical_not(trainValid.index.isin(trainIdx))]

#==============================================================================
# Over sampling test months and under sampling other months
#==============================================================================

if overSampleTestMonths == True:

    trainOverSample = (train
                 .assign(transactiondate = lambda x: pd.to_datetime(x['transactiondate']))
                 .assign(trans_month = lambda x: x['transactiondate'].dt.month)
                 .assign(month_flag = lambda x: x['trans_month'].apply(lambda y: TestMonthMask(y, testMonths))))
    
    trainOverSampleMinority = (trainOverSample
                          .loc[trainOverSample['month_flag'] == 1]
                          .sample(frac = minoritySampleRate))
        
    trainResult = (pd.concat([trainOverSample, trainOverSampleMinority], axis = 0)
                    .drop(['trans_month', 'month_flag'], axis = 1))
    
    train = trainResult 

if underSampleElseMonths == True:
    
    trainUnderSample = (train
                        .assign(trans_month = lambda x: x['transactiondate'].dt.month)
                        .assign(month_flag = lambda x: x['trans_month'].apply(lambda y: TestMonthMask(y, testMonths))))
    
    trainUnderSampleMajority = (trainUnderSample
                                    .loc[trainUnderSample['month_flag'] == 0]
                                    .sample(frac = trainUnderSample))
    
    
#asd;flkajsd;lfkjas;ldfjl;asdkjfa;lsdjfa;    
    
#==============================================================================
# Running light GBM
#==============================================================================

x_train, d_train, train_columns = LBMdata(properties, train, True)

if FeatEngineer == True:
    x_test = sqFtFeat(df_test)[train_columns]
else:    
    x_test = df_test[train_columns]

lbgParams = {}
lbgParams['max_bin'] = 10
lbgParams['learning_rate'] = 0.0021 # shrinkage_rate
lbgParams['boosting_type'] = 'gbdt'
lbgParams['objective'] = 'regression'
lbgParams['metric'] = 'l1'          # or 'mae'
lbgParams['sub_feature'] = 0.35      # feature_fraction (small values => use very different submodels)
lbgParams['bagging_fraction'] = 0.85 # sub_row
lbgParams['bagging_freq'] = 40
lbgParams['num_leaves'] = 512        # num_leaf
lbgParams['min_data'] = 500         # min_data_in_leaf
lbgParams['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lbgParams['verbose'] = 0
lbgParams['feature_fraction_seed'] = 2
lbgParams['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
clf = lgb.train(lbgParams, d_train, 430)

print("\nPrepare for LightGBM prediction ...")

print("   ...")

print("   Preparing x_test...")

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")

modelPreds['lgbm'] = clf.predict(x_test)

print( "\nUnadjusted LightGBM predictions:" )
print( modelPreds['lgbm'].head())

#==============================================================================
# Run Xgboost for the first time
#==============================================================================

num_boost_rounds = 500

x_train, x_test, y_train, y_mean = XGBdata(properties, train, FeatEngineer, -0.4, 0.419)

print("\nSetting up data for XGBoost ...")
# xgboost params
xgbParams = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgbParams, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
modelPreds['xgb_pred1'] = model.predict(dtest)

print( "\nFirst XGBoost predictions:" )
print( modelPreds['xgb_pred1'].head() )

#==============================================================================
# Running Xgboost for the second time
#==============================================================================

num_boost_rounds = 300

print("\nSetting up data for XGBoost ...")

xgbParams = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
model = xgb.train(dict(xgbParams, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost again ...")
modelPreds['xgb_pred2'] = model.predict(dtest)

print( "\nSecond XGBoost predictions:" )
print( modelPreds['xgb_pred2'].head() )

##### COMBINE XGBOOST RESULTS
modelPreds['xgb_pred'] = weights['XGB1_WEIGHT'] * modelPreds['xgb_pred1'] + (1-weights['XGB1_WEIGHT']) * modelPreds['xgb_pred2']

print( "\nCombined XGBoost predictions:" )
print( modelPreds['xgb_pred'].head() )

#==============================================================================
# Running OLS
#==============================================================================

print(len(train),len(properties),len(submission))

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values

test = pd.merge(sample, properties, how='left', on='parcelid')

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = GetFeatures(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = GetFeatures(test[col])

if basicReg == True:
    
    reg = LinearRegression(n_jobs=-1)
else:
    from sklearn.linear_model import Lasso
    reg = Lasso(alpha = 0.1)
    
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']

#==============================================================================
# Combining predictions
#==============================================================================

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )

pred0 = weights['xgb_weight0'] * modelPreds['xgb_pred'] + weights['baseline_weight0'] * BASELINE_PRED + weights['lgb_weight'] * modelPreds['lgbm']

print( "\nCombined XGB/LGB/baseline predictions:" )
print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = weights['OLS_WEIGHT']*reg.predict(GetFeatures(test)) + (1-weights['OLS_WEIGHT'])*pred0
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
print( submission.head() )

##### WRITE THE RESULTS

from datetime import datetime

os.getcwd()

print( "\nWriting results to disk ..." )
submission.to_csv('../../submissions/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')

print( "\nFinished ...")