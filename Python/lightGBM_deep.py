    # -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 21:38:45 2017

@author: evanm_000
"""

#==============================================================================
# Reading in libraries
#==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from datetime import datetime
import datetime as dt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.decomposition import PCA
import os
import time

project = 'Zillow'
basePath = 'C:/' + project
funcPath = basePath + '/code/Python'
subPath = 'C:/Zillow/submissions'

os.chdir(basePath)

#==============================================================================
# Loading data / functions
#==============================================================================

print('Loading data ...')

train = pd.read_csv('data/train_2016_v2.csv')
properties = pd.read_csv('data/properties_2016.csv', low_memory=False)
sample = (pd.read_csv('data/sample_submission.csv')
            .rename(columns = {'ParcelId':'parcelid'}))

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']

print("Prepare for the prediction ...")
df_test = sample.merge(properties, on='parcelid', how='left')

os.chdir(funcPath)

from modelFuncs import MAE, TrainValidSplit
from dataPrep import DataFrameDeets, ConvertCats
from featEngineering import ApplyFeatEngineering

#==============================================================================
# Setting up results logging
#==============================================================================

resLog = {}
resLog ['coresUsed'] = 6

featFuncs = ['ExtractTimeFeats', 'sqFtFeat']
resLog['funcsUsed'] = ', '.join(featFuncs)

dropCols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode']
resLog['colsDrop'] = ', '.join(dropCols)

resLog['overSampTestMonths'] = False
resLog['underSampElseMonths'] = False

resLog['minorSampRate'] = 0.10
resLog['majorRedRate'] = 0.2

funcsUsed = ['ExtractTimeFeats', 'sqFtFeat', 'ExpFeatures']
resLog['funcsUsed'] = ', '.join(funcsUsed)

resLog['grid_search'] = True

#==============================================================================
# Model hyperparameters
#==============================================================================

params = {}
params['learning_rate'] = 0.0001
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.50
params['num_leaves'] = 60
params['min_data'] = 500
params['min_hessian'] = 1
params['bagging_fraction'] = 0.55
params['max_depth'] = 10

resLog['paramsUsed'] = ', '.join([k + ' = ' + str(v) for k, v in params.items()])

#==============================================================================
# Feature engineering
#==============================================================================

for c, dtype in zip(properties.columns, properties.dtypes):	
    if dtype == np.float64:		
        properties[c] = properties[c].astype(np.float32)

df_train = (train.merge(properties, how='left', on='parcelid')
                .assign(transactiondate = lambda x: pd.to_datetime(x['transactiondate'])))

DataFrameDeets(df_train, 'train + properties file - before feat engineering..')

df_train = ApplyFeatEngineering(df_train, 'training + properties set', funcsUsed)

#==============================================================================
# Splitting off a validation set
#==============================================================================

resLog['trainTestMonths'] = 0.5 # Set to 1 for no validation
resLog['trainElseMonths'] = 0.8 # Set to 1 for no validation

monthSplits = {}
monthSplits['trainTestMonths'] = resLog['trainTestMonths']
monthSplits['trainElseMonths'] = resLog['trainElseMonths']

testMonths = [10, 11, 12]
resLog['testMonths'] = ', '.join(str(month) for month in testMonths)

train, valid = TrainValidSplit(df_train, testMonths, monthSplits)

DataFrameDeets(train, 'training')
DataFrameDeets(valid, 'validation')

#==============================================================================
# Preparing the data for modelling
#==============================================================================

x_train = train.drop(dropCols, axis=1)
y_train = train['logerror'].values
print(x_train.shape, y_train.shape)

x_valid = valid.drop(dropCols, axis=1)
y_valid = valid['logerror'].values
print(x_valid.shape, y_valid.shape)

train_columns = x_train.columns
valid_columns = x_valid.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

x_train = ConvertCats(x_train) 

for c in x_valid.dtypes[x_valid.dtypes == object].index.values:
    x_valid[c] = (x_valid[c] == True)

x_valid = ConvertCats(x_valid) 

x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

resLog['model'] = 'lightGBM'

#==============================================================================
# Setting up model run - Pipelines
#==============================================================================

pipeline = Pipeline([('imp', Imputer(missing_values='NaN', axis=0)),
                     ('feats', FeatureUnion([
                             ('feat2', PolynomialFeatures(2)),
                             ('feat3', PolynomialFeatures(3)),
                             ('pca5', PCA(n_components= 5)),
                             ('pca10', PCA(n_components= 5))
                             ])),
                     ('feat_select', SelectKBest(),
                     ('lgbm', LGBMRegressor(metric = 'mae'))
                     
])

parameters = dict(imp__strategy=['mean', 'median', 'most_frequent'],
                    feat_select__k=[10, 25, 50, 75], 
                    lgbm__boosting_type = ['gbdt', 'dart'],
                    lgbm__learning_rate = [0.001, 0.1, 1],
                    lgbm__n_estimators = [20, 80, 100],
                    lgbm__max_depth = [10, 4, 1],
                    lgbm__num_leaves = [5, 20, 30],
                    lgbm__sub_feature = [0.25, 0.75, 0.95],
                    lgbm__bagging_fraction = [0.25, 0.75, 0.95],
                    lgbm__min_data = [20, 100, 500],
                    lgbm__min_hessian = [1, 10]
                                                     
)

CV = GridSearchCV(pipeline, parameters, scoring = 'mean_absolute_error', 
                  n_jobs= resLog['coresUsed'])

start = time.time()
resLog['startTime'] = dt.datetime.fromtimestamp(start).strftime('%c')

CV.fit(x_train, y_train)    

end = time.time()

timeElapsed = end - start

m, s = divmod(timeElapsed, 60)
h, m = divmod(m, 60)

resLog['timeElapsed'] = "%d:%02d:%02d" % (h, m, s)
 
print(CV.best_params_)    
print(CV.best_score_)    

y_pred = CV.predict(x_valid)
resLog['cvAcc'] = round(MAE(y_valid, y_pred), 5)

#==============================================================================
# Preparing the submission
#==============================================================================

CV.reset_parameter({"num_threads":4})

print( "\nPredicting using LightGBM and month features: ..." )

for i in range(len(test_dates)):
    
    x_test = df_test.drop(['parcelid', 'propertyzoningdesc', 'propertycountylandusecode'], axis = 1)    
    
    x_test['transactiondate'] = test_dates[i]   
    x_test = ApplyFeatEngineering(x_test, 'test set', funcsUsed)   
    x_test = x_test.drop('transactiondate', axis = 1)
    
    for c in x_test.dtypes[x_test.dtypes == object].index.values:
        x_test[c] = (x_test[c] == True)
        
    x_test = ConvertCats(x_test)        
        
    x_test = x_test.values.astype(np.float32, copy=False)

    pred = CV.predict(x_test)
    sample[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    print('predict...', test_dates[i])    

#==============================================================================
# Saving predictions to file
#==============================================================================

print('Saving predictions to file..')

os.chdir(subPath)
sample.to_csv('sub{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), resLog['cvAcc']), index=False, float_format='%.4f')

#==============================================================================
# Logging outputs
#==============================================================================

print('Outputting logs..')

os.chdir(basePath)

logDF = (pd.DataFrame(list(resLog.items()))
            .rename(columns = {0: 'desc', 1: 'values'})
            .pivot(columns = 'desc')
            .fillna(method = 'ffill')
            .fillna(method = 'bfill'))
            
cols = [val[1] for val in logDF.columns.values]
logDF.columns = cols
           
logs = pd.read_csv('modelling_log.csv')
logDF = logs.append(logDF.loc[0, :])
logDF.to_csv('modelling_log.csv', index_label = False, index = False)