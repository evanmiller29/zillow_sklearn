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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import os
import time

homeComp = True # If home comp reduce the train set to ensure it doesn't blow out the memory

basePath = 'C:/Users/Evan/Documents/GitHub/Zillow'
funcPath = 'C:/Users/Evan/Documents/GitHub/zillow_sklearn/Python'
subPath = 'F:/Nerdy Stuff/Kaggle submissions/Zillow'

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
from custom_transformers import ColumnExtractor

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

#==============================================================================
# Feature engineering
#==============================================================================

for c, dtype in zip(properties.columns, properties.dtypes):	
    if dtype == np.float64:		
        properties[c] = properties[c].astype(np.float32)

df_train = (train.merge(properties, how='left', on='parcelid')
                .assign(transactiondate = lambda x: pd.to_datetime(x['transactiondate'])))

DataFrameDeets(df_train, 'train + properties file - before feat engineering..')

#==============================================================================
# df_train = ApplyFeatEngineering(df_train, 'training + properties set', funcsUsed)
#==============================================================================

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

if (homeComp == True):
    
    print('Reducing the training set for testing..')
    
    train = train.sample(frac = 0.2)
    DataFrameDeets(train, 'train for home prototyping')    
    

x_train = train.drop(dropCols, axis=1)
y_train = train['logerror'].values
print(x_train.shape, y_train.shape)

x_valid = valid.drop(dropCols, axis=1)
y_valid = valid['logerror'].values
print(x_valid.shape, y_valid.shape)

train_columns = x_train.columns
valid_columns = x_valid.columns

print('Converting some rogue booleans back to their proper format..')

boolVars = ['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']
x_train[boolVars] = x_train[boolVars].astype(bool)
x_valid[boolVars] = x_valid[boolVars].astype(bool)

print('Separating out the different variable types so I can perform different pipelines..')

idVars = [i for e in ['id',  'flag', 'has'] for i in list(train_columns) if e in i] + ['fips', 'hashottuborspa']
countVars = [i for e in ['cnt',  'year', 'nbr', 'number'] for i in list(train_columns) if e in i]
taxVars = [col for col in train_columns if 'tax' in col and 'flag' not in col]
          
ttlVars = idVars + countVars + taxVars
exclVars = [i for e in ['census',  'tude'] for i in list(train_columns) if e in i]

contVars = [col for col in train_columns if col not in ttlVars + exclVars]

#==============================================================================
# Filling missing values of categorical / tax variables
#==============================================================================

print('Filling the sparse categorical variables with 0..')

x_train[idVars] = x_train[idVars].fillna(0)
x_valid[idVars] = x_valid[idVars].fillna(0)

print('Filling the sparse tax variables with 0..')

x_train[taxVars] = x_train[taxVars].fillna(0)
x_valid[taxVars] = x_valid[taxVars].fillna(0)

#==============================================================================
# Recoding rare sq ft variables..
#==============================================================================

#==============================================================================
# x_train = x_train.values.astype(np.float32, copy=False)
# x_valid = x_valid.values.astype(np.float32, copy=False)
#==============================================================================

#==============================================================================
# Setting up model run - Pipelines
#==============================================================================

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)
    
for c in x_valid.dtypes[x_valid.dtypes == object].index.values:
    x_valid[c] = (x_valid[c] == True)

pipelineSmall = Pipeline([(('cont_feats'), ColumnExtractor(contCols)),
                     ('imp', Imputer(missing_values='NaN', axis=0)),
                     ('scaler', StandardScaler()),
                     ('feats', FeatureUnion([
                             ('feat2', PolynomialFeatures(2)),
                             ('pca5', PCA(n_components= 5)),
                             ('pca10', PCA(n_components= 10))
                             ])),
                     ('feat_select', SelectKBest()),
                     ('rf', RandomForestRegressor())
                     
])

#==============================================================================
# Try and check function transformer works like this:
# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html   
#==============================================================================

pipelineBigger = Pipeline([
        ('union', FeatureUnion([
            ('continuous', Pipeline([
                    ('contExtract', ColumnExtractor(contCols)),
                    ('imp', Imputer(missing_values='NaN', axis=0)),
                    ('feats', FeatureUnion([
                                 ('feat2', PolynomialFeatures(2)),
                                 ('pca5', PCA(n_components= 5)),
                                 ('pca10', PCA(n_components= 10))
                                 ])),
                    ('scaler', StandardScaler()),
                    ])
            ), 
            ('factors', Pipeline([
                    ('factExtract', ColumnExtractor(idCols)),
                    ('ohe', OneHotEncoder(n_values=5))
                    ]))                
                ])),
    ('feat_select', SelectKBest()),
    ('rf', RandomForestRegressor())              
])
     
parameters = dict(imp__strategy=['mean', 'median', 'most_frequent'],
                    feat_select__k=[10, 25, 50, 75], 
                    rf__n_estimators = [20]
                                                     
)    

CV = GridSearchCV(pipelineSmall, parameters, scoring = 'mean_absolute_error', n_jobs= 1)

start = dt.datetime.fromtimestamp(time.time()).strftime('%c')
CV.fit(x_train, y_train)    

end = time.time()
timeElapsed = end - start

m, s = divmod(timeElapsed, 60)
h, m = divmod(m, 60)

print("Time elapsed: %d:%02d:%02d" % (h, m, s))
 
print(CV.best_params_)    
print(CV.best_score_)    

y_pred = CV.predict(x_valid)
print('MAE on validation set: %s' % (round(MAE(y_valid, y_pred), 5))) #MAE on validation set: 0.0757

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