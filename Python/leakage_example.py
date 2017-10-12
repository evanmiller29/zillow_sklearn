# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:43:38 2017

@author: Evan
"""

#==============================================================================
# This code is taken from here (and isn't my original work):
# https://www.kaggle.com/aharless/kk-s-lgbm-with-full-data/code
# Also adding https://www.kaggle.com/arjanso/kernel-density-estimation-for-predicting-logerror
#==============================================================================

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import gc
import datetime as dt
import os
from sklearn.neighbors import KDTree

basePath = 'C:/Users/Evan/Documents/GitHub/Zillow/data'
subPath = 'F:/Nerdy Stuff/Kaggle submissions/Zillow'
os.chdir(basePath)

print('Loading data...')

properties2016_raw = pd.read_csv('properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('properties_2017.csv', low_memory = False)
train2016 = pd.read_csv('train_2016_v2.csv')
train2017 = pd.read_csv('train_2017.csv')
sample_submission = pd.read_csv('sample_submission.csv', low_memory = False)

#==============================================================================
# Model run meta-data
#==============================================================================

runDetails = dict()
modelDesc = dict()

runDetails['MAKE_SUBMISSION'] = True          # Generate output file.
runDetails['CV_ONLY'] = False                 # Do validation only; do not generate predicitons.
runDetails['FIT_FULL_TRAIN_SET'] = True       # Fit model to full training set after doing validation.
runDetails['FIT_2017_TRAIN_SET'] = False      # Use 2017 training data for full fit (no leak correction)
runDetails['USE_SEASONAL_FEATURES'] = False
runDetails['VAL_SPLIT_DATE'] = '2016-09-15'   # Cutoff date for validation split
runDetails['FUDGE_FACTOR_SCALEDOWN'] = 0.3    # exponent to reduce optimized fudge factor for prediction
runDetails['OPTIMIZE_FUDGE_FACTOR'] = True    # Optimize factor by which to multiply predictions.

if runDetails['USE_SEASONAL_FEATURES']:
    basedate = pd.to_datetime('2015-11-15').toordinal()

#==============================================================================
# Defining functions 
#==============================================================================

def calculate_features(df):
    
    # Nikunj's features
    # Number of properties in the zip
    df['N-zip_count'] = df['regionidzip'].agg('count')
    # Number of properties in the city
    df['N-city_count'] = df['regionidcity'].agg('count')
    # Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & \
                         (df['pooltypeid10']>0) & \
                         (df['airconditioningtypeid']!=5))*1 

    # More features
    # Mean square feet of neighborhood properties
    df['mean_area'] = df.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].agg('median')
    # Median year of construction of neighborhood properties
    df['med_year'] = df.groupby('regionidneighborhood')['yearbuilt'].agg('median')
    # Neighborhood latitude and longitude
    df['med_lat'] = df.groupby('regionidneighborhood')['latitude'].agg('median')
    df['med_long'] = df.groupby('regionidneighborhood')['longitude'].agg('median')

#    df['zip_std'] = df['regionidzip'].map(zipstd)
#    df['city_std'] = df['regionidcity'].map(citystd)
#    df['hood_std'] = df['regionidneighborhood'].map(hoodstd)
    
    if runDetails['USE_SEASONAL_FEATURES']:
        df['cos_season'] = ( (pd.to_datetime(df['transactiondate']).apply(lambda x: x.toordinal()-basedate)) * \
                             (2*np.pi/365.25) ).apply(np.cos)
        df['sin_season'] = ( (pd.to_datetime(df['transactiondate']).apply(lambda x: x.toordinal()-basedate)) * \
                             (2*np.pi/365.25) ).apply(np.sin)  
        
    return(df)

def preptest(test):
    test[['latitude', 'longitude']] /= 1e6
    test[['latitude', 'longitude']] /= 1e6
    test['censustractandblock'] /= 1e12
    test['censustractandblock'] /= 1e12

    for column in test.columns:
        if test[column].dtype == int:
            test[column] = test[column].astype(np.int32)
        if test[column].dtype == float:
            test[column] = test[column].astype(np.float32)
            
    return test

# Create a new version of 2016 properties data that takes all non-tax variables from 2017

taxvars = ['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']
tax2016 = properties2016_raw[['parcelid']+taxvars]
properties2016 = properties2017.drop(taxvars,axis=1).merge(tax2016, 
                 how='left', on='parcelid').reindex_axis(properties2017.columns, axis=1)

# Create a training data set
train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

train2016 = calculate_features(train2016)
train2017 =  calculate_features(train2017)

train = pd.concat([train2016, train2017], axis = 0)

# Create separate test data sets for 2016 and 2017
test2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')
test2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), 
                how = 'left', on = 'ParcelId')

test2016 = calculate_features(test2016)
test2017 =  calculate_features(test2017)

del properties2016, properties2017, train2016, train2017
gc.collect();

print('Memory usage reduction...')

train[['latitude', 'longitude']] /= 1e6
train['censustractandblock'] /= 1e12

test2016 = preptest(test2016)
test2017 = preptest(test2017)
        
print('Feature engineering...')

train['month'] = (pd.to_datetime(train['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(train['transactiondate']).dt.month
train = train.drop('transactiondate', axis = 1)
from sklearn.preprocessing import LabelEncoder
non_number_columns = train.dtypes[train.dtypes == object].index.values

for column in non_number_columns:
    train_test = pd.concat([train[column], test2016[column], test2017[column]], axis = 0)
    encoder = LabelEncoder().fit(train_test.astype(str))
    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)
    test2016[column] = encoder.transform(test2016[column].astype(str)).astype(np.int32)
    test2017[column] = encoder.transform(test2017[column].astype(str)).astype(np.int32)
    
feature_names = [feature for feature in train.columns[2:] if feature != 'month']

month_avgs = train.groupby('month').agg('mean')['logerror'].values - train['logerror'].mean()
                             
print('Preparing arrays and throwing out outliers...')
X_train = train[feature_names].values
y_train = train['logerror'].values
X_test2016 = test2016[feature_names].values
X_test2017 = test2017[feature_names].values

del test2016, test2017;
gc.collect();

month_values = train['month'].values
month_avg_values = np.array([month_avgs[month - 1] for month in month_values]).reshape(-1, 1)
X_train = np.hstack([X_train, month_avg_values])

X_train = X_train[np.abs(y_train) < 0.4, :]
y_train = y_train[np.abs(y_train) < 0.4]

print('Training LGBM model...')
ltrain = lgb.Dataset(X_train, label = y_train)

params = {}
params['metric'] = 'mae'
params['max_depth'] = 100
params['num_leaves'] = 32
params['feature_fraction'] = .85
params['bagging_fraction'] = .95
params['bagging_freq'] = 8
params['learning_rate'] = 0.0025
params['verbosity'] = 0

lgb_model = lgb.train(params, ltrain, valid_sets = [ltrain], verbose_eval=200, num_boost_round=2930)
                                   
print('Making predictions and praying for good results...')

X_test2016 = np.hstack([X_test2016, np.zeros((X_test2016.shape[0], 1))])
X_test2017 = np.hstack([X_test2016, np.zeros((X_test2017.shape[0], 1))])

folds = 20
n = int(X_test2016.shape[0] / folds)

for j in range(folds):
    results = pd.DataFrame()

    if j < folds - 1:
            X_test2016_ = X_test2016[j*n: (j+1)*n, :]
            X_test2017_ = X_test2017[j*n: (j+1)*n, :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: (j+1)*n]
    else:
            X_test2016_ = X_test2016[j*n: , :]
            X_test2017_ = X_test2017[j*n: , :]
            results['ParcelId'] = sample_submission['ParcelId'].iloc[j*n: ]

    for month in [10, 11, 12]:
        X_test2016_[:, -1] = month_avgs[month - 1]
        assert X_test2016_.shape[1] == X_test2016.shape[1]
        y_pred = lgb_model.predict(X_test2016_)
        results['2016'+ str(month)] = y_pred
        
    X_test2017_[:, -1] = month_avgs[20]
    assert X_test2017_.shape[1] == X_test2017.shape[1]
    y_pred = lgb_model.predict(X_test2017_)
    results['201710'] = y_pred
    results['201711'] = y_pred
    results['201712'] = y_pred
    
    if j == 0:
        results_ = results.copy()
    else:
        results_ = pd.concat([results_, results], axis = 0)
    print('{}% completed'.format(round(100*(j+1)/folds)))
    
    
print('Saving predictions...')
results = results_[sample_submission.columns]
assert results.shape == sample_submission.shape
results.to_csv('submission_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d_%H%M%S')), index = False, float_format = '%.5f')
print('Done!')
