# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 11:27:59 2017

@author: evanm_000
"""

def MAE(y, ypred):
    
    import numpy as np
    
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)


def TestMonthMask(month, months):
    """
    
    This function creates a 0/1 flag as to whether the month included is used in the pub/private leaderboard test set
    
    """
    if month in months: return 1
    else: return 0 

def LinearCombPreds(lgb_preds, xgb_pred, weights, baseline_pred, test, test_dates, reg):
    
    import pandas as pd
    
    print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
    
    pred0 = weights['xgb_weight0'] * xgb_pred + weights['baseline_weight0'] * baseline_pred + weights['lgb_weight'] * lgb_preds
    
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
    
def TrainValidSplit(df, Months, monthSplits):

    import pandas as pd
    import numpy as np
    import datetime as dt
    
    trainValid = (df
                      .assign(trans_month = lambda x: x['transactiondate'].dt.month))
    
    trainTestMonths = (trainValid
                           .loc[trainValid['trans_month'].isin(Months)]
                           .sample(frac = monthSplits['trainTestMonths'])
                           .index
                           .values)
    
    trainElseMonths = (trainValid
                           .loc[~trainValid['trans_month'].isin(Months)]
                           .sample(frac = monthSplits['trainElseMonths'])
                           .index
                           .values
                           )
    
    trainIdx = np.append(trainTestMonths, trainElseMonths)
    
    trainValid = trainValid.drop('trans_month', axis = 1)
    
    train = trainValid.loc[trainIdx, :]
    valid = trainValid[np.logical_not(trainValid.index.isin(trainIdx))]
    
    return train, valid