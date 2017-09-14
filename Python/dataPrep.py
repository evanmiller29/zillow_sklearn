# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 21:29:10 2017

@author: evanm_000
"""

def LBMdata(propDF, trainDF, featEngineer):
    
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    
    from featEngineering import sqFtFeat, ExtractTimeFeats

    print( "\nProcessing data for LightGBM ..." )
    for c, dtype in zip(propDF.columns, propDF.dtypes):	
        if dtype == np.float64:		
            propDF[c] = propDF[c].astype(np.float32)
    
    df_train = trainDF.merge(propDF, how='left', on='parcelid')
    df_train.fillna(df_train.median(),inplace = True)
    
    #==============================================================================
    # Doing some feature engineering
    #==============================================================================
    
    if featEngineer == True:
        
        df_train = sqFtFeat(df_train)
                
    x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                             'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
    
    y_train = df_train['logerror'].values
    print(x_train.shape, y_train.shape)
    
    train_columns = x_train.columns
    
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)
    
    x_train = x_train.values.astype(np.float32, copy=False)
    d_train = lgb.Dataset(x_train, label=y_train)
    
    return x_train, d_train, train_columns

def XGBdata(propDF, trainDF, FeatEnginnering, lowerBound, upperBound):
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from featEngineering import sqFtFeat
    
    print( "\nProcessing data for XGBoost ...")
    for c in propDF.columns:
        propDF[c]=propDF[c].fillna(-1)
        if propDF[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(propDF[c].values))
            propDF[c] = lbl.transform(list(propDF[c].values))
    
    train_df = trainDF.merge(propDF, how='left', on='parcelid')
    x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    
    x_test = propDF.drop(['parcelid'], axis=1)    
            
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
    
    # drop out ouliers
    train_df=train_df[ train_df.logerror > lowerBound ]
    train_df=train_df[ train_df.logerror < upperBound ]
    
    x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
    
    if FeatEnginnering == True:
    
        x_test = sqFtFeat(x_test)
        x_train = sqFtFeat(x_train)
    
    y_train = train_df["logerror"].values.astype(np.float32)
    y_mean = np.mean(y_train)
    
    print('After removing outliers:')     
    print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
    
    return x_train, x_test, y_train, y_mean

def DataFrameDeets(df, dfName):

    print("The %s dataset has %s columns and %s rows" % (dfName, df.shape[1], df.shape[0]))
    
def ConvertCats(df):
    
    import pandas as pd
    
    df = df.copy()
    
    for c in df.dtypes[df.dtypes == "category"].index.values:
    
        print(c + ' convered with OHE..')
    
        cat = pd.get_dummies(df[c])
        df = df.drop(c, axis = 1)
        
        df= pd.concat([df, cat], axis = 1)   
    
    return(df)