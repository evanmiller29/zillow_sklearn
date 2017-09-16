# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:46:47 2017

@author: Evan
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

#==============================================================================
# Scalar multiplication transformer
#==============================================================================

class FeatureMultiplier(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, X, *_):
        return X * self.factor

    def fit(self, *_):
        return self

#==============================================================================
# Testing
#
# fm = FeatureMultiplier(2)
# print(fm.transform(4))
#==============================================================================

#==============================================================================
# log + 1 transformer
#==============================================================================

class FeatureCustomLogger(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, X, *_):
        return np.log1p(X + 1)

    def fit(self, *_):
        return self

#==============================================================================
# Testing
# 
# print(np.log1p(2 + 1)) # 1.0986122886681098
# lg = FeatureCustomLogger(2)    
# print(lg.transform(2))
#==============================================================================

#==============================================================================
# Column selector transformer
#==============================================================================

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, subset):
        self.subset = subset

    def transform(self, X, *_):
        return X.loc[:, self.subset]

    def fit(self, *_):
        return self

#==============================================================================
# Testing
# 
# x_valid.head()
# x = x_train.loc[:, x_train.dtypes == 'object']
# print(x.head())
# 
# objFeat = ColumnExtractor('object')
# objFeat.transform(x_valid).head()
#==============================================================================
