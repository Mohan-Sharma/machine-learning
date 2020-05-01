#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:12:31 2020

@author: i504180
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sma


data_set = pd.read_csv("50_Startups.csv")
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4]
columnTrans = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(columnTrans.fit_transform(X), dtype=np.float)

X = X[:, 1:]

X = np.append(arr = np.ones(shape = (50, 1), dtype=np.float), values = X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]

def backward_elimination(matrix_of_features, independent_arr, significance_level) :
    regressor = sma.OLS(independent_arr, matrix_of_features).fit()
    no_of_columns = len(matrix_of_features[0])
    p_values = regressor.pvalues
    max_p_value = max(p_values)
    if max_p_value < significance_level :
        return matrix_of_features;
    for i in range(0, no_of_columns) :
        if max_p_value == p_values[i].astype(dtype = np.float) :
            matrix_of_features = np.delete(matrix_of_features, i, 1)
  
    return backward_elimination(matrix_of_features, independent_arr, significance_level)


def adj_r_backward_elimination(matrix_of_features, independent_arr, significance_level):
    regressor = sma.OLS(independent_arr, matrix_of_features).fit()
    no_of_columns = len(matrix_of_features[0])
    p_values = regressor.pvalues
    adjR_before = regressor.rsquared_adj.astype(float)
    max_p_value = max(p_values)
    if max_p_value < significance_level:
        return matrix_of_features;
    for i in range(0, no_of_columns):
        if max_p_value == p_values[i].astype(dtype = np.float) :
            new_matrix_of_features = np.delete(matrix_of_features, i, 1)
            tmp_regressor = sma.OLS(y, new_matrix_of_features).fit()
            adjR_after = tmp_regressor.rsquared_adj.astype(float)
            if adjR_before >= adjR_after:
                return matrix_of_features
            else:
                matrix_of_features = new_matrix_of_features
         
    return adj_r_backward_elimination(matrix_of_features, independent_arr, significance_level)


X_opt = adj_r_backward_elimination(X_opt, y, 0.05)