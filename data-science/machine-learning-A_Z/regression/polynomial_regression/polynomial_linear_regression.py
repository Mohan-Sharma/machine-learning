#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:39:47 2020

@author: i504180
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data_set = pd.read_csv('Position_Salaries.csv')
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values

#linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Linear Regression Position vs Salary')
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

linear_regressor.predict([[6.5]])

#polynomial regression
linear_poly_regressor = LinearRegression()
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
linear_poly_regressor.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_poly_regressor.predict(poly_features.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Linear Regression Position vs Salary')
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
    
linear_poly_regressor.predict(poly_features.fit_transform([[6.5]]))