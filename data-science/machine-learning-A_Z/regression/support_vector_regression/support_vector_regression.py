#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:34:28 2020

@author: i504180
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data_set = pd.read_csv('Position_Salaries.csv')
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values

scalar_X = StandardScaler()
scalar_y = StandardScaler()
X = scalar_X.fit_transform(X)
y = scalar_y.fit_transform(y.reshape(-1, 1))

regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
y_pred = scalar_y.inverse_transform(regressor.predict(np.array([[6.5]])))

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('SVR Regression Position vs Salary')
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
