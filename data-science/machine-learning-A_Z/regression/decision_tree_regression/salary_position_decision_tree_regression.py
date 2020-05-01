#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:03:05 2020

@author: i504180
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


data_set = pd.read_csv('Position_Salaries.csv')
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2].values

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression Position vs Salary')
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()