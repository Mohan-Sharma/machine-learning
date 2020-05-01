#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:59:12 2020

@author: i504180
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

data_set = pd.read_csv("Social_Network_Ads.csv")
X = data_set.iloc[:, 2:4].values
y = data_set.iloc[:, 4].values

scalar = StandardScaler();
X_scaled = scalar.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


matrix = confusion_matrix(y_test, y_pred)
print(matrix)

X_set, y_set = X_train, y_train
h,v = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(h, v, 
             classifier.predict(np.array([h.ravel(), v.ravel()]).T).reshape(h.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(h.min(), h.max())
plt.ylim(v.min(), v.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


X_set, y_set = X_test, y_test
h,v = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(h, v, 
             classifier.predict(np.array([h.ravel(), v.ravel()]).T).reshape(h.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(h.min(), h.max())
plt.ylim(v.min(), v.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()