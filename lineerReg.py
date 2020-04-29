# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:33:26 2019

@author: fatih
"""

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np


import matplotlib
import matplotlib.pyplot as plt



# veri yukleme
data = pd.read_csv('cereal_clean.csv')

x = data.iloc[:,0:-1]
y = data.iloc[:,-1:]
X = x.values
Y = y.values


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=0)


lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print( "Score: ", model.score(X_test, y_test))


plt.scatter(y_test, predictions)
plt.xlabel("True Values",size=10)
plt.ylabel("Predictions")

#R2 analiz
from sklearn.metrics import r2_score
print( "r2 Score: ")
print (r2_score(y_test,predictions))

#MSE analiz
from sklearn.metrics import mean_squared_error 
print( "mse Score: ")
print (mean_squared_error(y_test,predictions))
