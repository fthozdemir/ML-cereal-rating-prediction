# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:34:42 2019

@author: fatih
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
data = pd.read_csv('cereal_clean.csv')

x = data.iloc[:,0:-1]
y = data.iloc[:,-1:]
X = x.values
Y = y.values


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(x_olcekli, y_olcekli, test_size=0.2, random_state=25)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'sigmoid')
svr_reg.fit(X_train,y_train)

predictions =svr_reg.predict(X_test)

plt.scatter(y_test,predictions,color='blue')
#plt.plot(y_test,svr_reg.predict(X_test),color='blue')


from sklearn.metrics import r2_score
print( "test r2 Score: ")
print (r2_score(y_test,predictions))

from sklearn.metrics import r2_score
print( "test r2 Score: ")
print (r2_score(y_test,predictions))

predictions_train=svr_reg.predict(X_train)
print( "train r2 Score: ")
print (r2_score(y_train,predictions_train))


#MSE analiz
from sklearn.metrics import mean_squared_error 
print( "mse Score: ")
print (mean_squared_error(y_test,predictions))

