# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:57:02 2019

@author: fatih
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:52:08 2019

@author: fatih
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt



data=pd.read_csv("cereal_clean.csv")

#Random forest regression
X=data.drop('rating',axis=1)
y=data.rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

cereal_model = RandomForestRegressor(n_estimators=15, random_state=30)#30
cereal_model.fit(X_train, y_train)


predictions = cereal_model.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
#R2 analiz

from sklearn.metrics import r2_score
print( "r2 Score: ")
print (r2_score(y_test,predictions))

predictions_train=cereal_model.predict(X_train)
print( "train r2 Score: ")
print (r2_score(y_train,predictions_train))


#MSE analiz
from sklearn.metrics import mean_squared_error 
print( "mse Score: ")
print (mean_squared_error(y_test,predictions))

#MAE analiz
from sklearn.metrics import mean_absolute_error 
print( "msa Score: ")
print (mean_absolute_error(y_test,predictions))
