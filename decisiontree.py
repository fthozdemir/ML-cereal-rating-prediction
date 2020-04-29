# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:52:08 2019

@author: fatih
"""

import pandas as pd
import numpy as np
 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from matplotlib import pyplot as plt



data=pd.read_csv("cereal_clean.csv")

#Division of data into features and quality
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13:14].values


x_train, x_test, y_train, y_test = train_test_split(X, y ,test_size=0.2,random_state=42)




from sklearn.tree import DecisionTreeRegressor

regr=DecisionTreeRegressor(random_state=0)#29
regr.fit(x_train,y_train)

predictions =regr.predict(x_test)

print("Decision Tree Testing score:",regr.score(x_test,y_test))
plt.scatter(y_test,predictions ,color='red')
plt.xlabel("True Values")
plt.ylabel("Predictions")

from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 

export_graphviz(regr, out_file ='treelimited.dot', 
               feature_names =['mfr','calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups'],
                class_names = 'rating',
                rounded = True, proportion = False, precision = 2, filled = True)


#R2 analiz
from sklearn.metrics import r2_score
print( "test r2 Score: ")
print (r2_score(y_test,predictions))

predictions_train=regr.predict(x_train)

print( "train r2 Score: ")
print (r2_score(y_train,predictions_train))

#MSE analiz
from sklearn.metrics import mean_squared_error 
print( "mse Score: ")
print (mean_squared_error(y_test,predictions))


