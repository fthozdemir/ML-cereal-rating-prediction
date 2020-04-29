# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:52:08 2019

@author: fatih
"""

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso




data=pd.read_csv("cereal.csv")

##DATA HANDLING ===============================================================

data.drop('name',axis=1,inplace=True) # droped because unique data
#data.drop('type',axis=1,inplace=True) # droped because mostly same var.

##there are some negative valuse in data
data.replace(to_replace =-1,  
                            value =np.NaN, inplace=True) 

data.fillna(0, inplace=True)

#Label Encodincg to catogorical data*******************************************
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
data['mfr'] = labelencoder.fit_transform(data['mfr'])
data['type'] = labelencoder.fit_transform(data['type'])

data.drop('type',axis=1,inplace=True)



#Label Encodincg catogorical data END *****************************************


#FEATURE SELECTION START-------------------------------------------------------


X = data.drop("rating",1)
y = data["rating"]

#heatmap
plt.figure(figsize=(16,13))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()




##DATA HANDLING END ===========================================================


data.to_csv('cereal_clean.csv',index=False)


#feature importances grafical show----------------
plt.figure(figsize=(16, 9))

ranking = rf.feature_importances_
features = np.argsort(ranking)[::-1][:16]
columns = X.columns

plt.title("Feature importances based on Random Forest Regressor", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()

