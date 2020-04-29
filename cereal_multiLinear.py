# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:33:07 2019

@author: RamisYuksel
"""

import pandas
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
x = pandas.read_csv("cereal_clean.csv")

creal = pandas.DataFrame(x)

creal.columns
creal.shape

creal = creal.dropna(axis=0)

creal.corr()["rating"]

creal.corr()

# Get all the columns from the dataframe.
columns = creal.columns.tolist()
#print(columns)

#remove the columns we don't want.
columns = [c for c in columns if c not in ['name', 'mfr', 'type','calories', 'protein', 
                                           'fat', 'sodium', 'fiber', 'carbo', 'potass',
                                           'vitamins', 'shelf', 'weight', 'cups','rating']]

#selecting the predictors
#features = columns[9]
#print (features)

# Store the variable we'll be predicting on.
target = "rating"

# Import a convenience function to split the sets.

# Generate the training set.  Set random_state to be able to replicate results.
train = creal.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = creal.loc[~creal.index.isin(train.index)]
# Print the shapes of both sets.
print("train.shape: ")
print(train.shape)
print("test.shape: ")
print(test.shape)

# Import the linearregression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])
print('Variance score: %.2f' % model.score(train[columns], train[target]))

# The coefficients
print('Coefficients: \n', model.coef_)

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test[target])

# Get all the columns again from the dataframe.
columns = creal.columns.tolist()
print(columns)

#remove the columns we don't want.
columns_iter1 = [c for c in columns if c not in ['name', 'mfr', 
                                           'type','calories', 'protein', 
                                            'sodium', 'fiber', 'carbo', 
                                           'potass', 'vitamins', 'shelf',
                                           'weight', 'cups','rating']]
print(columns_iter1)

# Import the linearregression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model_iter1 = LinearRegression()
# Fit the model to the training data.
model_iter1.fit(train[columns_iter1], train[target])
print('Variance score: %.2f' % model_iter1.score(train[columns_iter1], train[target]))

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions= model_iter1.predict(test[columns_iter1])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test[target])

# The coefficients
print('Coefficients: \n', model_iter1.coef_)

# Get all the columns again from the dataframe.
columns = creal.columns.tolist()
print(columns)

#remove the columns we don't want.
columns_iter2 = [c for c in columns if c not in ['name', 'mfr', 
                                           'type','calories', 'protein', 
                                            'sodium', 'carbo', 
                                           'potass', 'vitamins', 'shelf',
                                           'weight', 'cups','rating']]
print(columns_iter2)

# Import the linearregression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model_iter2 = LinearRegression()
# Fit the model to the training data.
model_iter2.fit(train[columns_iter2], train[target])
print('Variance score: %.2f' % model_iter2.score(train[columns_iter2], train[target]))

# The coefficients
print('Coefficients: \n', model_iter2.coef_)


fig = plt.figure() 

ax1 = fig.add_subplot(111)

ax1.scatter(x['fat'], x['rating'], color = 'red') 

ax1.scatter(x['fiber'], x['rating'], color = 'blue') 


ax1.scatter(x['sugars'], x['rating'], color = 'green') 
plt.legend(loc='upper left') 
plt.show()

