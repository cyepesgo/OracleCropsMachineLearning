#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:47:03 2017

@author: caroyepes
"""

#Multiple Linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ph_Prediction.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

                
                
 #tokenizar las variables -encoding Categorical data- Toma las variables texto y las convierte en texto
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#El 3 es el indice de la columna
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
#El hot encoder es para resolver el problema de asignarle valor a las variables
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()


#Avoiding the Dummy Variable Trap
#Remover la primera columna de las dummy variables - no hay que hacerlo la  libreria se encarga de esto 
X = X[:, 1:]



                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit el multiple linear regressor al train set
regressor.fit(X_train, y_train)


#Predicting the test set results
y_pred = regressor.predict(X_test)

################################Hasta Aca Regresion Lineal Multiple############################

#building the optimal model using backward elimination 
import statsmodels.formula.api as sm
#agregar una columna de unos para representar la constante b0
#(50 filas, 1 columna)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X,  axis = 1)
#Vamor a crear una variables que contiene la  matriz X _opt para guardar las variables significativas del modelo
#X[:, [0,1,2,3,4,5]] para seleccionar todas las filas y las columnas
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()


X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()

# no lo he ejecutado  porque el p value es un cercano al significance level
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()


#Regresion Multiple con el modelo ajustado

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_optTrain, X_optTest = train_test_split(X_opt, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor_opt = LinearRegression()
#fit el multiple linear regressor al train set
regressor_opt.fit(X_optTrain, y_train)


#Predicting the test set results
y_predOpt = regressor_opt.predict(X_optTest)
