#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:25:28 2017

@author: caroyepes
"""

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
import csv, operator

# Importing the dataset
dataset = pd.read_csv('Pred_CE.csv')
#print (dataset)
#datadep = dataset.drop(labels=["dir -wind", "speed wind", "alerta","EC alert", "PH alert", "precipitacion"],  axis=1)
X = dataset.iloc[:, :-1].values                          
y = dataset.iloc[:, 5].values
#print (X)
                
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:4])
X[:, 0:4] = imputer.transform(X[:, 0:4])

 #tokenizar las variables -encoding Categorical data- Toma las variables texto y las convierte en texto
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#El 3 es el indice de la columna
#X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
#El hot encoder es para resolver el problema de asignarle valor a las variables
#onehotencoder = OneHotEncoder(categorical_features = [5])
#X = onehotencoder.fit_transform(X).toarray()


#Avoiding the Dummy Variable Trap
#Remover la primera columna de las dummy variables - no hay que hacerlo la  libreria se encarga de esto 
X = X[:, 1:]



                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit el multiple linear regressor al train set
regressor.fit(X_train, y_train)


#Predicting the test set results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination 
import statsmodels.formula.api as sm
#agregar una columna de unos para representar la constante b0
#(50 filas, 1 columna)
X = np.append(arr = np.ones((98, 1)).astype(int), values = X,  axis = 1)
#Vamor a crear una variables que contiene la  matriz X _opt para guardar las variables significativas del modelo
#X[:, [0,1,2,3,4,5]] para seleccionar todas las filas y las columnas
X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Va a retornar los valores que se necesitan para saber si el modelo es bueno. Como los p-values y R cuadrado
regressor_OLS.summary()

#Mostrar los Valores
print ('La prediccion es:',y_pred)
prediccion = pd.DataFrame(y_pred)
prediccion.to_csv('ce_pred.csv')