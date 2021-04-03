# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:09:57 2021

@author: Alec
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Ice Cream Data Regression
#Import Data
IceCream = pd.read_csv('IceCreamData.csv')

#Data Analysis
print(IceCream.head(10))
print(IceCream.tail(10))
print(IceCream)
print(IceCream.describe())
print(IceCream.info())

#Data Visualization
sns.jointplot(x= 'Temperature', y= 'Revenue', data= IceCream)
sns.jointplot(x= 'Revenue', y= 'Temperature', data= IceCream)

sns.pairplot(IceCream)
sns.lmplot(x= 'Temperature', y = 'Revenue', data= IceCream)


#Splitting Data into Train/Test Datasets
X = IceCream[['Temperature']]
Y = IceCream['Revenue']

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)

#Train the Model
regressor = LinearRegression(fit_intercept=(True)) #Obtain m and b from y = mx + b, Assuming there is a y-intercept if fit_intercept == True
regressor.fit(X_train, Y_train)
print('Linear Model Coefficient (m)', regressor.coef_)
print('Linear Model Coefficient (b)', regressor.intercept_)

Y_predict = regressor.predict(X_test)

#Visualize Model Predictions
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.ylabel('Revenue [Dollars]')
plt.xlabel('Temperature [C]')
plt.title('Revenue vs TemperatureTrain')

print('NewLine')
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test, regressor.predict(X_test), color='gray')
plt.ylabel('Revenue [Dollars]')
plt.xlabel('Temperature [C]')
plt.title('Revenue vs TemperatureTest')

#Manual Prediction
T = [[40]]
revenue_predict = regressor.predict(T)

#Fuel Economy Regression
#Import Data
FuelEconomy = pd.read_csv('FuelEconomy.csv')

print(FuelEconomy.head(10))
print(FuelEconomy.tail(10))
print(FuelEconomy)
print(FuelEconomy.describe())
print(FuelEconomy.info())

sns.jointplot(x='Horse Power', y='Fuel Economy (MPG)', data= FuelEconomy)
sns.pairplot(FuelEconomy)
sns.lmplot(x='Horse Power', y='Fuel Economy (MPG)', data= FuelEconomy)

I = FuelEconomy[['Horse Power']]
J = FuelEconomy[['Fuel Economy (MPG)']]
print(I.shape)
print(J.shape)

I_train, I_test, J_train, J_test = train_test_split(I, J, test_size=0.2, random_state= 42)


fuel_regressor = LinearRegression(fit_intercept=(True))
fuel_regressor.fit(I_train,J_train)
print('Linear Model Coefficient (m)', fuel_regressor.coef_)
print('Linear Model Coefficient (b)', fuel_regressor.intercept_)

J_predict = fuel_regressor.predict(I_test)

plt.scatter(I_train, J_train, color='red')
plt.plot(I_train, fuel_regressor.predict(I_train), color='green')
plt.ylabel('Horse Power')
plt.xlabel('Fuel Economy (MPG)')
plt.title('Horse Power vs Fuel Economy (MPG)')

print('NewLine')
plt.scatter(I_test, J_test, color='blue')
plt.plot(I_test, fuel_regressor.predict(J_test), color='gray')
plt.ylabel('Horse Power')
plt.xlabel('Fuel Economy (MPG)')
plt.title('Horse Power vs Fuel Economy (MPG)')
