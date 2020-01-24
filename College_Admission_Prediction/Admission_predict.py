# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:17:45 2020

@author: Dell
"""

# Data Preprocessing-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Admission_Predict.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values



#Splitting the data set into train set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

#x_test = np.asarray([[int(input()),int(input())]])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting classifier to the Training set
# Create your classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
classifier.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 2, epochs = 300)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#x_test = np.asarray([[int(302),int(102),int(1),float(2),float(1.5),float(8),int(0)]])
x_test = np.asarray([[337 , 118 , 4 , 4.5 , 4.5 , 9.65 , 1]])
x_test = sc.transform(x_test)
x_pred = classifier.predict(x_test)

