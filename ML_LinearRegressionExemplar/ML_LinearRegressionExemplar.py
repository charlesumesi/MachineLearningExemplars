# -*- coding: utf-8 -*-
"""
Created on 7 Mar 2023
Name: ML_LinearRegressionExemplar.py
Purpose: Machine Learning Program based on Linear Regression Model
@author: Charles Umesi (charlesumesi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from colorama import Fore

# Note: This code requires a virtual environment to function properly

# Introduction
print(Fore.YELLOW + "Welcome!\nThis program will enable your computer learn your data and predict things from it using the 'Linear Regression Model'\n ")
print(Fore.WHITE + 'Note that this model will not explicitly tell you a prediction\nIt will instead, give you results from your data that are potentially predictive')
print('It is up to you to interpret the results\n ')
print('Also note that this model only works with CONTINOUS numerical data')
ready = input('Are your required variables in your data continous numerical values? Y/N: ')
if ready == 'N':
  print('That means your data is unsuitable for this model') 
  print('You could try cleaning it using one of my data cleaners')
  quit('N')
elif ready == 'Y' or 'y' and ready != 'n':
  # Prompt
  print(' \nMake sure your data document is in the form of a csv file and in the same directory as this program\n ')

  # Inputs
  question = input('What are you wanting to predict from your data?: ')
  file = input('Enter the name of your file (with csv extension): ')
  target = input("Enter the name of your 'target' (dependent variable) exactly as it appears in your csv file: ")
  features_numbers = int(input("How many 'features' (independent variables) are you including?: "))
  features = 'Enter the name of one feature exactly as it appears in your csv file: '
  features_list = [list(input(features)) for _ in [0]*features_numbers]

  # Collect requested data from csv file
  df = pd.read_csv(file)

  # Tidy the list of features by converting to string and reconverting back to a list
  features_tidylist = []
  for i in features_list:
    j = ''.join(i)
    features_tidylist.append(j)
    
  # Storing all features into 'X'
  X = df[features_tidylist]

  # Storing target into 'y'
  y = df[target]

  # Tuple unpacking and splitting of data into 'training' and 'testing' sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

  # Create an instance of LinearRegression for your computer
  lm = LinearRegression()

  # Train/fit your computer's lm on training data
  lm.fit(X_train, y_train)

  # Evaluate the performance of the training/fit by predicting off the test set of data

  # Generate predicted values
  predictions = lm.predict(X_test)

  # Generate a scatterplot of the real feature values vs the predicted feature values
  plt.xlabel('Actual '+ target)
  plt.ylabel('Predicted ' + target)
  plt.scatter(y_test,predictions)
  plt.savefig('scatterplot1.png')

  # Generate a histogram of the residuals
  sns.displot((y_test-predictions),kde=True,bins=50)
  plt.savefig('displot1.png')

  print(Fore.GREEN + ' \nThe computer has now learnt your data and produced predictive results from it (based on the linear regression model)\nGenerated graphs are in the current directory\n ')
  print(Fore.WHITE + 'The mean absolute error\nMAE:',metrics.mean_absolute_error(y_test,predictions))
  print('The mean squared error\nMSE:',metrics.mean_squared_error(y_test,predictions))
  print('The root mean squared error\nRMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
  print('The explained variance score\nR^2:',metrics.explained_variance_score(y_test,predictions))
  print(' ')
  print('The coefficient(s) is/are:')
  print(pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient']))

  # Generate a pairplot for a visual display of the correlations between variables (this step can interfere with the generation of the scatterplot so is placed several steps further down)
  sns.pairplot(df)
  plt.savefig('pairplot1.png')
