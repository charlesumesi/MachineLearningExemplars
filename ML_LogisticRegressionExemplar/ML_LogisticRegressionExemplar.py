# -*- coding: utf-8 -*-
"""
Created on 15 Mar 2023
Name: ML_LogisticRegressionExemplar.py
Purpose: Machine Learning Program based on Logistic Regression Model
@author: Charles Umesi (charlesumesi)
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from colorama import Fore

# Note: This code requires a virtual environment to function properly

# Introduction
print(Fore.YELLOW + "Welcome!\nThis program will enable your computer learn your data and predict things from it using the 'Logistic Regression Model'\n ")
print(Fore.WHITE + 'Note that this model will not explicitly tell you a prediction\nIt will instead, tell you how well it can predict your chosen variable from a list of variables in the data you provide')
print('It is up to you to interpret the results\n ')
print('Also note that this model only works with numerical data')
ready = input('Do your required variables in your data only have numerical values? Y/N: ')
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
  logmodel = LogisticRegression(solver='liblinear')

  # Train/fit your computer's lm on training data
  logmodel.fit(X_train, y_train)

  # Evaluate the performance of the training/fit by predicting off the test set of data

  # Generate predicted values
  predictions = logmodel.predict(X_test)

  # Generate a tabulated result
  print(Fore.GREEN + ' \nThe computer has now learnt your data and produced predictive results from it (based on the logistic regression model)\n ')
  print(Fore.WHITE + "See table below ('0' is a negative result for your chosen variable and '1' is a positive one)\n",classification_report(y_test,predictions))
