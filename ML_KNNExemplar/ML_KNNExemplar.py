# -*- coding: utf-8 -*-
"""
Created on 17 Mar 2023
Name: ML_KNNExemplar.py
Purpose: Machine Learning Program based on K-Nearest Neighbour (KNN) Model
@author: Charles Umesi (charlesumesi)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from colorama import Fore

# Note: This code requires a virtual environment to function properly

# Introduction
print(Fore.YELLOW + "Welcome!\nThis program will enable your computer learn your data and make predictions from it using the 'K-Nearest Neighbour (KNN) Model'\n ")
print(Fore.WHITE + 'Using your data, it will tell you how well it can predict whether or not a datapoint belongs to your chosen variable depending on the number of neighbouring datapoints you ask to be considered in the calculations')
print('The neighbouring datapoints will belong to one or more variables')
print("This 'prediction by association' is done for each row in your 'test' data")
print('It is, however, up to you to interpret the results!\n ')
print('Also note that this model only works with numerical data')
ready = input('Do your required variables in your data only have numerical values? Y/N: ')
if ready == 'N':
  print('That means your data is unsuitable for this model') 
  print('You could try cleaning it using one of my data cleaners')
  quit('N')
elif ready == 'Y':
  # Prompt
  print(' \nMake sure your data document is in the form of a csv file and in the same directory as this program\n ')

  # Inputs
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

  # Put distances for all datapoints surrounding the test datapoint on the same scale
  scaler = StandardScaler()
  scaler.fit(df[features_tidylist])
  scaled_features = scaler.transform(df[features_tidylist])

  # Place transformed datapoints into a new dataframe
  df_scaled = pd.DataFrame(scaled_features,columns=features_tidylist)

  # Storing all features in scaled data into 'X'
  X = df_scaled

  # Storing target into 'y'
  y = df[target]

  # Tuple unpacking and splitting of data into 'training' and 'testing' sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

  # Choose method of selecting k number
  method = input("The number of datapoints surrounding the test datapoint you include in the calculations is called the 'k number'. Do you wish to choose k by the random or elbow method? (R/E): ")

  # Random method
  if method == 'R':
    k_number = int(input(' \nChoose your k number (must be a whole number): '))

    # Create an instance of LogisticRegression for your computer
    knn = KNeighborsClassifier(n_neighbors=k_number)

    # Train/fit your computer's lm on training data
    knn.fit(X_train,y_train)

    # Evaluate the performance of the training/fit by predicting off the test set of data

    # Generate predicted values
    pred = knn.predict(X_test)

    # Generate a tabulated result
    print(Fore.GREEN + " \nYour computer split your data into 'training' and 'testing' parts. It learnt from the training part and produced predictive results from the testing part (based on the KNN model)\n ")
    print(Fore.WHITE + "See table below ('0' means the test datapoint is predicted not to belong to the target variable and '1' is the reverse)\n",classification_report(y_test,pred))

  # Elbow method
  elif method == 'E':
    k_lower = int(input(' \nYou will need to choose a range; first enter the lower limit (must be a whole number): '))
    k_upper = int(input('Now enter the upper limit (must be a whole number): '))
    k_step = int(input('And choose the size of the step in which you wish to iterate from your chosen lower to upper limits (must be a whole number): '))

    # Iterate over chosen range, using KNN to determine error rate of each k number in range
    error_rate = []
    for i in range(k_lower, k_upper+1, k_step):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

    # Plot graph of k number in chosen range against error rate
    plt.figure(figsize=(10,6))
    plt.plot(range(k_lower, k_upper+1),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
    plt.title('Error Rate v k Number')
    plt.xlabel('k Number')
    plt.ylabel('Error Rate')
    plt.style.use('seaborn-darkgrid')
    plt.savefig('k_v_error1.png')

    print(Fore.LIGHTYELLOW_EX + 'A graph has been generated (k_v_error1.png) and is in the current directory')
    print(Fore.WHITE + 'In the graph, each k number in your chosen range is plotted againt the average (mean) number of times the predicted outcome did not equal the actual outcome across all rows in your test data for the k number in question\n ')
    print('Study the graph and decide which k number to use before continuing\n ')
    question = input('Are you ready to continue? (Y/N): ')
    if question == 'N':
      print('Okay, this program will terminate while you consider what to do next')
      quit('N')
    elif question == 'Y':
      k_number = int(input('Choose your k number (must be a whole number): '))

      # Create an instance of LogisticRegression for your computer
      knn = KNeighborsClassifier(n_neighbors=k_number)

      # Train/fit your computer's knn on training data
      knn.fit(X_train,y_train)

      # Evaluate the performance of the training/fit by predicting off the test set of data

      # Generate predicted values
      pred = knn.predict(X_test)

      # Generate a tabulated result
      print(Fore.GREEN + " \nYour computer split your data into 'training' and 'testing' parts. It learnt from the training part and produced predictive results from the testing part (based on the KNN model)\n ")
      print(Fore.WHITE + "See table below ('0' means the test datapoint is predicted not to belong to the target variable and '1' is the reverse)\n",classification_report(y_test,pred))

  else:
    quit()
      
