#################################################################################################
#
# Application : Advertisement predictor using Regression.
# Author      : Harish Vijay Bavne.
#
#################################################################################################

# Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

###############################################################################################

# Function of M.L. using Multi LinearRegression Algorithm
def MultiLinearRegression()
    #Load Data
    Data = pd.read_csv("Advertising.csv")

    print("First five records of Data set")
    print(Data.head())

    #Features and label
    X = Data.iloc[:,1:4]
    Y = Data.iloc[:,-1:]

    #Selection of algorithm
    LR = LinearRegression()

    #Training on Data
    LR = LR.fit(X,Y)

    #Training accuracy 
    print("Training accuracy is:",(LR.score(X,Y))*100,"%")

    #Model coefficients for features i.e. "M"
    print("Value of M1,M2,M3=",LR.coef_)

    #Model Intercept i.e "C"
    print("Value of C:",LR.intercept_)


    #Testing
    Test = LR.predict([[230.1,37.8,69.2]])
    print("Predicted Sale is:"Test)
###############################################################################################

# Mian.
def main():
    
    print("Multiple Linear regression")
    MultiLinearRegression()

###############################################################################################

# Main Starter
if __name__ == "__main__":
    main()
    
###############################################################################################    