#################################################################################################
#
# Description: Case study about play prediction on the basis of Wether and Temperature forecast.
# Classifier : KNeighborsClassifier.
# DataSet :    MarvellousInfosystems_PlayPredictor.csv.
# Features :   Wether,Temperature.
# Labels :     Play.
# Functions :  Training ,Testing,Accuracy.
# Date:        07-06-2021
# Author nmae: HARISH VIJAY BAVNE.

#################################################################################################

# Required packages.
import pandas as pd
from pandas.core import reshape
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#################################################################################################

#Function for Training and Testing.
def PlayPredictor(data):
    data.drop('Unnamed: 0',axis=1,inplace=True)
    print(data.head())
    print(data.columns)

    #Separating Features and Targets
    X = data.iloc[:, [0, 1]].values
    Y = data.iloc[:, 2].values
   
    clf = KNeighborsClassifier(n_neighbors=3)  # Algorithm.
    clf.fit(X,Y)  # Training of Data.

    result = clf.predict([[0,2]])
    if (result == 0):
        print("No")
    else:
        print("Yes")

#################################################################################################

#Function of Accuracy of algorithm.
def CheckAccuracy(data):
    print("Accuracy of Algorithm")
    X = data.iloc[:, [0, 1]].values
    Y = data.iloc[:, 2].values
    X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.5)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_Train,Y_Train)

    print("Training Accuracy:",clf.score(X_Train,Y_Train)*100,"%")
    print("Testing Accuracy:",clf.score (X_Test, Y_Test)*100,"%")

#################################################################################################

# Entry Function.
def main():
    print("Marvellous Infosystems Play Predictor")
    data = pd.read_csv("PlayPredictor.csv")  # Loading data from csv file.
    le = LabelEncoder()

    # Label Encoding
    data["Play"] = le.fit_transform(data["Play"])
    data["Wether"] = le.fit_transform(data["Wether"])
    data["Temperature"] = le.fit_transform(data["Temperature"])

    PlayPredictor(data)
    CheckAccuracy(data)

#################################################################################################
    
# Main Starter.
if __name__ == "__main__":
    main()

#################################################################################################
