###############################################################################################################
# 
#Description: Case study about wine prediction.
# Classifier : KNeighborsClassifier.
# DataSet :    WinePredictor.csv.
# Features :   Alcohol,Malic acid,Ash,Alcalinity of ash,Magnesium,Total phenols,Flavanoids,Nonflavanoid phenols,
#             Proanthocyanins,Color intensity,Hue,OD280/OD315 of diluted wines,Proline.
# Labels :     Class1,Class2,Class3.
# Functions :  Training ,Testing,Accuracy.
# Date:        07-06-2021
# Author nmae: HARISH VIJAY BAVNE.
#
###############################################################################################################


# Required packages.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###############################################################################################################

#Function for Wine predictor model using KNN.
def WinePredictor():

    #Load Data.
    Data = pd.read_csv("WinePredictor.csv")
    print(Data.head())

    #Features.
    X = Data.drop("Class",axis = 1)
    print("First Five Features records")
    print(X.head())

    #Targets.
    Y = Data["Class"]

    #Algorithm selection
    clf = KNeighborsClassifier(n_neighbors = 5)
    print("Algorithm for M.L.",clf)

    #Split Data Set into Training  and Testing set.
    X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 0.3)# For Testing  30% data  and for  Training 70%

    #Train the model using Training Data
    clf.fit(X_Train,Y_Train)

    print("Training accuracy is:",(clf.score(X_Train,Y_Train))*100,"%")
    #Test the model using testing Data
    result = clf.predict(X_Test)

    Accuracy = accuracy_score(result,Y_Test)
    print("Accuracy of Testing is:",Accuracy*100,"%")

###############################################################################################################    

# Entry function.
def main():
    print("Wine predictor apllication using KNN M.L. Algorithm")
    WinePredictor()

###############################################################################################################

#Main Starter.
if __name__ == "__main__":
    main()

###############################################################################################################



