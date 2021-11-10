###############################################################################################################
# 
#Description : Case study about Diabetes detection.
# DataSet    : Diabetes
# Features   : Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age.
# Labels     : Outcome
# Date       : 25-05-2021
# Author nmae: HARISH VIJAY BAVNE.
#
###############################################################################################################

# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import countplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

################################################################################################

# Function of M.L.
def DiabetesPredictor():
    
    # Load data.
    Data = pd.read_csv("diabetes.csv")

    print("First five entries from dataset.")
    print(Data.head())

    print(f"Dimension of diabetes Data:{Data.shape}")
    print()
    
    line = ("*"*50)
    # Features and Targets
    X = Data.drop("Outcome",axis=1)
    Y = Data["Outcome"]


    # Data spliting
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.5)

    # Traing using RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x_train, y_train)
    print()
    print(line)
    print("Using RandomForestClassifier")
    print(f"Accuracy on training set: {rf.score(x_train, y_train)*100}")
    print(f"Accuracy on test set: {rf.score(x_test, y_test)*100}")
    print(line)
    print()
    
    # Traing using RandomForestClassifier
    rf1 = RandomForestClassifier(max_depth=6, n_estimators=100,random_state=0)
    rf1.fit(x_train, y_train)
    print()
    print(line)
    print("Using RandomForestClassifier")
    print("Accuracy after some tuning of parameter")
    print(f"Accuracy on training set: {rf1.score(x_train, y_train)*100}")
    print(f"Accuracy on test set: {rf1.score(x_test, y_test)*100}")

    #Traing using LogisticRegression

    LR = LogisticRegression(max_iter=1000)
    LR.fit(x_train, y_train)
    print()
    print(line)
    print("Using Logistic Regression")
    print(f"Accuracy on training set: {LR.score(x_train, y_train)*100}")
    print(f"Accuracy on test set: {LR.score(x_test, y_test)*100}")
    print(line)
    # Feature importance
    def plot_feature_importances_diabetes(model):
        plt.figure(figsize=(8,6))
        n_features = 8
        plt.barh(range(n_features), model.feature_importances_, align='center')
        diabetes_features = [x for i,x in enumerate(Data.columns) if i!=8]
        plt.yticks(np.arange(n_features), diabetes_features)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title("Featur Importance")
        plt.ylim(-1, n_features)
        plt.show()

    plot_feature_importances_diabetes(rf)

################################################################################################

# Main.
def main():
    print("Diabetes predictor application")
    DiabetesPredictor()

################################################################################################

# Main Starter
if __name__ == "__main__":
    main()

################################################################################################    