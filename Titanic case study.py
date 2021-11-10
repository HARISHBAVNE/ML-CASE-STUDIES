#################################################################################################
#
# Application : Titanic survival Case study using Logistic regression.
# Date        : 23/05/2021.
# Author      : Harish Vijay Bavne.
#
#################################################################################################

# Required packages
import pandas as pd
from matplotlib.pyplot import figure, show
from seaborn import countplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#######################################################################################################

# Machine Learning function. 
def TitanicCaseStudy():
    # Step1: Load data
    Data = pd.read_csv("Titanic.csv")
    print("First Five entries from loaded Dataset")
    print(Data.head())

    print("Number of passangers are:", len(Data))

    # Step2: Analyze data.
    print("Visualization : Survived and nonsurvived passangers")
    figure()
    target = "Survived"
    countplot(data=Data, x=target).set_title("Survived and nonsurvived passangers")
    show()

    print("Visualization : Survived and nonsurvived passangers based on Gender.")
    figure()
    target = "Survived"
    countplot(data=Data, x=target, hue="Sex").set_title("Survived and nonsurvived passangers based on Gender")
    show()

    print("Visualization : Survived and nonsurvived passangers based on the passanger class.")
    figure()
    target = "Survived"
    countplot(data=Data, x=target, hue="Pclass").set_title(
        "Survived and nonsurvived passangers based on the passanger class")
    show()

    print("Visualization :passangers based on the age.")
    figure()
    Data["Age"].plot.hist().set_title("passangers based on the age")
    show()

    # Step3: Data Cleaning.
    Data.drop("zero", axis=1, inplace=True)  # Removing zero column.

    print("First five entries after removing zero column")
    print(Data.head())

    # Data Wrangling
    Sex = pd.get_dummies(Data["Sex"], drop_first=True)
    print("Values of sex column after removing one field")
    print(Sex.head(5))

    Pclass = pd.get_dummies(Data["Pclass"], drop_first=True)
    print("Values of Pclass column after removing one field")
    print(Pclass.head())

    print("Data set after concat updated Pclas and Sex")
    Data = pd.concat([Data, Sex, Pclass], axis=1)
    print(Data.head())

    print("Data after removing irrelevant columns")
    Data.drop(["Sex", "sibsp", "Parch", "Pclass", "Embarked"], inplace=True, axis=1)
    print(Data.head())

    # Step4: Training
    X = Data.drop("Survived", axis=1)
    Y = Data["Survived"]

    # Data spliting for training and testing.
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.5)

    # Selection of algorithm
    Log = LogisticRegression(max_iter=1000)

    # Training
    Log = Log.fit(X_Train, Y_Train)

    # Testing
    Result = Log.predict(X_Test)

    # Confusion matrix
    print("Confusion matrix of logistic regression is")
    print(confusion_matrix(Y_Test, Result))

    print("Accuracy of Logistic regression is")
    print(accuracy_score(Y_Test, Result)*100,"%")

#######################################################################################################

# Main
def main():
    print("Supervised machine learning")
    print("*****Logistic Regression on Titanic Dataset*****")

    TitanicCaseStudy()

#######################################################################################################

# Main Starter
if __name__ == "__main__":
    main()

#######################################################################################################