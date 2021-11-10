############################################################################################
#
# About     : Implementing Random Forest Classifier to predict the breast cancer.
# Functions : 1.Main, 2.read_data, 3. get_headers, 4.split_datset, 
#             5.handel_missing_values, 6.RandomForestClassifier, and 7. dataset_statistics.
# Date      : 14.06.2021 .
# Author    : HARISH VIJAY BAVNE .
#
#############################################################################################

# Required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#################################################################

# File Path
INPUT_PATH = "breast-cancer-wisconsin.csv"

# Headers
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]


#################################################################

# Function to Read data of file.
def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return Dataset:
    """
    data = pd.read_csv(path)
    return data

#################################################################

# Function to Get headers of Dataset.
def get_headers(dataset):
    """
    dataset headers
    :param dataset:
    :return columns header names:
    """
    return dataset.columns.values

#################################################################

# Function to split dataset for training and testing
def split_datset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return splitted features and target:
    """
    # Split dataset into train and test dataset
    x_train, x_test, y_train, y_test = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return x_train, x_test, y_train, y_test

################################################################## Function to filter missing values.
def handel_missing_values(data, header, missing_label):
    """
    Filter missing values from the dataset.
    :param data:
    :param header:
    :param missing_label:
    :return modified dataset:
    """
    dataset =data[data[header] != missing_label]
    return dataset

##################################################################

# Function to Selction of algorithm and training of dataset
def random_forest_classifier():
    """
    Algorithm selection
    :return classifier object:
    """
    clf = RandomForestClassifier()
    return clf

##################################################################

# function to describe dataset
def dataset_statistics(dataset):
    """
    Get the Description of dataset.
    :param dataset:
    :return:
    """
    print(dataset.describe())


##################################################################

def main():
    
    # Load dataset
    dataset = read_data(INPUT_PATH)
    print(dataset.head())
    # Get basic statistics of dataset
    dataset_statistics(dataset)

    # Filter missing values from dataset
    dataset = handel_missing_values(dataset,HEADERS[6],'?')

    # Split dataset for training and testing purpose
    x_train, x_test, y_train, y_test = split_datset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

    # create classifier object
    RFclf = random_forest_classifier()

    # Training and testing on dataset
    RFclf = RFclf.fit(x_train,y_train)
    Result = RFclf.predict(x_test)

    # Accuracy of Training and testing
    Accuracy_Training = RFclf.score(x_train,y_train)
    Accuracy_Testing = RFclf.score(x_test,y_test)
    print("Training Accuracy:", Accuracy_Training*100,"%")
    print("Testing Accuracy:",Accuracy_Testing*100,"%")

    # Confusion matrix
    print("confusion matrix")
    print(confusion_matrix(y_test,Result))

if __name__ == "__main__":
    print()
    print("Breast Cancer Case study")
    print("Implementing Random Forest Classifier to predict the breast cancer.")
    print()
    main()
