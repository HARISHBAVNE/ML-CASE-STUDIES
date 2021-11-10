#################################################################################################
#
# Application : Iris case study with KNN algorithm.
# Author      : Harish Vijay Bavne.
#
#################################################################################################

# Required ppackages.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#######################################################################################################

# Function to perform Traing and Testing.
def MarvellousDesicion(data_train,data_test,target_test,target_train):
    dataset = load_iris()
    
    cobj = tree.DecisionTreeClassifier()
    
    cobj.fit(data_train,target_train)
    
    output = cobj.predict(data_test)
    
    Accuracy = accuracy_score(target_test,output)
    
    return Accuracy
    
#######################################################################################################
    
# Function to select algorithm.
def MarvellousKNN(data_train,data_test,target_test,target_train):
    
    # data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)
    
    cobj = KNeighborsClassifier()
    
    cobj.fit(data_train,target_train)
    
    output = cobj.predict(data_test)
    
    Accuracy = accuracy_score(target_test,output)
    
    return Accuracy

#######################################################################################################

# Main.
def main():
    
    # Load data.
    dataset = load_iris()
    
    # Features and Targets.
    data = dataset.data
    target = dataset.target
    
    # Data spliting for testing and training.
    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)
    ret = MarvellousKNN(data_train,data_test,target_test,target_train)
    print("Accuracy of Decision Tree Algorithm is",ret*100,"%")
    ret = MarvellousKNN(data_train,data_test,target_test,target_train)
    print("Accuracy of KNN Algorithm is",ret*100,"%")
    
#######################################################################################################

# Main Starter.
if __name__ == "__main__":
    main()
    
#######################################################################################################




