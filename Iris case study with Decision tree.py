#################################################################################################
#
# Application : Iris case study with DecisionTreeClassifier.
# Author      : Harish Vijay Bavne.
#
#################################################################################################

# Required packages
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
###############################################################################

# Main
def main():
    # Load dataset.
    iris = load_iris()

    print("Feature names of iris data sets")
    print(iris.feature_names)
    print("Target names of iris data sets")
    print(iris.target_names)

    #indices of removed elements
    test_index = [1,51,101]

    #traing data with removed elements
    train_target = np.delete(iris.target,test_index)
    train_data = np.delete(iris.data,test_index,axis = 0)
    #Testing data for testing on training data
    test_target = iris.target[test_index]
    test_data = iris.data[test_index]

    #Decesion tree classifier
    classifier = tree.DecisionTreeClassifier()

    #training data
    classifier = classifier.fit(train_data,train_target)

    print("Vales removed for testing")
    print(test_target)

    # testing
    print("Result of testing")
    print(classifier.predict(test_data))
    
#################################################################################

# Main Starter
if __name__ == "__main__":
    main()
    
#################################################################################
