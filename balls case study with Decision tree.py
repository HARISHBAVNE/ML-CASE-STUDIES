#################################################################################################
#
# Application : Ball case study with DecisionTreeClassifier.
# Author      : Harish Vijay Bavne.
#
#################################################################################################

# Required packages.
from sklearn import tree

###############################################################################################################################

# Function to perform M.L.
def BallFinder(weight,surface):

    BallsFeatures = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    
    labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]
    
    # Algorithm selection.
    clf = tree.DecisionTreeClassifier()
    
    # Training.
    clf = clf.fit(BallsFeatures,labels)
    
    # Testing.
    result = clf.predict([[weight,surface]])
    
    if result == 1:
        print("Objects looks like Tennis Ball")
    elif result == 2:
        print("Objects looks like Cricket Ball")

#######################################################################################################################################

# Main.
def main():
    weight = int(input("Enter weight of Ball:"))
    surface = input("Enter a ball surface:")
    
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Wrong input")
        
    BallFinder(weight,surface)
    
#######################################################################################################################################    

# Main Starter.
if __name__ =="__main__":
    main()
    
#######################################################################################################################################    