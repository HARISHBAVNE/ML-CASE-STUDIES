############################################################################
#
# Application : Brain size predicton . 
# Date        : 11.06.2021
# Author      : HARISH VIJAY BAVNE.
#
############################################################################


# Required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################################################################

# M.L. using user defined Linear regression.
def HeadBrain():
    Data = pd.read_csv("HeadBrain.csv")
    print("First Five records of data set")
    print(Data.head())

    #Data Cleaning
    Data = Data.drop(["Gender"],axis = 1)
    print(Data.head())

    X = Data["Head Size(cm^3)"].values
    Y = Data["Brain Weight(grams)"].values

    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    #Regression line Equation is y = mx + c
    # m = slope of line
    # c = intercept of Y

    n = len(X)

    Numerator = 0
    Denomenator = 0

    for i in range(n):
        Numerator += (X[i]-X_mean)*(Y[i]-Y_mean)
        Denomenator += (X[i]-X_mean)**2

    m = Numerator/Denomenator
    print("Slope of Regression line is",m)

    c = (Y_mean)-(m*X_mean)

    print("Y intersept:",c)

    max_x = np.max(X)
    min_x = np.min(X)
    x = np.linspace(min_x,max_x,n)

    y = m*x+c
    #ploting
    #Regression line plotting
    plt.plot(x,y,color='#58b970',label = "Regression Line")

    #all points
    plt.scatter(X,Y,color='#ef5423',label = "Scatter plot")

    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")

    plt.legend()
    plt.show()


    #Goodness of fit RSquare

    SSr = 0
    SSt = 0
    #SSr is the residual sum of squares.
    #SSt is the total sum of squares.


    for i in range(n):
        yprd = c + m * X[i]
        SSr += (Y[i]-yprd)**2
        SSt += (Y[i]-Y_mean)**2
        
    R2 = 1-(SSr/SSt)

    print(f"Goodness of Fit is:{R2}")

############################################################################

# Entry function.
def main():
    HeadBrain()
    
############################################################################

# Main Starter
if __name__ == "__main__":
    main()
    
############################################################################