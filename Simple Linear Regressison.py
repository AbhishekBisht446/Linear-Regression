#Simple Linear Regerimport pandas as pd
import matplotlib.pyplot as py
import numpy as np
import pandas as pd 

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values #independent variable
Y=dataset.iloc[:,1].values   #dependent variable




# spliting the data set into Training set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state=0)

#fitting Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the Test set results
Y_pred = regressor.predict(X_test)

#Visualsing the training  set result
py.scatter(X_train,Y_train,color ='red')
py.plot(X_train,regressor.predict(X_train),color ='blue')
py.title("Salary vs Experience(Training Set)")
py.xlabel("Years of Experience")
py.ylablel("Salary")



#test set
py.scatter(X_test,Y_test,color ='green')
py.plot(X_test,regressor.predict(X_test),color ='blue')
py.title("Salary vs Experience(Training Set)")
py.xlabel("Years of Experience")
py.ylablel("Salary")
