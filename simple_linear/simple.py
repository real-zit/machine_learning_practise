import numpy as np
import matplotlib as plt
import pandas as pd

#importing the dataset
df = pd.read_csv('cricket.xls')

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#splitting the dataset into train test
from sklearn.model_selection import train_test_split
X_train , X_test, y_train , y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0) 


#fitting the simple linear regression to data
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train , y_train)

#predicting the result 
y_pred = linear.predict(X_test)


#visualinsing the result

plt.scatter(x_test, y_test , color = 'red')
plt.plot(X_train ,   regressor.predict(X_train), color = 'blue' )
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of experience")
plt.ylabel("salary")
plt.show()