import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer


# import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting the non Decision Tree to Data set
from sklearn.tree import DecisionTreeRegressor
deci_reg = DecisionTreeRegressor(random_state=0)
deci_reg.fit(X, y)



# Predicting Result with Linear Regression
y_pred = deci_reg.predict((np.array([6.5]).reshape(1, 1)))


# Visualizing and plotting graphs for Decision Tree
'''plt.scatter(X, y, color='red')
plt.plot(X, deci_reg.predict(X), color='blue')
plt.title('Truth Or Bluff (03.Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()'''

# visualizing and plotting Decision Tree graphs for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, deci_reg.predict(X_grid), color='blue')
plt.title('Truth Or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
