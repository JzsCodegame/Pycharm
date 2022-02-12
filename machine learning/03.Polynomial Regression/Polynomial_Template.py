import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values




# Fitting the non linear reg to Data set




# Predicting Result with Linear Regression
y_pred = reg.predict((np.array([6.5]).reshape(1, 1)))



# visualizing and plotting graphs for polynomial
plt.scatter(X, y, color='red')
plt.plot(X, reg.predict(X), color='blue')
plt.title('Truth Or Bluff (03.Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing and plotting graphs for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, reg.predict(X), color='blue')
plt.title('Truth Or Bluff (03.Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


