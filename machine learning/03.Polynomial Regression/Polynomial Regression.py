import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

# import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Simple linear regresseur to Data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial linear reg to Data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Add Linear Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# visualizing and plotting graphs for linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth Or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# visualizing and plotting graphs for polynomial
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth Or Bluff (03.Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting Result with Linear Regression
bluff_Or = lin_reg.predict(np.array([6.5]).reshape(1, 1))

# Predicting Result with Linear Regression
bluff_no = lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))
























