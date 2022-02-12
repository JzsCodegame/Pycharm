import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



# Splitting dataset into train_set and test_set
# 20% for test set/ 10 observation for test and 40 for training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting Simple linear regresseur to train set

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# Fitting the SVR to Data set
from sklearn.ensemble import RandomForestRegressor
'''regr_Rndforest = RandomForestRegressor(n_estimators)
regr_Rndforest.fit(X, y)'''

from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)


# Predicting Result with Linear Regression
#y_pred = regr_Rndforest.predict((np.array([6.5]).reshape(1, 1)))
y_pred = rf_random.predict((np.array([6.5]).reshape(1, 1)))

# visualizing and plotting Random Forest Regression graphs for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
#plt.plot(X_grid, regr_Rndforest.predict(X_grid), color='blue')
plt.plot(X_grid, rf_random.predict(X_grid), color='blue')
plt.title('Truth Or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




