import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer


# import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# missing values
from sklearn.impute import SimpleImputer as imp
imputer = imp(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



# avoiding Dummy variable/redundant dependancy
X = X[:, 1:]



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
enc = ColumnTransformer([('Position', OneHotEncoder(),[0])], remainder='passthrough')
X = enc.fit_transform(X)



#Splitting dataset into train_set and test_set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.astype(float))
y = sc_y.fit_transform(y.reshape(-1, 1).astype(float)).ravel()


2ND VERSION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.reshape(-1, 1).astype(float)).ravel())
X_test = sc.transform(X_test.reshape(-1, 1).astype(float)).ravel())



# 1. Fitting Simple linear regresseur to train set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)



# 2. Fitting Polynomial linear reg to Data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)



# 3. Add Linear Regression on Polynomial
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)



# Fitting the SVR to Data set
from sklearn.ensemble import RandomForestRegressor
regr_Rndforest = RandomForestRegressor(n_estimators=100, random_state=0)
regr_Rndforest.fit(X, y)




#Fitting Simple linear regresseur to train set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)




# Fitting the non Decision Tree to Data set
from sklearn.tree import DecisionTreeRegressor
deci_reg = DecisionTreeRegressor(random_state=0)
deci_reg.fit(X, y)



# Fitting the SVR to Data set
from sklearn.ensemble import RandomForestRegressor
regr_Rndforest = RandomForestRegressor(n_estimators=100, random_state=0)
regr_Rndforest.fit(X, y)


# Fitting the LogisticRegression to Data set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


#predicting the test set Result
y_pred = reg.predict(X_test.reshape(1, 1))


# Predicting Result with Linear Regression
y_pred = deci_reg.predict((np.array([6.5]).reshape(1, 1)))

# Predicting Result with Linear Regression
y_pred = sc_y.inverse_transform((sc_X.transform(np.array([[6.5]]).reshape(1, 1))))
#1.(np.array([[6.5]]).reshape(1, 1)))
#2.((np.array([6.5]).reshape(-1, 1)))



#Building Optimum Model by Backward Elimination

# Backward elimination
''' 1 . Select signaficant level for the model : Independent Value
    2 . Fit the model to all posible predictors
    3 . Select the highest p_VALUE predictor
              - If P > Sl go to step 4
              - else go to FIN : Model IS Ready  
    4 . Remove the Predictor
    5 . Fit Model without this Variable

'''

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0 , 1, 2, 3, 4, 5]]
'''3 . Select the highest p_VALUE predictor
        - If P > Sl go to step 4
        - else go to FIN : Model IS Ready'''
X_opt = np.array(X_opt, dtype=float)
mdl = sm.OLS(endog = y, exog = X_opt).fit()
'''5 . Fit Model without this Variable'''
print (mdl.summary())

X_opt = X[:, [0 , 1, 3, 4, 5]]
'''4 . Remove the Predictor'''
X_opt = np.array(X_opt, dtype=float)
mdl = sm.OLS(endog = y, exog = X_opt).fit()
'''5 . Fit Model without this Variable'''
print(mdl.summary())

X_opt = X[:, [0, 3, 4, 5]]
'''4 . Remove the Predictor'''
X_opt = np.array(X_opt, dtype=float)
mdl = sm.OLS(endog = y, exog = X_opt).fit()
'''5 . Fit Model without this Variable'''
print(mdl.summary())
''' 3 . Select the highest p_VALUE predictor
        - If P > Sl go to step 4
        - else go to FIN : Model IS Ready'''
X_opt = X[:, [0, 3, 5]]
'''4 . Remove the Predictor'''
X_opt = np.array(X_opt, dtype=float)
mdl = sm.OLS(endog = y, exog = X_opt).fit()
'''5 . Fit Model without this Variable'''
print(mdl.summary())
''' 3 . Select the highest p_VALUE predictor
        - If P > Sl go to step 4
        - else go to FIN : Model IS Ready'''
X_opt = X[:, [0, 3]]
'''4 . Remove the Predictor'''
X_opt = np.array(X_opt, dtype=float)
mdl = sm.OLS(endog = y, exog = X_opt).fit()
'''5 . Fit Model without this Variable'''
print(mdl.summary())
''' 3 . Select the highest p_VALUE predictor
        - If P > Sl go to step 4
        - else go to FIN : Model IS Ready'''



import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)


SL = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
X_Modeled = backwardElimination(X_opt, SL)








# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing and plotting graphs for Decision Tree
plt.scatter(X, y, color='red')
plt.plot(X, deci_reg.predict(X), color='blue')
plt.title('Truth Or Bluff (03.Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# visualizing and plotting Decision Tree graphs for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, deci_reg.predict(X_grid), color='blue')
plt.title('Truth Or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Predicting Result with Linear Regression
bluff_Or = lin_reg.predict(np.array([6.5]).reshape(1, 1))

# Predicting Result with Linear Regression
bluff_no = lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))

