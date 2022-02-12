import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
enc = ColumnTransformer([('State', OneHotEncoder(),[3])], remainder='passthrough')
X = enc.fit_transform(X)

# avoiding Dummy variable/redundant dependancy
X = X[:, 1:]


# Splitting dataset into train_set and test_set
# 20% for test set/ 10 observation for test and 40 for training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting Simple linear regresseur to train set

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)



#predicting the test set Result
y_pred = reg.predict(X_test)


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


'''import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)'''


'''import statsmodels.formula.api as sm
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
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)'''