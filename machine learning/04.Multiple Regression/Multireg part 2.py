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
'''Automated Backwaerd Elimination 
    1 . Select signaficant level for the model : Independent Value
    2 . Fit the model to all posible predictors
    3 . Select the highest p_VALUE predictor
              - If P > Sl go to step 4
              - else go to FIN : Model IS Ready  
    4 . Remove the Predictor
    5 . Fit Model without this Variable

'''
import statsmodels.formula.api as sm


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        mdl = sm.OLS(endog = y, exog= x).fit()
        maxVar = max(mdl.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (mdl.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(mdl.summary())
    return x

SL = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
X_Modeled = backwardElimination(X_opt, SL)

