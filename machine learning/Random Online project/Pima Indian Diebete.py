import pandas as pd
import numpy as np
# list for column headers
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# open file with pd.read_csv
dfset = pd.read_csv('pimadata.csv', names=names)
print(dfset.shape)
# print head of data set
print(dfset.head())


#Creating a Random Forest Model
#Let’s create our model.
#We are trying to predict whether a patient has diabetes.
#This coincides with the ‘class’ column, which will be our independent variable.
#We’ll use all the other columns as features for our model.

X = dfset.drop('class', axis=1)
y = dfset['class']


#We’ll use train-test-split to split the data into training data and testing data.
from sklearn.model_selection import train_test_split
# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)


# Now, we can create the random forest model.
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


# predictions
rfc_predict = rfc.predict(X_test)




#Evaluating Performance
#We’ll import cross_val_score, classification_report, and confusion_matrix.
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix



#We’ll also run cross-validation to get a better overview of the results.
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')


#Now, we’ll print out the results.
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())



from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
#number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)


# create random grid
random_grid = {'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }


# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train, y_train)
# print results
print(rfc_random.best_params_)


rfc = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())