# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:49:01 2021

@author: akarsh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'G:\mini_project\data_fs.csv',index_col=0)


X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Scaled_X_train=scaler.fit_transform(X_train)
Scaled_X_test=scaler.transform(X_test)


parameter_space = {'random_state':[5,100],
                   'hidden_layer_sizes': [(10,30,10),(20,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive']}

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
snn_classifier = MLPClassifier()
clf = GridSearchCV(snn_classifier, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train)
snn_predictions = clf.predict(X_test)



from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
clf_ADB = AdaBoostClassifier(random_state=0,base_estimator=RandomForestClassifier(random_state=0))
clf_ADB=clf_ADB.fit(X_train,y_train)

y_pred=clf_ADB.predict(X_test)

# _________ACCURACIES_________________
import sklearn.metrics as metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, snn_predictions))



