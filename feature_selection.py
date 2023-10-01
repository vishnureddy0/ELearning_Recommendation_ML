# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:31:56 2021

@author: akarsh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

df=pd.read_csv(r'G:\mini_project\label_data.csv',index_col=0)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

ranks= mutual_info_classif(X,y)

feat_importances = pd.Series(ranks,df.columns[0:len(df.columns)-1])
feat_importances.plot(kind='barh',color='teal')
plt.show()

features=[]
index=[]
for i in range(len(ranks)):
    if ranks[i]>=0.005:
        features.append(df.columns[i])
        index.append(i)
list(df.columns)
        
new_df=df.drop(columns =Diff(list(df.columns),features))

new_df['learning_style']=df['learning_style']

new_df.to_csv("G:\mini_project\data_fs.csv")
