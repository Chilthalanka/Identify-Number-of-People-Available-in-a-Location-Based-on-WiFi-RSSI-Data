# 1. Importing libraries
import csv
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as pl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.externals import joblib   #to save the classifier model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors as neighbours

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#2. Load the Unknown Data
unknownData = pd.read_csv("unknownFinal_All.csv")


#3. Preprocess Unknown Data
feature_names = ['Have a Nice Day10', 'WiFire1']
X_Unknown_test = unknownData.drop(['No_of_P'], axis=1)
X_Unknown_test_mat = X_Unknown_test.as_matrix()


#4. Load the Saved Model
filename = 'DT_Model_Final.sav'
model = joblib.load(filename)


#5. Predict Unknown Data
y_Unknown_pred = model.predict(X_Unknown_test_mat)


#6. Add Results to a csv File

#Read lines from the unknownData.csv file
with open('unknownFinal_All.csv', 'r') as fi:
    lines = [[i.strip() for i in line.strip().split(',')] \
             for line in fi.readlines()]

tempList = list(y_Unknown_pred)
col = ['No_of_P'] +  tempList
col = np.asarray(col)
#print(col)

#Concatenate each row with corresponding element of col. Using enumerate to help keep track of which list index to use.
new_lines = [line + [str(col[i])] for i, line in enumerate(lines)]

#Write to file
with open('Unknown_Predicted_DT.csv', 'w') as fo:
    for line in new_lines:
        fo.write(','.join(line) + '\n')

print("\nPrediction completed!")

