# 1. Importing libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 2. Importing the dataset
data = pd.read_csv('dataFinal_All.csv')

# 3. Data Preprocessing
X = data.drop(['No_of_P'], axis=1)
y = data['No_of_P']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5) #shuffle = True, random_state = Nona

# 4. Model Implementation
xg_clf = xgb.XGBClassifier(objective='multi:softmax',
                           colsample_bytree= 0.8,
                           learning_rate=0.1,
                           max_depth=5,
                           min_child_weight=1,
                           gamma=0,
                           reg_lambda=0,
                           n_estimators=100,
                           eval_metric='mlogloss')

eval_set = [(X_train, y_train), (X_test, y_test)]
xg_clf.fit(X_train, y_train, eval_set= eval_set, eval_metric=["mlogloss", "mlogloss"] )
preds = xg_clf.predict(X_test)
preds1 = xg_clf.predict(X_train)

# 5. Retrieve performance metrics
results = xg_clf.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# 6. Plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

# 7. plot feature importance graph
plot_importance(xg_clf)
pyplot.show()

# 8. Accuracy Reports
### Train Results ###
accuracy_scr = accuracy_score(y_train, preds1)
report = classification_report(y_train, preds1)
print('Train set results:')
print("Accuracy score of XGBoost classifier on training set: %f" % (accuracy_scr * 100))
print(report)

### Test Results ###
accuracy_scr = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
print('Test set results:')
print("Accuracy score of XGBoost classifier on test set: %f" % (accuracy_scr * 100))
print(report)

# 9. creating the output dataset
outputDataset = [['y_test','y_pred']]
count = 0
for i in list(y_test.index):
    record = []
    record.append(y_test[i])
    record.append(preds[count])
    count+=1
    outputDataset.append(record)

### Creating CSV File to save the outputs ###
myFile = open('wifi-classifier-results-final.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(outputDataset)

# 10. Save to file in the current working directory
joblib_file = "joblib_model.pkl"
joblib.dump(xg_clf, joblib_file)








