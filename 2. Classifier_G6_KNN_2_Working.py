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

tic = time.time()


# 2. Importing the dataset
wifidata = pd.read_csv("dataFinal_All.csv")


# 3. Exploratory Data Analysis
#check the dimensions of the data and see first few records
print("Dimensions of the data: {}".format(wifidata.shape))
print("\nFirst few records:")
print(wifidata.head())
print ("\nNumber of unique labels: {}".format(wifidata['No_of_P'].unique()))
print ("\nNumber of examples in each category: {}".format(wifidata.groupby('No_of_P').size()))


'''
#4. Visualize data
#Histogram
wifidata.drop('No_of_P' ,axis=1).hist(bins=30)
pl.suptitle("Histogram for Each Input Variable")
plt.show()

#Example count from each category
fig1 = plt.figure(figsize=(8, 6))
sns.countplot(wifidata['No_of_P'], label='Count')
plt.title('No of Examples from Each Category')
fig1.show()

#Measure the distribiution of each feature
wifidata.drop('No_of_P', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for Each Input Variable')
plt.show()
'''


# 5. Data Preprocessing
feature_names = ['Have a Nice Day10', 'WiFire1']
all_features = (wifidata.columns).drop(['No_of_P'])
all_features = all_features.values
#X = wifidata[feature_names]
X = wifidata.drop(['No_of_P'], axis=1) #contains attributes
y = wifidata['No_of_P'] #contains coresponding labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)  #shuffle = True, random_state=None

X_train_mat = X_train.as_matrix()
y_train_mat = y_train.as_matrix()
X_test_mat = X_test.as_matrix()
y_test_mat = y_test.as_matrix()
X_train_plot = X_train[feature_names].as_matrix()
X_test_plot = X_test[feature_names].as_matrix()


#6. Check Accuracies of Several Models
#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('\n\nAccuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('\n\nAccuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))

#K-NN
knn = neighbours.KNeighborsClassifier()
knn.fit(X_train, y_train)
print('\n\nAccuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
pred_KNN = knn.predict(X_test_mat)
print("Confusion Matrix and Classification Report of K-NN:")
print(confusion_matrix(y_test_mat, pred_KNN))
print(classification_report(y_test_mat, pred_KNN))
#filename = 'KNN_Model_Final.sav'  #Save K-NN model
#joblib.dump(knn, filename)


#Support Vector Machine - SVM
svm = SVC()
svm.fit(X_train, y_train)
print('\n\nAccuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
pred_SVM = svm.predict(X_test_mat)
print("Confusion Matrix and Classification Report of SVM:")
print(confusion_matrix(y_test_mat, pred_SVM))
print(classification_report(y_test_mat, pred_SVM))
#filename = 'SVM_Model_Final.sav'  #Save SVM model
#joblib.dump(svm, filename)


#Decision Tree
dt = DecisionTreeClassifier().fit(X_train, y_train)
print('\n\nAccuracy of Decision Tree classifier on training set: {:.2f}'.format(dt.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dt.score(X_test, y_test)))
pred_DT = dt.predict(X_test_mat)
print("Confusion Matrix and Classification Report of Decision Tree:")
print(confusion_matrix(y_test_mat, pred_DT))
print(classification_report(y_test_mat, pred_DT))
#filename = 'DT_Model_Final.sav'  #Save DT model
#joblib.dump(dt, filename)







# 7. Training the Most Suitable Algorithms: K-NN | SVM  |   Decision Tree

def plot_K_NN(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, n_neighbors = 5, Weights='uniform', Step_Size = 0.02):
    print("\n\n Classification Using K-NN\n\n")
    #Compare error rate with different K values
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):  
        knn = neighbours.KNeighborsClassifier(n_neighbors=i, weights=Weights)
        knn.fit(X_train_mat, y_train_mat)
        pred_i = knn.predict(X_test_mat)
        error.append(np.mean(pred_i != y_test))

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error')
    fig1.show()
    
    error_np = np.array(error)    
    minIndex_K = np.argmin(error_np) + 1
    if (error_np[minIndex_K] == error_np[minIndex_K+1]):
        minIndex_K += 1
    print ("\nK value with minimum error rate: {}\n".format(minIndex_K))

    n_neighbors = minIndex_K  #Update the minimum error rate K value
    
    ##### Plot most effective two features #####
    clf = neighbours.KNeighborsClassifier(n_neighbors, weights=Weights)
    clf.fit(X_train_plot, y_train_mat)

    # Plot the decision boundary by assigning a color in the color map to each mesh point.    
    mesh_step_size =  Step_Size # step size in the mesh  - 0.01
    plot_symbol_size = 25   #50

    x_min, x_max = X_train_mat[:, 0].min() - 1, X_train_mat[:, 0].max() + 1
    y_min, y_max = X_train_mat[:, 1].min() - 1, X_train_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print("\nDone plotting the meshgrid!")

    Z = Z.reshape(xx.shape)

    #####  Plot Train Set  #####
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    colors = {0: 'violet', 1: 'blue', 5: 'yellow'}     #{0: 'violet', 1: 'indigo', 5: 'palegreen'}
    
    # Plot the contour map
    ax2.contourf(xx, yy, Z, cmap=plt.cm.viridis)   #cmap=plt.cm.PRGn    |  RdYlGn
    ax2.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_train_mat):
        indices = np.where(y_train_mat == label)
        ax2.scatter(X_train_mat[indices, 0], X_train_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
    
    ax2.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("3 Class classification (k = %i, weights = '%s') - Train Set" % (n_neighbors, Weights))
    fig2.show()

    #####  Plot Test Set   #####
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)    
    
    # Plot the contour map
    ax3.contourf(xx, yy, Z, cmap=plt.cm.viridis)
    ax3.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_test_mat):
        indices = np.where(y_test_mat == label)
        ax3.scatter(X_test_mat[indices, 0], X_test_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
    
    ax3.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("3 Class classification (k = %i, weights = '%s') - Test Set" % (n_neighbors, Weights))
    fig3.show()

    #Accuracy of K-NN Algorithm
    knn = neighbours.KNeighborsClassifier(n_neighbors=n_neighbors, weights=Weights)
    knn.fit(X_train_mat, y_train_mat)
    print('\n\nAccuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_mat, y_train_mat)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test_mat, y_test_mat)))
    pred = knn.predict(X_test_mat)
    print(confusion_matrix(y_test_mat, pred))
    print(classification_report(y_test_mat, pred))







def plot_SVM(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, Kernel, Step_Size=0.02):
    print("\n\n Classification Using SVM\n\n")
        
    ##### Plot most effective two features #####
    clf = SVC(kernel=Kernel)
    clf.fit(X_train_plot, y_train_mat)    

    # Plot the decision boundary by assigning a color in the color map to each mesh point.    
    mesh_step_size =  Step_Size # step size in the mesh  - 0.01
    plot_symbol_size = 25   #50

    x_min, x_max = X_train_mat[:, 0].min() - 1, X_train_mat[:, 0].max() + 1
    y_min, y_max = X_train_mat[:, 1].min() - 1, X_train_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print("\nDone plotting the meshgrid!")

    Z = Z.reshape(xx.shape)
    
    colors = {0: 'violet', 1: 'blue', 5: 'yellow'}     #{0: 'violet', 1: 'indigo', 5: 'palegreen'}

    #####  Plot SVM - Train Set   #####   
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)    
    
    # Plot the contour map
    ax2.contourf(xx, yy, Z, cmap=plt.cm.viridis)   #cmap=plt.cm.PRGn    |  RdYlGn
    ax2.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_train_mat):
        indices = np.where(y_train_mat == label)
        ax2.scatter(X_train_mat[indices, 0], X_train_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
        
    ax2.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("SVM with {} Kernel - Train Set".format(Kernel))
    fig2.show()

    #####  Plot SVM - Test Set   #####
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)    
    
    # Plot the contour map
    ax3.contourf(xx, yy, Z, cmap=plt.cm.viridis)
    ax3.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_test_mat):
        indices = np.where(y_test_mat == label)
        ax3.scatter(X_test_mat[indices, 0], X_test_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
    
    ax3.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("SVM with {} Kernel - Test Set".format(Kernel))
    fig3.show()

    #Accuracy of K-NN Algorithm
    clf_svm = SVC(kernel=Kernel)     #kernel=Kernel
    clf_svm.fit(X_train_mat, y_train_mat)
    print('\n\nAccuracy of SVM classifier with {} Kernel on training set: {:.2f}'.format(Kernel, clf_svm.score(X_train_mat, y_train_mat)))
    print('Accuracy of SVM classifier with {} Kernel on test set: {:.2f}'.format(Kernel, clf_svm.score(X_test_mat, y_test_mat)))
    pred = clf_svm.predict(X_test_mat)
    print(confusion_matrix(y_test_mat, pred))
    print(classification_report(y_test_mat, pred))





    


def plot_DT(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, feature_cols, Step_Size=0.02):
    print("\n\n Classification Using Decision Tree\n\n")
        
    ##### Plot most effective two features #####
    clf_1 = DecisionTreeClassifier()
    #params = {'criterion': ['gini', 'entropy'], 'max_depth': [3,4,5,6,7], 'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],  'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11], 'random_state':[10]}
    #clf_1 = GridSearchCV(clf, param_grid=params, n_jobs=-1) #Making models with hyper parameters sets
    clf_1.fit(X_train_plot, y_train_mat)

    #print("Best Hyper Parameters:", clf_1.best_params_)

    # Plot the decision boundary by assigning a color in the color map to each mesh point.    
    mesh_step_size =  Step_Size # step size in the mesh  - 0.01
    plot_symbol_size = 25   #50

    x_min, x_max = X_train_mat[:, 0].min() - 1, X_train_mat[:, 0].max() + 1
    y_min, y_max = X_train_mat[:, 1].min() - 1, X_train_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))    
    Z = clf_1.predict(np.c_[xx.ravel(), yy.ravel()])
    print("\nDone plotting the meshgrid!")

    Z = Z.reshape(xx.shape)
    
    colors = {0: 'violet', 1: 'blue', 5: 'yellow'}     #{0: 'violet', 1: 'indigo', 5: 'palegreen'}

    #####  Plot Decision Tree - Train Set   #####   
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)    
    
    # Plot the contour map
    ax2.contourf(xx, yy, Z, cmap=plt.cm.viridis)   #cmap=plt.cm.PRGn    |  RdYlGn
    ax2.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_train_mat):
        indices = np.where(y_train_mat == label)
        ax2.scatter(X_train_mat[indices, 0], X_train_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
        
    ax2.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("Decision Tree Classifier - Train Set")
    fig2.show()

    #####  Plot Decision Tree - Test Set   #####
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)    
    
    # Plot the contour map
    ax3.contourf(xx, yy, Z, cmap=plt.cm.viridis)
    ax3.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y_test_mat):
        indices = np.where(y_test_mat == label)
        ax3.scatter(X_test_mat[indices, 0], X_test_mat[indices, 1], s=plot_symbol_size, c=colors[label], edgecolor = 'black', alpha=0.8, label='Number of People: {}'.format(label)) 
    
    ax3.legend(loc = 2)
    plt.xlabel('Have a Nice Day11 (dB)')
    plt.ylabel('WiFire6 (dB)')
    plt.title("Decision Tree Classifier - Test Set")
    fig3.show()

    #Accuracy of K-NN Algorithm
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_train_mat, y_train_mat)
    print('\n\nAccuracy of Decision Tree classifier on training set: {:.2f}'.format(clf_dt.score(X_train_mat, y_train_mat)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf_dt.score(X_test_mat, y_test_mat)))
    pred = clf_dt.predict(X_test_mat)
    print(confusion_matrix(y_test_mat, pred))
    print(classification_report(y_test_mat, pred))

    '''
    dot_data = StringIO()
    export_graphviz(clf_dt, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names=['0','1', '5'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('Decision_Tree_Graph.png')
    Image(graph.create_png())
    '''




plot_K_NN(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, n_neighbors = 5, Weights='uniform', Step_Size=0.05)   #Classify using K-NN

#plot_SVM(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, Kernel='rbf', Step_Size=0.02)   #Classify using SVM

#plot_DT(X_train_mat, y_train_mat, X_test_mat, y_test_mat, X_train_plot, X_test_plot, feature_cols = all_features, Step_Size=0.02)    #Classify using Decision Tree




toc = time.time()
difference = float((toc - tic)/60.0)
print("\nTime taken to run: {}min".format(difference))
