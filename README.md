# credit-card
Credit Card Fraud Detection algorithm using smote , confusion matrix, correlation matrix, density plots and ROC-AUC curve
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 17:02:55 2020

@author: sudevpradhan
CREDIT CARD FRAUD DETECTION
Data sets from https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#Load the Dataset
data = pd.read_csv('creditcard.csv')

data.head(10)

df = pd.read_csv('creditcard.csv')
print(df.shape)
df.head()

df.info()

df.describe()

class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))


fig = plt.figure(figsize = (15, 12))

def subplot():
    plt.subplot(5, 6, 1) ; plt.plot(df.V1) ; plt.subplot(5, 6, 15) ; plt.plot(df.V15)
    plt.subplot(5, 6, 2) ; plt.plot(df.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df.V16)
    plt.subplot(5, 6, 3) ; plt.plot(df.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df.V17)
    plt.subplot(5, 6, 4) ; plt.plot(df.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df.V18)
    plt.subplot(5, 6, 5) ; plt.plot(df.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df.V19)
    plt.subplot(5, 6, 6) ; plt.plot(df.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df.V20)
    plt.subplot(5, 6, 7) ; plt.plot(df.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df.V21)
    plt.subplot(5, 6, 8) ; plt.plot(df.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df.V22)
    plt.subplot(5, 6, 9) ; plt.plot(df.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df.V23)
    plt.subplot(5, 6, 10) ; plt.plot(df.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df.V24)
    plt.subplot(5, 6, 11) ; plt.plot(df.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df.V25)
    plt.subplot(5, 6, 12) ; plt.plot(df.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df.V26)
    plt.subplot(5, 6, 13) ; plt.plot(df.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df.V27)
    plt.subplot(5, 6, 14) ; plt.plot(df.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df.V28)
    plt.subplot(5, 6, 29) ; plt.plot(df.Amount)
    plt.show()


data.isnull().sum()


def imbalance():


    #Print the value counts of frauds and non-frauds in the data
    print(data['Class'].value_counts())

    #Calculate the percentage of Fraud and Non-fraud transactions.
    print('Valid Transactions: ', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

    print('Fraudulent Transactions: ', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

    #Visualizing the class Imbalance
    colors = ['blue','red']
    sns.countplot('Class', data=data, palette=colors)




def preprocessing(X,y):
    #Splitting the Data


    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=0)

    print("Transactions in X_train dataset: ", X_train.shape)
    print("Transaction classes in y_train dataset: ", y_train.shape)

    print("Transactions in X_test dataset: ", X_test.shape)
    print("Transaction classes in y_test dataset: ", y_test.shape)

    #Feature Scaling

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()

    #   normalising
    X_train['normAmount'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1, 1))

    X_test['normAmount'] = scaler_amount .transform(X_test['Amount'].values.reshape(-1, 1))

    X_train['normTime'] = scaler_time .fit_transform(X_train['Time'].values.reshape(-1, 1))

    X_test['normTime'] = scaler_time .transform(X_test['Time'].values.reshape(-1, 1))


    X_train = X_train.drop(['Time', 'Amount'], axis=1)
    X_test = X_test.drop(['Time', 'Amount'], axis=1)

    X_train.head()
    return X_train, X_test, y_train, y_test
    
def using_smote(X_train, X_test, y_train, y_test):   


    print("Before over-sampling:\n", y_train['Class'].value_counts())

    sm = SMOTE()

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train['Class'])

    print("After over-sampling:\n", y_train_res.value_counts())

    #Build the Model


    parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
    k = clf.fit(X_train_res, y_train_res)
    
    print(k.best_params_)

    #Evaluate the Model
    lr_gridcv_best = clf.best_estimator_

    y_test_pre = lr_gridcv_best.predict(X_test)

    cnf_matrix_test = confusion_matrix(y_test, y_test_pre)

    print("Recall metric in the test dataset:", (cnf_matrix_test[1,1]/(cnf_matrix_test[1,0]+cnf_matrix_test[1,1] )))

    y_train_pre = lr_gridcv_best.predict(X_train_res)

    cnf_matrix_train = confusion_matrix(y_train_res, y_train_pre)

    print("Recall metric in the train dataset:", (cnf_matrix_train[1,1]/(cnf_matrix_train[1,0]+cnf_matrix_train[1,1] )))
    return k,X_test, y_test, X_train_res, y_train_res
    
def confusion(k,X_test, y_test, X_train_res, y_train_res):    
    #Visualize the Confusion Matrix

    plt.style.use('seaborn')
    class_names = ['Not Fraud', 'Fraud']
    plot_confusion_matrix(k, X_test, y_test,  values_format = '.5g', display_labels=class_names)
    plt.title("Test data Confusion Matrix")
    plt.show()


    plot_confusion_matrix(k, X_train_res, y_train_res,  values_format = '.5g', display_labels=class_names) 
    plt.title("Oversampled Train data Confusion Matrix")
    plt.show()
    
def ROC(X_test,y_test):
    y_k =  k.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_k)

    roc_auc = auc(fpr, tpr)

    print("ROC-AUC:", roc_auc)


    # Now visualize the roc_auc curve.
    plt.style.use('seaborn')
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.style.use('classic')
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='white', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
    
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
    
#MAIN FUNCTION    
    
    
#Exploring the Class Column
X = data.loc[:, data.columns != 'Class']
y = data.loc[:, data.columns == 'Class']  


nRowsRead = 1000
df1 = pd.read_csv('creditcard.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'creditcard.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


#FUNCTION CALL
imbalance()
X_train, X_test, y_train, y_test=preprocessing(X,y)
k,X_test, y_test, X_train_res, y_train_res=using_smote(X_train, X_test, y_train, y_test)
confusion(k,X_test, y_test, X_train_res, y_train_res)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
ROC(X_test, y_test)









