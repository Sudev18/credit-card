# credit-card
Credit Card Fraud Detection algorithm using smote , confusion matrix, correlation matrix, density plots and ROC-AUC curve
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

# Other Libraries
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit as sss
import warnings
warnings.filterwarnings("ignore")

#Load the Dataset
data = pd.read_csv('creditcard.csv')


df = pd.read_csv('creditcard.csv')
print(df.shape)
df.head()

df.info()

df.describe()

class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))


fig = plt.figure(figsize = (15, 12))


def imbalance():


    #Print the value counts of frauds and non-frauds in the data
    print(data['Class'].value_counts())

    #Calculate the percentage of Fraud and Non-fraud transactions.
    print('Valid Transactions: ', round(data['Class'].value_counts()[0]/len(data) * 100,2)
          , '% of the dataset')

    print('Fraudulent Transactions: ', round(data['Class'].value_counts()[1]/len(data) * 100,2)
          , '% of the dataset')

    #Visualizing the class Imbalance
    colors = ['blue','red']
    sns.countplot('Class', data=data, palette=colors)


def data_visualisation():
    #Distribution of the amount and time in the data sets
    fig, ax = plt.subplots(1, 2, figsize=(16,4))
    fig.suptitle('Distribution', fontsize=16)
    amount_val = df['Amount'].values
    time_val = df['Time'].values
    colors = ["#0101DF", "#DF0101"]
    sns.distplot(amount_val, ax=ax[0], color=colors[0])
    ax[0].set_title('Distribution of Transaction Amount')
    ax[0].set_xlim([min(amount_val), max(amount_val)])
    
    sns.distplot(time_val, ax=ax[1], color=colors[1])
    ax[1].set_title('Distribution of Transaction Time')
    ax[1].set_xlim([min(time_val), max(time_val)])
    plt.show()
    
    #fraud amount and non fraud amount
    fig, ax = plt.subplots(1, 2, figsize=(16,4), sharex=True)
    fig.suptitle('Amount/transaction', fontsize=16)
    colors = ["#0101DF", "#DF0101"]

    sns.distplot(df[df['Class']==1].Amount, ax=ax[0], color=colors[0])
    plt.xlabel('Amount')
    plt.ylabel('Number of Transactions')
    ax[0].set_title('Distribution of Transaction Amount (Fraud)')    
    sns.distplot(df[df['Class']==0].Amount, ax=ax[1], color=colors[1])
    ax[1].set_title('Distribution of Transaction Amount (Valid)')
    plt.xlabel('Amount')
    plt.ylabel('Number of Transactions')
    plt.xlim((0, 20000))
    plt.yscale('log')
    plt.show()
    
    
    # scatter plot of the fraudulent and non fraudulent data against time   
    fig, ax = plt.subplots(1, 2, figsize=(16,4), sharex=True)
    fig.suptitle('Time of transaction vs Amount', fontsize=16)
    colors = ["#0101DF", "#DF0101"]
    ax[0].scatter(df[df['Class']==1].Time, df[df['Class']==1].Amount)
    ax[0].set_title('Fraud')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    ax[1].scatter(df[df['Class']==0].Time, df[df['Class']==0].Amount)
    ax[1].set_title('Valid')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.show()


    

def plotCorrelationMatrix():
    features = df.columns.values
    
    correlation_matrix = df.corr()
    fig = plt.figure(figsize=(12,8))
    fig.suptitle('Correlation Plot', fontsize=16)
    sns.heatmap(correlation_matrix,vmax=0.8,square = True)
    plt.show()
    correlations = df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    print("Top 6 correlated features")
    print(correlations.head(5)) #
    print("\n Least 6 correlated features")
    print(correlations.tail(5))
    
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

    print(classification_report(y_test,y_test_pre))
    

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
    
def regression():


    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.01, 0.1, 1]}
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)
    log_reg = grid_log_reg.best_estimator_
    log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
    print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
    
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import learning_curve
    
    '''def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        f, ((ax1)) = plt.subplots(1,1, figsize=(16,8), sharey=True)
        if ylim is not None:
            plt.ylim(*ylim)
            train_sizes, train_scores, test_scores = learning_curve(
                estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="#ff9124")
            ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
            ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                     label="Training score")
            ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                     label="Cross-validation score")
            ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
            ax1.set_xlabel('Training size (m)')
            ax1.set_ylabel('Score')
            ax1.grid(True)
            ax1.legend(loc="best")
            return plt'''

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=21)
    #plot_learning_curve(log_reg, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
    log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

    print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))




    
def Isolation_forest_algorithm():

    classifiers = {
        "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), verbose=0)}

    for i, (clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(X_train)
        scores_prediction = clf.decision_function(X_train)
        y_pred = clf.predict(X_train)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != y_train).sum()

        print("{}: {}".format(clf_name,n_errors))
        print("Accuracy Score :")
        print(accuracy_score(y_train,y_pred))
        print("Classification Report :")
        print(classification_report(y_train,y_pred))
        print('\n')


def knn():
    knn = KNeighborsClassifier(n_neighbors = 5,n_jobs=16)
    knn.fit(X_train,y_train)
    k1=knn.fit(X_train,y_train)
    print("")
    print("knn classifier created")
    score = knn.score(X_test,y_test)
    print("knn model score-")
    print(score)
    pred = knn.predict(X_test)
    print(classification_report(y_test,pred))
    matrix=confusion_matrix(y_test,pred)
    print(matrix)
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix,annot=True)
    prob=knn.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    plt.figure(figsize=(10,6))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
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

data_visualisation()
plotCorrelationMatrix()
plotScatterMatrix(df1, 20, 10)


confusion(k,X_test, y_test, X_train_res, y_train_res)
ROC(X_test, y_test)
knn()
X_train, X_test, y_train, y_test=preprocessing(X,y)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_train = X_train.values
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.values
y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = y_train.values
y_test = y_test.replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = y_test.values
Isolation_forest_algorithm()


#activate this code separetly, for the regression plot
'''regression()'''







