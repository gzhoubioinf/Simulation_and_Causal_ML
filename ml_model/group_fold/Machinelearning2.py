#!/usr/bin/env python
# coding: utf-8


# set up environment
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import mean_squared_error
#import seaborn as sb
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = LabelEncoder()

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.cluster.hierarchy import fcluster



def readfile(length,outfile,path):

    dl = 1061554//length
    rtab = 'len'+str(length)
    #pheno = pd.read_csv("pheno_data/"+out10+".phen",delimiter="\t",names=['id1','id2','value'])
    pheno = pd.read_csv(path+"data/pheno_data/"+outfile+".phen",
                        delim_whitespace=True,
                        names=['id1','id2','value'],
                        index_col=0)
    pheno = pheno.dropna(subset=['value'])
    pheno = pheno['value']
    finname = path+'/data/'+rtab+'unitigs_data.Rtab'
    if not  os.path.isfile(finname):
        fin = open(path+'/data/'+rtab+'unitigs_data.Rtab','w')
    
        with open(path+'/data/unitigs_trim.Rtab', 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                if length == 123456789 :
                    fin.write(line)
                else:

                    if index == 0:
                        fin.write(line)
                    elif index % dl == 1:
                        fin.write(line)

        fin.close()

    
    # read in unitig data
    X = pd.read_csv(path+'data/'+rtab+'unitigs_data.Rtab', 
                    sep=" ",
                    error_bad_lines=False,
                    index_col=0,
                    engine ='python')

    X = X.transpose()
    X = X[X.index.isin(pheno.index)] # only keep rows with a resistance measure
    pheno = pheno[pheno.index.isin(X.index)]
    #print(pheno.index)
    #print(X.index)
    return pheno, X


# function for fitting a model
def fitmodel(X, pheno, estimator, parameters, modelname, method, performance, times) :
    
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()
        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
        
        # perform grid search to identify best hyper-parameters
        gs_clf = GridSearchCV(estimator=estimator, 
            param_grid=parameters, 
            cv=3, 
            n_jobs=-1,
            scoring='neg_mean_squared_error')
        gs_clf.fit(X_train, y_train)
        
        # predict resistance in test set
        y_pred = gs_clf.predict(X_test)
                
        # call all samples with a predicted value less than or equal to 0.5 as sensitive to the antibiotic, 
        # and samples with predicted value >0.5 resistant to the antibiotic
        #y_pred[y_pred<=0.5] = 0
        #y_pred[y_pred>0.5] = 1

        score = mean_squared_error(y_test, y_pred)
        performance = np.append(performance, score)
        method = np.append(method, modelname)
        times = np.append(times, (time.process_time() - start))

        print("Best hyperparameters for this fold")
        print(gs_clf.best_params_)
        print("mean_squared_error for this fold")
        print(score)
        
    return gs_clf, method, performance, times


def fitmodel_groupfold(X, pheno, kcluster,model):
    # normalise the phenotype data by using Z-score normalization

    mean = pheno.mean()
    std = pheno.std()

    pheno_normalized = (pheno - mean) / std
    pheno = pheno_normalized
    # print(pheno_normalized)
    if len(X.shape) != 2:
        raise ValueError('X is not a 2-dimensional array.')

    sample_names = X.index 
    # Compute Hamming distance matrix
    # Compute the Hamming distance
    Y = pdist(X.values, 'hamming')
    # Perform hierarchical clustering
    # Generate the linkage matrix
    Z = linkage(Y, method='ward')

    clusters = fcluster(Z, kcluster, criterion='maxclust')
    # sample_labels = {name: label for name, label in zip(sample_names, clusters)}

    # 5 fold for groupfold
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(sample_names, pheno, clusters)
    
    train_index, test_index = next(group_kfold.split(sample_names, pheno, clusters))
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = pheno.iloc[train_index]
    y_test = pheno.iloc[test_index]

    model.fit(X_train, y_train)

    # predict resistance in test set
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    performance = score
    print("mean_squared_error for this fold")
    print(score)

    return model, performance


def sbplot(X, pheno, estimator, parameters, modelname, method, performance, times) :
    results = []
    
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()
        
        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
        
        # perform grid search to identify best hyper-parameters
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        gs_clf.fit(X_train, y_train)
        
        # predict resistance in test set
        y_pred = gs_clf.predict(X_test)
        
        results.append([y_test, y_pred])
        
    return results


