#!/usr/bin/env python
# coding: utf-8
import sys

# set up environment
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVC
import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

# from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#import seaborn as sb
import time
import pandas

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow import keras 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import pickle
from Machinelearning import fitmodel,readfile




def main(length,outfile):
    path = './'
    # length =123456789 for using all unitigs_trim.Rtab
    phenotype ='value'

    pheno,X = readfile(length,outfile,path)
    performance = []
    method = []
    times = []

    enet = SGDRegressor(loss="squared_loss")
    enet_params =  {"max_iter": [1, 5, 10],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": np.arange(0.0, 1.0, 0.1)}    

    enet_model, method, performance, times,best_params = fitmodel(X,
                                                      pheno,
                                                      enet, 
                                                      enet_params,
                                                     "Elastic net", 
                                                      method,     
                                                      performance, 
                                                      times )
    
    data = {'enet_model':enet_model, 
            'method':method, 
            'performance':performance,
            'times':times}
            #'pred_test': pred_test }
    
    with open('./result/'+outfile+'_enet_len'+str(length)+'.pickle','wb') as f:
        pickle.dump(data,f)
    with open('./result/'+outfile+'_enet_len'+str(length)+'.dat','wb') as f:
        f.write('method:' + str(method)+'\n')
        f.write('performance:' + str(performance)+'\n')
        f.write('time:' + str(time)+'\n')
        for sc, param in zip(score, best_params):
            f.write(str(sc)+'\n')
            f.write(param+'\n')



    print('data saved')

    

if __name__ == '__main__':
    length = int(sys.argv[1])
    outfile = sys.argv[2]
    main(length,outfile)
