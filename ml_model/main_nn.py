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
from Machinelearning import fitmodel,readfile
import matplotlib.pyplot as plt
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
import sys



def main(length,outfile):
    path = './'
    # length =123456789 for using all unitigs_trim.Rtab
    phenotype ='value'

    pheno,X = readfile(length,outfile,path)

    performance = []
    method = []
    times = []

    # 160, 480 256 0.01
    hidden_units1 = 256
    hidden_units2 = 128
    hidden_units3 = 64
    learning_rate = 0.01

    # Creating model using the Sequential in tensorflow
    def build_model_using_sequential():
        model = Sequential([
        #Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        #Dropout(0.2),
        #Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        #Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error',
                     metrics=['mse'])
        return model

    optimizers = ['rmsprop', 'adam']
    #init = ['glorot_uniform', 'normal', 'uniform']
    #epochs = np.array([50, 100, 150])
    #batches = np.array([5, 10, 20])
    #keras_params = dict(optimizers =optimizers, nb_epoch=epochs, batch_size=batches)
    nn_params = {'batch_size' : [5, 10, 20],
                  'epochs' :  [20,50, 100] 
                  #'optimizer':['adam','rmsprop']
                }

    nn_mod = KerasRegressor(build_fn=build_model_using_sequential,verbose=0)

    nn_model, method, performance, times,best_params = fitmodel(X,
                                                   pheno, 
                                                   nn_mod, 
                                                   nn_params, 
                                                   "keras", 
                                                   method, 
                                                   performance,
                                                   times )



    data = {'method':method, 
            'performance':performance,
            'times':times}


    
    with open('./result/'+outfile+'_nn_len'+str(length)+'.pickle','wb') as f:
        pickle.dump(data,f)
    with open('./result/'+outfile+'_nn_len'+str(length)+'.dat','wb') as f:

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
