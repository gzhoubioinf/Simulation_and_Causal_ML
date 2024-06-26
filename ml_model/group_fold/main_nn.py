#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from Machinelearning2 import fitmodel_groupfold, readfile
from datetime import datetime

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

def main(kcluster, length, outfile):
    path = '/home/zhoug/individual_project/'
    pheno,X = readfile(length, outfile, path)
    # Define hyperparameters

    # 160, 480 256 0.01
    hidden_units1 = 256
    hidden_units2 = 128
    hidden_units3 = 64
    learning_rate = 0.01
    

    # Creating model using the Sequential in tensorflow
    def build_model_using_sequential():
        model = Sequential([
        #Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        #Dropout(0.2),
        Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error',
                     metrics=['mse'])
        return model
    hyperparameters = {'batch_size': 20, 'epochs': 100} 



    model = KerasRegressor(build_fn=build_model_using_sequential,verbose=0,**hyperparameters)
    model, performance = fitmodel_groupfold(X, pheno, kcluster, model)
    now = datetime.now()
    formatted_now = now.strftime('%m%d%H%M')
    with open(f"nn_groupfold_kcluster{kcluster}All{outfile}_{formatted_now}.txt", "w") as file:
        file.write(str(hyperparameters))
        file.write(f"\nkcluster={kcluster}\n")
        file.write(f"length={length}\n")
        file.write(outfile + '\n')
        file.write(f'score={performance}')


if __name__ == '__main__':
    kcluster = int(sys.argv[1])
    length = int(sys.argv[2])
    outfile = sys.argv[3]
    main(kcluster, length, outfile)
