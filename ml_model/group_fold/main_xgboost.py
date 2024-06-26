#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
from Machinelearning2 import fitmodel_groupfold, readfile
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

import xgboost as xgb

def main(kcluster, length, outfile):
    path = '/home/zhoug/individual_project/'
    pheno,X = readfile(length, outfile, path)
    # Define hyperparameters
    hyperparameters = {'alpha': 1e-05, 
            'colsample_bytree': 0.6, 
            'gamma': 0.05,
            'learning_rate': 0.1,
            'max_depth': 2,
            'subsample': 0.6}
    model = xgb.XGBRegressor(**hyperparameters)

    model, performance = fitmodel_groupfold(X, pheno, kcluster, model)
    now = datetime.now()
    formatted_now = now.strftime('%m%d%H%M')
    with open(f"xgb_groupfold_kcluster{kcluster}All{outfile}{formatted_now}.txt", "w") as file:
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
