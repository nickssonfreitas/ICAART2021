import pandas as pd
import numpy as np
import mplleaflet as mpl
import traceback
import time
import gc
import os
import itertools
import collections
import itertools
from os import path
from tqdm.notebook import tqdm
from glob import glob
from joblib import load, dump
from pymove.models.classification import RandomForest as rf
from pymove.models import datautils
from pymove.core import utils
import json


paper = 'SAC'
dataset = 'gowalla' #['fousquare_nyc', 'brightkite', 'foursquare_global', gowalla,'criminal_id', 'criminal_activity']
file_train = 'data/{}/train.csv.gz'.format(dataset)
file_val = 'data/{}/val.csv.gz'.format(dataset)
file_test = 'data/{}/test.csv.gz'.format(dataset)
dir_validation = '{}/{}/randomforest/validation/'.format(paper, dataset)
dir_evaluation = '{}/{}/randomforest/'.format(paper, dataset)

df_train = pd.read_csv(file_train)
df_val = pd.read_csv(file_val)
df_test = pd.read_csv(file_test)
df = pd.concat([df_train, df_val, df_test])


# ## GET TRAJECTORIES

features = ['tid','label','hour','day','poi','indexgrid30']

data = [df_train[features], df_val[features], df_test[features]]
X, y, dic_parameters = datautils.generate_X_y_machine_learning(data= data,
                                        features_encoding=True,       
                                        y_one_hot_encodding=False)

X_train = X[0] 
X_val = X[1]
X_test = X[2]
y_train = y[0] 
y_val = y[1]
y_test = y[2]


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]# Number of trees in random forest
max_depth = [int(x) for x in np.linspace(20, 40, num = 3)] # Maximum number of levels in tree
min_samples_split =  [2, 5, 10] # Minimum number of samples required to split a node
min_samples_leaf =  [1, 2, 4] # Minimum number of samples required at each leaf node
max_features= ['auto', 'sqrt'] # Number of features to consider at every split
bootstrap =  [True, False] # Method of selecting samples for training each tree
n_jobs=10
random_state = 42
verbose = 1


total = len(n_estimators) * len(max_depth) * len(min_samples_split) * len(min_samples_leaf) * len(max_features) *        len(bootstrap)
print('There are {} iteration'.format(total))

for c in tqdm(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,                                bootstrap), total=total):
    ne=c[0]
    md=c[1]
    mss=c[2]
    msl=c[3]
    mf=c[4]
    bs=c[5]
              
    filename = dir_validation + 'randomforest-{}-{}-{}-{}-{}-{}-{}.csv'.format(                                                                                    ne, md, mss, msl, mf,                                                                                     bs, features)
    
    if not path.exists(dir_validation):
        print('this directory {} is not exist'.format(dir_validation))
        break
    elif path.exists(filename):
        print('skip ---> {}\n'.format(filename))
    else:
        print('Creating model...')
        print(filename)
        
        RF = rf.RFClassifier(n_estimators=ne,
                             max_depth=md,
                             max_features=mf,
                             min_samples_split=mss,
                             min_samples_leaf=msl,
                             bootstrap=bs,
                             random_state=random_state,
                             verbose=verbose,
                             n_jobs=n_jobs)

        RF.fit(X_train, y_train)
        
        validation_report = RF.predict(X_val, y_val)
        
        validation_report.to_csv(filename, index=False)
        

        RF.free()


files = utils.get_filenames_subdirectories(dir_validation)

data = []
marksplit = '-'
for f in files:
    df_ = pd.read_csv(f)
    f = f.split('randomforest')[-1]
    df_['ne']= f.split(marksplit)[1]
    df_['md']= f.split(marksplit)[2]
    df_['mss']= f.split(marksplit)[3]
    df_['msl']= f.split(marksplit)[4]
    df_['mf']= f.split(marksplit)[5]
    df_['bs']= f.split(marksplit)[6]
    df_['feature']= f.split(marksplit)[7].split('.csv')[0]
    data.append(df_)
    
df_result = pd.concat(data)
df_result.reset_index(drop=True, inplace=True)




df_result.sort_values('acc', ascending=False, inplace=True)
df_result.head(5)

model = 0
ne = int(df_result.iloc[model]['ne'])
md = int(df_result.iloc[model]['md'])
mss = int(df_result.iloc[model]['mss'])
msl = int(df_result.iloc[model]['msl'])
mf = df_result.iloc[model]['mf']
bs = utils.str_to_bool(df_result.iloc[0]['bs'])



filename = dir_evaluation + 'eval_randomforest-{}-{}-{}-{}-{}-{}-{}.csv'.format(ne, md, mss, msl, mf, bs, features)

print("filename: {}".format(filename))

if not path.exists(filename):
    print('Creating a model to test set')
    evaluate_report = []
    rounds = 10


    for e in tqdm(range(rounds)):
        print('Rounds {} de {}'.format(e, rounds))
        RF = rf.RFClassifier(n_estimators=ne,
                        max_depth=md,
                        min_samples_split=mss,
                        min_samples_leaf=msl,
                        max_features=mf,
                        bootstrap=bs,
                        random_state=e,
                        verbose=verbose,
                        n_jobs=n_jobs)


        RF.fit(X_train, y_train)
        #RF.fit(X_val, y_val)
        evaluate_report.append(RF.predict(X_test, y_test))
        
        RF.free()

    evaluate_report = pd.concat(evaluate_report)
    evaluate_report.to_csv(filename, index=False)
else:
    print('... there are a model on disk to this parameters')

