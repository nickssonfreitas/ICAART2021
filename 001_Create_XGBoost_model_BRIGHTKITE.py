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
from pymove.models import datautils
from pymove.core import utils
import json
from pymove.models.classification import XGBoost as xg

paper = 'SAC'
dataset = 'brightkite' #['fousquare_nyc', 'brightkite', 'foursquare_global', gowalla,'criminal_id', 'criminal_activity']
file_train = 'data/{}/train.csv.gz'.format(dataset)
file_val = 'data/{}/val.csv.gz'.format(dataset)
file_test = 'data/{}/test.csv.gz'.format(dataset)
dir_validation = '{}/{}/xgboost/validation/'.format(paper, dataset)
dir_evaluation = '{}/{}/xgboost/'.format(paper, dataset)


df_train = pd.read_csv(file_train)
df_val = pd.read_csv(file_val)
df_test = pd.read_csv(file_test)
df = pd.concat([df_train, df_val, df_test])


### Get trajectory to XGBOOST
features = ['tid','label','hour','day','poi','indexgrid30']

data = [df_train[features], df_val[features], df_test[features]]
X, y, dic_parameters = datautils.generate_X_y_machine_learning(data=data,
                                                               features_encoding=True,
                                                               y_one_hot_encodding=False)

X_train = X[0] 
X_val = X[1]
X_test = X[2]
y_train = y[0] 
y_val = y[1]
y_test = y[2]

n_estimators = [2000]
max_depth = [3, 5]
learning_rate = [0.01]
gamma = [0.0, 1, 5]
subsample = [0.1, 0.2, 0.5, 0.8]
colsample_bytree = [0.5 , 0.7]
reg_alpha_l1 = [1.0]#[0.0, 0.01, 1.0]
reg_lambda_l2 = [100]#[0.0, 1.0, 100]
eval_metric = ['merror']#, 'mlogloss'] #merror #(wrong cases)/#(all cases) Multiclass classification error // mlogloss:
tree_method = 'auto' #   
esr = [20]

total = len(n_estimators) * len(max_depth) * len(learning_rate) * len(gamma) * len(subsample) *         len(colsample_bytree) * len(reg_alpha_l1) * len(reg_lambda_l2) * len(eval_metric) * len(esr) 
print('There are {} iteration'.format(total))


for c in tqdm(itertools.product(n_estimators, max_depth, learning_rate, gamma, subsample,                                colsample_bytree, reg_alpha_l1, reg_lambda_l2, eval_metric,                                 esr), total=total):
    ne=c[0]
    md=c[1]
    lr=c[2]
    gm=c[3]
    ss=c[4]
    cst=c[5]
    l1=c[6]
    l2=c[7]
    loss=c[8]
    epch=c[9] 

    filename = dir_validation + 'xgboost-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(                                                                                    ne, md, lr, gm, ss,                                                                                     cst, l1, l2, loss,                                                                                     epch, features)
    
    if not path.exists(dir_validation):
        print('this directory {} is not exist'.format(dir_validation))
        break
    elif path.exists(filename):
        print('skip ---> {}\n'.format(filename))
    else:
        print('Creating model...')
        print(filename)
        
        xgboost = xg.XGBoostClassifier(n_estimators=ne,
                                       max_depth=md,
                                       lr=lr,
                                       gamma=gm,
                                       colsample_bytree=cst,
                                       subsample=ss,
                                       l1=l1,
                                       l2=l2,
                                       random_state=42,
                                       tree_method=tree_method,
                                       early_stopping_rounds=epch)

        xgboost.fit(X_train, 
                    y_train, 
                    X_val,
                    y_val,
                    loss=loss, 
                    early_stopping_rounds=epch)
        
        validation_report = xgboost.predict(X_val, y_val)
        
        validation_report.to_csv(filename, index=False)

files = utils.get_filenames_subdirectories(dir_validation)

data = []
marksplit = '-'
for f in files:
    df_ = pd.read_csv(f)
    df_['ne']=   f.split(marksplit)[1]
    df_['md']=     f.split(marksplit)[2]
    df_['lr']=     f.split(marksplit)[3]
    df_['gm'] = f.split(marksplit)[4]
    df_['ss'] = f.split(marksplit)[5]
    df_['cst'] = f.split(marksplit)[6]
    df_['l1'] = f.split(marksplit)[7]
    df_['l2'] = f.split(marksplit)[8]
    df_['loss']  = f.split(marksplit)[9]
    df_['epoch'] = f.split(marksplit)[10]
    df_['features'] = f.split(marksplit)[11].split('.csv')[0]
    data.append(df_)


df_result = pd.concat(data)
df_result.reset_index(drop=True, inplace=True)
df_result.sort_values('acc', ascending=False, inplace=True)

model = 0
ne = int(df_result.iloc[model]['ne'])
md = int(df_result.iloc[model]['md'])
lr = float(df_result.iloc[model]['lr'])
gm = float(df_result.iloc[model]['gm'])
ss = float(df_result.iloc[model]['ss'])
cst = float(df_result.iloc[model]['cst'])
l1 = float(df_result.iloc[model]['l1'])
l2 = int(df_result.iloc[model]['l2']) 
loss = df_result.iloc[model]['loss']
epch = int(df_result.iloc[model]['epoch'])

filename = dir_evaluation + 'eval_xgboost-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(ne, md, lr, gm, ss,                                                                                     cst, l1, l2, loss, epch, features)



print("filename: {}".format(filename))

if not path.exists(filename):
    print('Creating a model to test set')

    evaluate_report = []
    rounds = 10


    for e in tqdm(range(rounds)):
        print('Rounds {} de {}'.format(rounds, e))
        xgboost = xg.XGBoostClassifier(n_estimators=ne,
                                    max_depth=md,
                                    lr=lr,
                                    gamma=gm,
                                    subsample=ss,
                                    l1=l1,
                                    l2=l2,
                                    random_state=e,
                                    tree_method=tree_method,
                                    early_stopping_rounds=epch)

        xgboost.fit(X_train, 
                    y_train, 
                    X_val,
                    y_val,
                    loss=loss, 
                    early_stopping_rounds=epch)

        evaluate_report.append(xgboost.predict(X_test, y_test))
    
    evaluate_report = pd.concat(evaluate_report)
    evaluate_report.to_csv(filename, index=False)
else:
    print('... there are a model on disk to this parameters')
