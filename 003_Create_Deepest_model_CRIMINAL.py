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
from pymove.models.classification import DeepestST as DST
from pymove.models import datautils
from pymove.core import utils
import json


### 1. Read Dataset

paper = 'SAC'
dataset = 'criminal_activity' 
file_train = 'data/{}/train.csv.gz'.format(dataset)
file_val = 'data/{}/val.csv.gz'.format(dataset)
file_test = 'data/{}/test.csv.gz'.format(dataset)
dir_validation = '{}/{}/deepest/validation/'.format(paper, dataset)
dir_evaluation = '{}/{}/deepest/'.format(paper, dataset)

df_train = pd.read_csv(file_train)
df_val = pd.read_csv(file_val)#
df_test = pd.read_csv(file_test)#
df = pd.concat([df_train, df_val, df_test])


### Get Trajectories
y_one_hot_encodding=True
features = ['day','hour', 'label', 'tid', 'indexgrid30']
data = [df_train[features], df_val[features], df_test[features]]


X, y, dic_parameters = datautils.generate_X_y_rnn(data=data,
                                            features_encoding=True,
                                            y_one_hot_encodding=y_one_hot_encodding,
                                            label_y='label',
                                            label_segment='tid')



max_lenght = dic_parameters['max_lenght']
num_classes = dic_parameters['num_classes']
vocab_size = dic_parameters['vocab_size']
features = dic_parameters['features']
encode_features = dic_parameters['encode_features']
encode_y = dic_parameters['encode_y']


X_train = X[0] 
X_val = X[1]
X_test = X[2]
y_train = y[0] 
y_val = y[1]
y_test = y[2]

## GRID SEARCH PARAMETERS
rnn = ['bilstm', 'lstm']
units = [100, 200, 300, 400]
merge_type = ['concat']
dropout_before_rnn=[0]
dropout_after_rnn=[0.5]

embedding_size = [50, 100, 200, 300, 400]
batch_size = [64]
epochs = [1000]
patience = [20]
monitor = ['val_acc']

optimizer = ['ada']
learning_rate = [0.001]
loss = ['CCE']
loss_parameters = [{}]

y_ohe = y_one_hot_encodding

total = len(rnn)*len(units)*len(merge_type)*len(dropout_before_rnn)* len(dropout_after_rnn)*        len(embedding_size)* len(batch_size) * len(epochs) * len(patience) * len(monitor) *        len(optimizer) * len(learning_rate) * len(loss) * len(loss_parameters) 
print('There are {} iteration'.format(total))

count = 0

for c in tqdm(itertools.product(rnn, units, merge_type, dropout_before_rnn, dropout_after_rnn,                                embedding_size, batch_size, epochs, patience, monitor,                                 optimizer, learning_rate, loss, loss_parameters), total=total):
    
    nn=c[0]
    un=c[1]
    mt=c[2]
    dp_bf=c[3]
    dp_af=c[4]
    em_s=c[5]
    bs=c[6]
    epoch=c[7] 
    pat=c[8] 
    mon=c[9] 
    opt=c[10] 
    lr=c[11]
    ls=c[12]
    ls_p=c[13]
        
    filename = dir_validation + 'deepest-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(                                                                                    nn, un, mt, dp_bf, dp_af,                                                                                     em_s, bs, epoch, pat, mon,                                                                                     opt, lr, ls, ls_p,                                                                                     y_ohe, features)
    count += 1

    print('{}  ----------------   {}'.format(count, total))
    if not path.exists(dir_validation):
        print('this directory {} is not exist'.format(dir_validation))
        break
    elif path.exists(filename):
        print('skip ---> {}\n'.format(filename))
    else:
        print('Creating model...')
        print(filename)
        
        deepest = DST.DeepeST(max_lenght=max_lenght,
                    num_classes=num_classes,
                    vocab_size = vocab_size,
                    rnn=nn,
                    rnn_units=un,
                    merge_type = mt,
                    dropout_before_rnn=dp_bf,
                    dropout_after_rnn=dp_af,
                    embedding_size = em_s)

        deepest.fit(X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=bs,
                    epochs=epoch,
                    monitor=mon,
                    min_delta=0,
                    patience=pat,
                    verbose=0,
                    baseline=0.5,
                    optimizer=opt,
                    learning_rate=lr,
                    mode='auto',
                    new_metrics=None,
                    save_model=False,
                    modelname='',
                    save_best_only=True,
                    save_weights_only=False,
                    log_dir=None,
                    loss=ls,
                    loss_parameters=ls_p)
        
        validation_report = deepest.predict(X_val, y_val)
        validation_report.to_csv(filename, index=False)

        deepest.free()

files = utils.get_filenames_subdirectories(dir_validation)

data = []
marksplit = '-'
for f in files:
    df_ = pd.read_csv(f)
    f = f.split('deepest')[-1]
    df_['nn']= f.split(marksplit)[1]
    df_['un']= f.split(marksplit)[2]
    df_['mt']= f.split(marksplit)[3]
    df_['dp_bf']= f.split(marksplit)[4]
    df_['dp_af']= f.split(marksplit)[5]
    df_['em_s']= f.split(marksplit)[6]
    df_['bs']= f.split(marksplit)[7]
    df_['epch']= f.split(marksplit)[8]
    df_['pat']= f.split(marksplit)[9]
    df_['mon']= f.split(marksplit)[10]
    df_['opt']= f.split(marksplit)[11]
    df_['lr']= f.split(marksplit)[12]
    df_['ls']= f.split(marksplit)[13]
    df_['ls_p']= f.split(marksplit)[14]
    df_['ohe'] = y_one_hot_encodding
    
    data.append(df_)
    
df_result = pd.concat(data)
df_result = df_result[df_result['nn'] == 'lstm']
df_result.reset_index(drop=True, inplace=True)
df_result.sort_values('acc', ascending=False, inplace=True)


model = 0
nn =  df_result.iloc[model]['nn']
un =  int(df_result.iloc[model]['un'])
mt =  df_result.iloc[model]['mt']
dp_bf = float(df_result.iloc[model]['dp_bf'])
dp_af = float(df_result.iloc[model]['dp_af'])

em_s = int(df_result.iloc[model]['em_s'])

bs = int(df_result.iloc[0]['bs'])
epoch = int(df_result.iloc[model]['epch'])
pat = float(df_result.iloc[model]['pat'])
mon = df_result.iloc[model]['mon']

opt = df_result.iloc[model]['opt']
lr = float(df_result.iloc[0]['lr'])
ls = df_result.iloc[model]['ls']
ls_p = json.loads(df_result.iloc[model]['ls_p'].replace("'", "\""))

y_ohe = y_one_hot_encodding

filename = dir_evaluation + 'eval_deepest-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(nn, un, mt, dp_bf, dp_af, em_s, bs, epoch, pat, mon, opt, lr, ls, ls_p, y_ohe, features)
print("filename: {}".format(filename))

if not path.exists(filename):
    print('Creating a model to test set')
    evaluate_report = []
    rounds = 10

    for e in tqdm(range(rounds)):
        print('Rounds {} de {}'.format(e, rounds))
        
        deepest = DST.DeepeST(max_lenght=max_lenght,
                num_classes=num_classes,
                vocab_size = vocab_size,
                rnn=nn,
                rnn_units=un,
                merge_type = mt,
                dropout_before_rnn=dp_bf,
                dropout_after_rnn=dp_af,
                embedding_size = em_s)

        deepest.fit(X_train,
                y_train,
                X_val,
                y_val,
                batch_size=bs,
                epochs=epoch,
                monitor=mon,
                min_delta=0,
                patience=pat,
                verbose=0,
                baseline=None,
                optimizer=opt,
                learning_rate=lr,
                mode='auto',
                new_metrics=None,
                save_model=False,
                modelname='',
                save_best_only=True,
                save_weights_only=False,
                log_dir=None,
                loss=ls,
                loss_parameters=ls_p)
        
        evaluate_report.append(deepest.predict(X_test, y_test))
        
        deepest.free()
        
    evaluate_report = pd.concat(evaluate_report)
    evaluate_report.to_csv(filename, index=False)
else:
    print('... there are a model on disk to this parameters')

