import pandas as pd
import numpy as np
import time
from os import path
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from pymove.processing import geoutils, trajutils
from pymove.models import lossutils
from sklearn.model_selection import train_test_split

def generate_X_y_rnn(data=[],
                        features_encoding=True,
                        y_one_hot_encodding=True,
                        label_lat='lat', 
                        label_lon='lon', 
                        label_y='label',
                        label_segment='tid'):

    print('\n\n###########      DATA PREPARATION        ###########\n')
    input_total = len(data)
    assert (input_total > 0) & (input_total <=3), "ERRO: data is not set or dimenssion > 3"
    
    print('... input total: {}'.format(input_total))
    start_time = time.time()
    
    if input_total > 1:
        print('... concat dataframe')
        df_ = pd.concat(data)
    else:
        print('... df_ is data')
        df_ = data[0]

    assert isinstance(df_, pd.DataFrame), "Erro: inform data as a list of pandas.Dataframe()"
    assert label_y in df_, "ERRO: Label y in not on dataframe"
    assert label_segment in df_, "ERRO: Label Segment in not on dataframe"
    
    
    features = list(df_.columns)
    num_classes = len(set(df_[label_y]))#)df_[label_y].nunique()
    max_lenght = df_.groupby(label_segment).agg({label_y:'count'}).max()[0]

    dic_parameters = {}
    dic_parameters['max_lenght'] = max_lenght
    dic_parameters['num_classes'] = num_classes


    dic_tid = {}
    for i, d in enumerate(data):
        dic_tid[i] = d[label_segment].unique()
        print('... tid_{}: {}'.format(i, len(dic_tid[i])))
    
    dic_parameters['dic_tid'] = dic_tid


    print('... col_name: {}...\n... num_classes: {}\n... max_lenght: {}'.format(features, num_classes, max_lenght))
    
    col_drop = [label_segment, label_lat, label_lon, label_y] 
    
    for c in col_drop:
        if c in features:
            print('... removing column {} of attr'.format(c))
            features.remove(c)
    
    if features_encoding == True:
        print('\n\n#####   Encoding string data to integer   ######')
        if len(features) > 0:
            dic_parameters['encode_features'] = label_encoding_df_to_rnn(df_, col=features)
        else:
            print('... encoding was not necessary')

    col_groupby = {}
    for c in features:
        col_groupby[c] = list
    col_groupby[label_y] = 'first'
    
    dic_parameters['col_groupby'] = col_groupby

    traj = df_.groupby(label_segment, sort=False).agg(col_groupby)

    print('\n\n###########      Generating y_train and y_test     ###########')       
    if y_one_hot_encodding == True:
        print('... one hot encoding on label y')
        ohe_y = OneHotEncoder()
        y = ohe_y.fit_transform(pd.DataFrame(traj[label_y])).toarray()
        dic_parameters['encode_y'] = ohe_y 
    else:
        print('... Label encoding on label y')
        le_y = LabelEncoder()
        y = np.array(le_y.fit_transform(pd.DataFrame(traj[label_y])))
        dic_parameters['encode_y'] = le_y
        
    print('... input total: {}'.format(input_total))
    if input_total == 1:
        y = np.array(y, ndmin=2)
    elif input_total == 2:
        y_train = y[:len(dic_tid[0])]
        y_test = y[len(dic_tid[0]):]
        y = []
        y.append(y_train)
        y.append(y_test)
        
    elif input_total == 3:
        y_train = y[:len(dic_tid[0])]
        y_val = y[len(dic_tid[0]):len(dic_tid[0])+len(dic_tid[1])]
        y_test = y[len(dic_tid[0])+len(dic_tid[1]):]
        y = []
        y.append(y_train)
        y.append(y_val)
        y.append(y_test)

    X = []

    dic_parameters['features'] = features    
    vocab_size = {}
    for i, ip in enumerate(dic_tid):
        X_aux = []
        for c in features:
            if c == 'geohash':
                vocab_size[c] = traj[c].iloc[0][0].shape[0] #len(traj.iloc[0][c][0]) #geohash_precision * 5
                pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                    maxlen=max_lenght, 
                    padding='pre',
                    value=0.0)

            else:
                vocab_size[c] = df_[c].max() + 1 # label_encode + 1 due to padding sequence
                pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                                    maxlen=max_lenght, 
                                    padding='pre',
                                    value=0.0)
        
            X_aux.append(pad_col)  
        X.append(X_aux)

    dic_parameters['vocab_size'] = vocab_size
    print('\n--------------------------------------\n')
    end_time = time.time()
    print('total Time: {}'.format(end_time - start_time))
    return X, y, dic_parameters

def generate_X_y_machine_learning(data=[],
                                features_encoding=True,
                                y_one_hot_encodding=True,
                                label_lat='lat', 
                                label_lon='lon', 
                                label_y='label',
                                label_segment='tid'):
    
    print('\n\n###########      DATA PREPARATION        ###########\n')
    input_total = len(data)
    assert (input_total > 0) & (input_total <=3), "ERRO: data is not set or dimenssion > 3"
    
    print('... input total: {input_total}')
    start_time = time.time()
    
    if input_total > 1:
        print('... concat dataframe')
        df_ = pd.concat(data)
    else:
        print('... df_ is data')
        df_ = data[0]
    
    assert isinstance(df_, pd.DataFrame), "Erro: inform data as a list of pandas.Dataframe()"
    assert label_y in df_, "ERRO: Label y in not on dataframe"
    assert label_segment in df_, "ERRO: Label Segment in not on dataframe"
    
    features = list(df_.columns)
    num_classes = len(set(df_[label_y]))#)df_[label_y].nunique()
    max_lenght = df_.groupby(label_segment).agg({label_y:'count'}).max()[0]
    dic_parameters = {}

    dic_tid = {}
    for i, d in enumerate(data):
        dic_tid[i] = d[label_segment].unique()
        print('... tid_{}: {}'.format(i, len(dic_tid[i])))
    
    print('... col_name: {}...\n... num_classes: {}\n... max_lenght: {}'.format(features, num_classes, max_lenght))
    
    col_drop = [label_segment, label_lat, label_lon, label_y] 
    
    for c in col_drop:
        if c in features:
            print('... removing column {} of attr'.format(c))
            features.remove(c)
    
    if features_encoding == True:
        print('\n\n#####   Encoding string data to integer   ######')
        if len(features) > 0:
            dic_parameters['encode_features'] = label_encoding_df_to_rnn(df_, col=features)
        else:
            print('... encoding was not necessary')
    
    col_groupby = {}
    for c in features:
        col_groupby[c] = list
    col_groupby[label_y] = 'first'
    
    traj = df_.groupby(label_segment, sort=False).agg(col_groupby)

    print('\n\n###########      Generating y_train and y_test     ###########')       
    if y_one_hot_encodding == True:
        print('... one hot encoding on label y')
        ohe_y = OneHotEncoder()
        y = ohe_y.fit_transform(pd.DataFrame(traj[label_y])).toarray()
        dic_parameters['encode_y'] = ohe_y 
    else:
        print('... Label encoding on label y')
        le_y = LabelEncoder()
        y = np.array(le_y.fit_transform(pd.DataFrame(traj[label_y])))
        dic_parameters['encode_y'] = le_y
    
    print('... input total: {}'.format(input_total))
    if input_total == 1:
        y = np.array(y, ndmin=2)
    elif input_total == 2:
        y_train = y[:len(dic_tid[0])]
        y_test = y[len(dic_tid[0]):]
        y = []
        y.append(y_train)
        y.append(y_test)
        
    elif input_total == 3:
        y_train = y[:len(dic_tid[0])]
        y_val = y[len(dic_tid[0]):len(dic_tid[0])+len(dic_tid[1])]
        y_test = y[len(dic_tid[0])+len(dic_tid[1]):]
        y = []
        y.append(y_train)
        y.append(y_val)
        y.append(y_test)

    X = []
    
    print('\n\n###########      Generating X_Train and X_Test     ###########') 
    for i, ip in enumerate(dic_tid):
        X_aux = []
        for c in features:
            pad_col = pad_sequences(traj.loc[dic_tid[i], c], 
                                    maxlen=max_lenght, 
                                    padding='pre',
                                    value=0.0)
        
            X_aux.append(pad_col) 
        X.append(np.concatenate(X_aux, axis=1))
    
    dic_parameters['features'] = features

    print('\n--------------------------------------\n')
    end_time = time.time()
    print('total Time: {}'.format(end_time - start_time))
    return X, y, dic_parameters
    
def label_encoding_df_to_rnn(df_, col=[]): 
    if len(col) == 0:
        print('... if col is empty, than col equal to df_columns')
        col = df_.columns
    
    assert set(col).issubset(set(df_.columns)), "Erro: some columns is not exist in df"
    label_encode = {}
    
    for colname in col:
        if not isinstance(df_[colname].iloc[0], np.ndarray):
            print('... encoding: {}'.format(colname))
            le = LabelEncoder()
            df_[colname] = le.fit_transform(df_[colname])
            label_encode[colname] = le
    return label_encode

def dencoding_df_to_rnn(df_, label_encode):
    for le in list(label_encode.keys()):
        print('decoding le: {}'.format(le))
        df_[le] = label_encode[le].inverse_transform(df_[le])

def split_traj_train_test(df_, label_id='id', label_tid ='tid', test_size=0.3, random_state=42):
    
    print('\n#######          Split trajectory to each id      #######')
    train_set = []
    test_set = []
    
    
    if df_.index.name is None:
        print('...Set {} as index to a higher peformance'.format(label_tid))
        df_.set_index(label_tid, inplace=True)

    ids = df_[label_id].unique()
    
    for ids in tqdm(ids):
        all_tid = df_[df_[label_id] == ids].index.unique()
        
        if len(all_tid) <= 1:
            train_set = np.concatenate((train_set, all_tid))
        else:
            train, test = train_test_split(all_tid, test_size=test_size, random_state=random_state)
            
            train_set = np.concatenate((train_set, train))
            test_set = np.concatenate((test_set, test))
    
    print('...Reset index...')
    df_.reset_index(inplace=True)
    
    df_train = df_[df_[label_tid].isin(train_set)]
    df_teste = df_[df_[label_tid].isin(test_set)]
    
    print('... There are {} IDS no train and {} IDS no teste'.format(df_train[label_id].nunique(), df_teste[label_id].nunique()))
    print('... {} traj no train and {} traj no teste'.format(df_train[label_tid].nunique(), df_teste[label_tid].nunique()))
       
    return (df_train, df_teste)