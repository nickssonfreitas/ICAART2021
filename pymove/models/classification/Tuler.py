import pandas as pd
import numpy as np
import time
from os import path
from datetime import datetime
from keras.layers import Dense, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from keras.initializers import he_normal, he_uniform
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm_notebook as tqdm
from keras.regularizers import l1
from pymove.models import metrics
from pymove.processing import geoutils, trajutils

class BiTulerLSTM(object):

    def __init__(self, 
                max_lenght,
                num_classes,
                vocab_size,
                rnn_units=100,
                dropout=0.5,
                embedding_size = 250,
                stack=2):

        print('... max_lenght: {}\n... vocab_size: {}\n... classes: {}'.format(max_lenght, vocab_size, num_classes))
        
        
        input_model= Input(shape=(max_lenght,), name='spatial_poi') 
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = embedding_size, 
                              name='embedding_poi', input_length=max_lenght)(input_model)
        

        rnn_cell = Bidirectional(LSTM(units=rnn_units))(embedding_layer)
                            
        hidden_dropout = Dropout(dropout)(rnn_cell)
        output_model = Dense(num_classes, activation='softmax')(hidden_dropout)
        
        self.model = Model(inputs=input_model, outputs=output_model)
                  
    def fit(self, 
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64, 
            epochs=1000,
            learning_rate=0.00005, 
            save_model=False,
            save_best_only=False,
            save_weights_only=False):
        
        ## seting parameters
        optimizer = Adam(lr=learning_rate)
        loss = ['sparse_categorical_crossentropy']
        metric = ['acc']  
        monitor='val_acc'
        
        
        print('\n\n########      Compiling TULER Model    #########')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        
        early_stop = EarlyStopping(monitor='val_acc',
                                   min_delta=0, 
                                   patience=50, 
                                   verbose=0, # without print 
                                   mode='auto',
                                   restore_best_weights=True)
        
        print('... Defining checkpoint')
        if save_model == True:
            modelname = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_tuler.h5'
            ck = ModelCheckpoint(modelname, 
                                 monitor=monitor, 
                                 save_best_only=save_best_only,
                                 save_weights_only=save_weights_only)   
            my_callbacks = [early_stop, ck]    
        else:
            my_callbacks= [early_stop]   

            # if log_dir is not None:
            #     assert path.exists(log_dir) == True, 'ERRO: log_dir is not exist on disk' 
            #     my_callbacks.append(TensorBoard(log_dir=log_dir))

            # customizar callback deepest
            #https://www.tensorflow.org/guide/keras/custom_callback
        print('... Starting training')
        self.history = self.model.fit(X_train, 
                                    y_train,
                                    epochs=epochs,
                                    callbacks=my_callbacks,
                                    validation_data=(X_val, y_val),
                                    verbose=1,
                                    shuffle=True,
                                    use_multiprocessing=True,          
                                    batch_size=batch_size)

        # print('\n\n#######    Predict on validation dataset    ######')
        # y_pred = self.model.predict(X_val)
        # y_pred = y_pred.argmax(axis=1)
        # print('... generating Classification Report')
        # self.classification_report_fit = metrics.compute_acc_acc5_f1_prec_rec(y_val, y_pred)

    def predict(self, X_test, y_test):
        print('\n\n#######    Predict on validation dataset    ######')
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.argmax(axis=1)
        print('... generating Classification Report')
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report

    def free(self):
        print('\n\n#######     Cleaning TULER model      #######')
        print('... Free memory')
        start_time = time.time()
        K.clear_session()
        print('... total_time: {}'.format(time.time()-start_time))
    
class TulerStackLSTM(object):

    def __init__(self, 
                max_lenght,
                num_classes,
                vocab_size,
                rnn_units=100,
                dropout=0.5,
                embedding_size = 250,
                stack=2):

        print('... max_lenght: {}\n... vocab_size: {}\n... classes: {}'.format(max_lenght, vocab_size, num_classes))
        
        #### variables locals ##
        input_model = []
        embedding_layers = []
        hidden_input = []
        hidden_dropout  = []
        
        input_model= Input(shape=(max_lenght,), name='spatial_poi') 
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = embedding_size, 
                              name='embedding_poi', input_length=max_lenght)(input_model)
        
        print('... Creating stack to TULER')
        if stack > 1:
            rnn_cell = LSTM(units=rnn_units, return_sequences=True)(embedding_layer)
            for i in range(1, stack-1):
                rnn_cell = LSTM(units=rnn_units, return_sequences=True)(rnn_cell)
            rnn_cell = LSTM(units=rnn_units)(rnn_cell)
        else:
            rnn_cell = LSTM(units=rnn_units)(embedding_layer)
                            
        hidden_dropout = Dropout(dropout)(rnn_cell)
        output_model = Dense(num_classes, activation='softmax')(hidden_dropout)
        
        self.model = Model(inputs=input_model, outputs=output_model)
                  
    def fit(self, 
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64, 
            epochs=1000,
            learning_rate=0.00005, 
            save_model=False,
            save_best_only=False,
            save_weights_only=False):
        
        ## seting parameters
        optimizer = Adam(lr=learning_rate)
        loss = ['sparse_categorical_crossentropy']
        metric = ['acc']  
        monitor='val_acc'
        
        
        print('\n\n########      Compiling TULER Model    #########')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)
        
        early_stop = EarlyStopping(monitor='val_acc',
                                   min_delta=0, 
                                   patience=50, 
                                   verbose=0, # without print 
                                   mode='auto',
                                   restore_best_weights=True)
        
        print('... Defining checkpoint')
        if save_model == True:
            modelname = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_tuler.h5'
            ck = ModelCheckpoint(modelname, 
                                 monitor=monitor, 
                                 save_best_only=save_best_only,
                                 save_weights_only=save_weights_only)   
            my_callbacks = [early_stop, ck]    
        else:
            my_callbacks= [early_stop]   

            # if log_dir is not None:
            #     assert path.exists(log_dir) == True, 'ERRO: log_dir is not exist on disk' 
            #     my_callbacks.append(TensorBoard(log_dir=log_dir))

            # customizar callback deepest
            #https://www.tensorflow.org/guide/keras/custom_callback
        print('... Starting training')
        self.history = self.model.fit(X_train, 
                                    y_train,
                                    epochs=epochs,
                                    callbacks=my_callbacks,
                                    validation_data=(X_val, y_val),
                                    verbose=1,
                                    shuffle=True,
                                    use_multiprocessing=True,          
                                    batch_size=batch_size)

        # print('\n\n#######    Predict on validation dataset    ######')
        # y_pred = self.model.predict(X_val)
        # y_pred = y_pred.argmax(axis=1)
        # print('... generating Classification Report')
        # self.classification_report_fit = metrics.compute_acc_acc5_f1_prec_rec(y_val, y_pred)

    def predict(self, X_test, y_test):
        print('\n\n#######    Predict on validation dataset    ######')
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.argmax(axis=1)
        print('... generating Classification Report')
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report

    def free(self):
        print('\n\n#######     Cleaning TULER model      #######')
        print('... Free memory')
        start_time = time.time()
        K.clear_session()
        print('... total_time: {}'.format(time.time()-start_time))
    
