import pandas as pd
import numpy as np
import time
from os import path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm_notebook as tqdm
from keras.layers import Dense, Lambda, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from keras.initializers import he_normal, he_uniform
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, ConvLSTM2D, BatchNormalization, RepeatVector, Conv2D
from keras.regularizers import l1
from keras import backend as K
from pymove.models import metrics
from pymove.processing import trajutils



class TulvaeClassier(object):

    def __init__(self, 
                max_lenght,
                num_classes,
                vocab_size,
                rnn_units=100,
                dropout=0.5,
                embedding_size=250,
                z_values=100,
                stack=1):

        print('\n\n######               Create TULVAE model            ########\n')
        print('... max_lenght: {}\n... vocab_size: {}\n... classes: {}'.format(max_lenght, vocab_size, num_classes))
        
        #### variables locals ##
        input_model = []
        embedding_layers = []
        hidden_input = []
        hidden_dropout  = []
        
        #Input
        input_model= Input(shape=(max_lenght,), name='spatial_poi') 
        aux = RepeatVector(1)(input_model)

        # Embedding
        embedding_layer = Embedding(input_dim = vocab_size, output_dim = embedding_size,name='embedding_poi', input_length=max_lenght)(input_model)
        
        # Encoding
        encoder_lstm = Bidirectional(LSTM(rnn_units))(embedding_layer)
        encoder_lstm_dropout = Dropout(dropout)(encoder_lstm)

        # Latent
        z_mean = Dense(z_values)(encoder_lstm_dropout)
        z_log_sigma = Dense(z_values)(encoder_lstm_dropout)
        z = Lambda(sampling, output_shape=(z_values,))([z_mean, z_log_sigma,aux])

        # Decoding
        decoder_lstm = Bidirectional(LSTM(rnn_units))(RepeatVector(2)(z))
        decoder_lstm_dropout = Dropout(dropout)(decoder_lstm)

        #Output       
        output_model = Dense(num_classes, activation='softmax')(decoder_lstm_dropout)
    
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
            
            
            print('\n\n########      Compiling TULVAE Model    #########')
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metric)
            
            early_stop = EarlyStopping(monitor='val_acc',
                                    min_delta=0, 
                                    patience=50, 
                                    verbose=0, # without print 
                                    mode='auto',
                                    restore_best_weights=True)
            
            print('... Defining checkpoint')
            if save_model == True:
                modelname = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'_tuvae.h5'
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
    def predict(self, X_test, y_test):
            print('\n\n#######    Predict on validation dataset    ######')
            y_pred = self.model.predict(X_test)
            y_pred = y_pred.argmax(axis=1)
            print('... generating Classification Report')
            classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
            return classification_report

    def free(self):
        print('\n\n#######     Cleaning TULVAE model      #######')
        print('... Free memory')
        start_time = time.time()
        K.clear_session()
        print('... total_time: {}'.format(time.time()-start_time))

def sampling_error(args):
    z_mean, z_log_sigma,aux = args
    bs = aux.shape[0]
    if(bs == None):
        epsilon = K.random_normal(shape=(1, 100),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon
    else:
        epsilon = K.random_normal(shape=(bs, 100),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon

def sampling(args):
    z_mean, z_log_sigma,aux = args
    bs = aux.shape[0]
    if(bs == None):
        epsilon = K.random_normal(shape=(1, z_mean.shape[1]),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon
    else:
        epsilon = K.random_normal(shape=(bs, z_mean.shape[1]),mean=0., stddev=1)
        return z_mean + z_log_sigma * epsilon

#(parameters['max_lenght'],parameters['num_classes'],parameters['vocab_size']['poi'],rnn_units=64,dropout=0.5,embedding_size=100,z_values=100) 
#plot_model(model_TULER.model,show_shapes=True,show_layer_names=True)
#model_TULER.fit(X_train[0],y_train,X_val[0],y_val,batch_size=64,epochs=1000,learning_rate=0.005)