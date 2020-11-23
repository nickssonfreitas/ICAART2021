import pandas as pd
import numpy as np
import time
from os import path
from datetime import datetime
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Dense, LSTM, GRU, Bidirectional, Concatenate, Add, Average, Embedding, Dropout, Input
from keras.initializers import he_normal, he_uniform
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l1
from keras import backend as K
from pymove.models import metrics
from pymove.processing import trajutils


class DeepeST(object):
    
    def __init__(self, 
                max_lenght,
                num_classes,
                vocab_size = {},
                rnn='lstm',
                rnn_units=100,
                merge_type = 'concat',
                dropout_before_rnn=0.5,
                dropout_after_rnn=0.5,
                embedding_size = {}):
            
        start_time = time.time()

        print('\n\n##########           CREATE A DEEPEST MODEL       #########\n')
        print('... max_lenght: {}\n... vocab_size: {}\n... classes: {}'.format(max_lenght, vocab_size, num_classes))
        self.vocab_size = vocab_size
        self.col_name = list(vocab_size.keys())
        self.max_lenght = max_lenght
        input_model = []
        embedding_layers = []
        hidden_input = []
        hidden_dropout  = []

        
        if not isinstance(embedding_size, dict):
            embbeding_default = embedding_size
            print('... embedding size dictionary is empty')
            print('... setting embedding size default: {}'.format(embbeding_default))
            embedding_size = dict(zip(self.col_name, np.full(len(self.col_name), embbeding_default)))
        
        assert set(vocab_size) == set(embedding_size), "ERRO: embedding size is different from vocab_size"

        assert len(embedding_size) > 0, "embedding size was not defined"

        print('\n\n#######         DATA TO MODEL CREATION       #######\n')
        print('... features to input: {}'.format(self.col_name))
        print('... embedding_size: {}'.format(embedding_size))
        print('... max_lenght: {}'.format(max_lenght))

        print('\n\n###########      Building Input and Embedding Layers      ###########') 
        for c in tqdm(self.col_name):
            print('... creating layer to column : {}'.format(c))
            print('... vocab_size to column {}: {}'.format(c, self.vocab_size[c]))
            i_model= Input(shape=(self.max_lenght,), 
                            name='Input_{}'.format(c)) 
            e_output_ = Embedding(input_dim = self.vocab_size[c], 
                                output_dim = embedding_size[c], 
                                name='Embedding_{}'.format(c), 
                                input_length=self.max_lenght)(i_model)

            input_model.append(i_model)  
            embedding_layers.append(e_output_)             
            
    
        print('... defining merge type on embedding as {}'.format(merge_type))
        # MERGE Layer
        if len(embedding_layers) == 1:
            hidden_input = embedding_layers[0]
        elif merge_type == 'add':
            hidden_input = Add()(embedding_layers)
        elif merge_type == 'avg':
            hidden_input = Average()(embedding_layers)
        else:
            hidden_input = Concatenate(axis=2)(embedding_layers) #Lucas fez assim
            #hidden_layers = Concatenate()(embedding_layers)#eu fazia assim 92,25 sparse_val_acc

         #DROPOUT before RNN
        #if self.dropout_before_rnn > 0:
        print('... creating a dropout Layer before RNN using {}'.format(dropout_before_rnn))
        hidden_dropout = Dropout(dropout_before_rnn)(hidden_input)
    
        print('\n\n###### Creating a recurrent neural network ####\n')
        # Recurrent Neural Network Layer
        # https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell
        if rnn == 'bilstm':
            print('... creating a BiLSTM')
            rnn_cell = Bidirectional(LSTM(units=rnn_units, recurrent_regularizer=l1(0.02)))(hidden_dropout)
            #input_shape=(embedding_size[cy_true]*len(embedding_layers),))(hidden_dropout)    
        else:
            print('... creating a LSTM')
            rnn_cell = LSTM(units=rnn_units, recurrent_regularizer=l1(0.02))(hidden_dropout)
            #input_shape=(embedding_size[c]*len(embedding_layers),))(hidden_dropout)

        print('... creating a dropout Layer after RNN using {}'.format(dropout_after_rnn))
        rnn_dropout = Dropout(dropout_after_rnn)(rnn_cell)
        
        #https://keras.io/initializers/#randomnormal
        output_model = Dense(num_classes, 
                            kernel_initializer=he_uniform(),
                            activation='softmax')(rnn_dropout)

        #– Encoding the labels as integers and using the sparse_categorical_crossentropy asloss function
        self.model = Model(inputs=input_model, outputs=output_model)
        print('... a deepest model was built')

        print('\n--------------------------------------\n')
        end_time = time.time()
        print('total Time: {}'.format(end_time - start_time))

    def fit(self, 
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64,
            epochs=1000,
            monitor='val_acc',
            min_delta=0, 
            patience=30, 
            verbose=0,
            baseline=0.5,
            optimizer = 'ada',
            learning_rate = 0.001,
            mode = 'max',
            new_metrics=None,
            save_model=False,
            modelname='',
            save_best_only=True,
            save_weights_only=False,
            log_dir=None,
            loss="CCE",
            loss_parameters={'gamma': 0.5, 'alpha': 0.25}):
            
            print('\n\n##########      FIT DEEPEST MODEL       ##########')
                       
            assert (y_train.ndim == 1) |  (y_train.ndim == 2), "ERRO: y_train dimension is incorrect"            
            assert (y_val.ndim == 1) |  (y_val.ndim == 2), "ERRO: y_test dimension is incorrect"
            assert (y_train.ndim == y_val.ndim), "ERRO: y_train and y_test have differents dimension"

            if y_train.ndim == 1:
                y_one_hot_encodding = False
            elif y_train.ndim == 2:
                y_one_hot_encodding = True
          

            if y_one_hot_encodding == True:
                print('... categorical_crossentropy was selected')
                loss = ['categorical_crossentropy'] #categorical_crossentropy
                my_metrics = ['acc', 'top_k_categorical_accuracy'] 
            else:
                print('... sparse categorical_crossentropy was selected')
                loss = ['sparse_categorical_crossentropy'] #sparse_categorical_crossentropy
                my_metrics = ['acc', 'sparse_top_k_categorical_accuracy']  
           
            if new_metrics is not None:
                my_metrics = new_metrics + my_metrics

            if optimizer == 'ada':
                print('... Optimizer was setting as Adam')
                optimizer = Adam(lr=learning_rate)
            else:
                print('... Optimizer was setting as RMSProps')
                optimizer = RMSprop(lr=learning_rate)

            print('\n\n########      Compiling DeepeST Model    #########')
            self.model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=my_metrics)

            early_stop = EarlyStopping(monitor=monitor,
                                        min_delta=min_delta, 
                                        patience=patience, 
                                        verbose=verbose, # without print 
                                        mode=mode,
                                        baseline=baseline,
                                        restore_best_weights=True)
        
        
            print('... Defining checkpoint')
            if save_model == True:
                if (not modelname) | (modelname == None):
                    modelname = 'deepeSTmodel_'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'.h5'         
                ck = ModelCheckpoint(modelname, 
                            monitor=monitor, 
                            save_best_only=save_best_only,
                            save_weights_only=save_weights_only)   
                my_callbacks = [early_stop, ck]    
            else:
                my_callbacks= [early_stop]    

            print('... Starting training')
            self.history = self.model.fit(X_train, y_train,
                                        epochs=epochs,
                                        callbacks=my_callbacks,
                                        validation_data=(X_val, y_val),
                                        verbose=1,
                                        shuffle=True,
                                        use_multiprocessing=True,          
                                        batch_size=batch_size)

    def predict(self, 
                X_test,
                y_test, 
                verbose=0,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        
                   
        print('\n\n##########      PREDICT DEEPEST MODEL       ##########\n')
        assert (y_test.ndim == 1) |  (y_test.ndim == 2), "ERRO: y_train dimension is incorrect"       

        if y_test.ndim == 1:
            y_one_hot_encodding = False
        elif y_test.ndim == 2:
            y_one_hot_encodding = True

        y_pred = np.array(self.model.predict(X_test))
        
        if y_one_hot_encodding == True:
            argmax = np.argmax(y_pred, axis=1)
            y_pred = np.zeros(y_pred.shape)
            for row, col in enumerate(argmax):
                y_pred[row][col] = 1
        else:
            y_pred = y_pred.argmax(axis=1)
        
        print('... generate classification Report')  
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report
       

    def summary(self):
        if self.model is None:
           print('Erro: model is not exist') 
        else:
            self.model.summary()

    def get_params(self):
        print('get parameterns')
    
    def score(self, X, y):
        print('Score')
    
    def free(self):
        print('\n\n#######     Cleaning DeepeST model      #######')
        print('... Free memory')
        start_time = time.time()
        K.clear_session()
        print('... total_time: {}'.format(time.time()-start_time))