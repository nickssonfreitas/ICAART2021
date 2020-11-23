import pandas as pd
import time
import xgboost as xgb
from pymove.models import metrics
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from pymove.core import video as vi

class XGBoostClassifier(object):
    
    def __init__(self, 
                max_depth=13, 
                lr=0.005, 
                n_estimators=2000, 
                tree_method='gpu_hist',
                n_jobs=-1,
                gamma=0.1, 
                colsample_bytree=0.5,
                subsample=0.2, 
                l1=0.01, 
                l2=0.0, 
                random_state=42, 
                early_stopping_rounds=10):
        
        start_time = time.time()

        print('\n\n###########      Building XGBoost Model      ###########') 

        if tree_method == 'auto':
            tree_method = 'hist' if vi.nvidia_gpu_count() == 0 else 'gpu_hist'
        else:
            tree_method = 'gpu_hist'

        print('... tree_method: {}'.format(tree_method))

        self.model = xgb.XGBClassifier(max_depth=max_depth, 
                                  learning_rate=lr,
                                  n_estimators=n_estimators, 
                                  tree_method=tree_method,
                                  subsample=subsample, 
                                  gamma=gamma,
                                  reg_alpha_l1=l1, 
                                  reg_lambda_l2=l2,
                                  n_jobs=-1, 
                                  early_stopping_rounds=early_stopping_rounds,
                                  random_state=random_state, 
                                  objective='multi:softmax')
        print('\n--------------------------------------\n')
        end_time = time.time()
        print('total Time: {}'.format(end_time - start_time))
        
    def fit(self, 
            X_train, 
            y_train, 
            X_val,
            y_val, 
            loss='merror', 
            early_stopping_rounds=10, 
            verbose=True):
        
        assert (loss == 'merror') | (loss == 'mlogloss'), "ERRO: invalid loss, set loss as mlogloss or merror"    

        eval_set = [(X_train, y_train), (X_val, y_val)]
        print('... Training model...\n')
        self.model.fit(X_train, y_train, 
                      eval_set=eval_set, 
                      early_stopping_rounds=early_stopping_rounds,
                      eval_metric=loss, #mlogloss or merror
                      verbose=verbose) 
        
    def predict(self,                 
                X_test,
                y_test,
                batch_size=64):
     
        print('... Predict on test dataset')
        y_pred = self.model.predict(X_test) 
        print('... Generating Classification Report')
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report