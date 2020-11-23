import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing.sequence import pad_sequences
from pymove.models import metrics

class RFClassifier(object):
    def __init__(self, 
                n_estimators=2000, 
                max_depth=13, 
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='auto',
                bootstrap=True,
                n_jobs=-1,
                verbose=0,
                random_state=42):
                 
        print('\n\n##########           CREATE A RANDON FOREST  MODEL       #########\n')
        start_time = time.time()


        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                max_features=max_features,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                bootstrap=bootstrap,
                                random_state=random_state,
                                verbose=verbose,
                                n_jobs=n_jobs)
        print('\n--------------------------------------\n')
        end_time = time.time()
        print('total Time: {}'.format(end_time - start_time))
        
    def fit(self, 
            X_train, 
            y_train):
        
        print('Training Random Forest model...\n')
        self.model.fit(X_train, y_train) 

       
    def predict(self, X_test, y_test):
        print('Predict on test dataset')
        y_pred = self.model.predict(X_test) 
        
        print('Generating Classification Report')
        classification_report = metrics.compute_acc_acc5_f1_prec_rec(y_test, y_pred)
        return classification_report
    
    def free(self):
        print('free memory')
        del self.model 
       

