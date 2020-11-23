import numpy as np
from itertools import groupby
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

def regularity(sequence):
    return len(sequence) / len(set(sequence))    

def stationarity(sequence):
    rle = [(value, sum(1 for i in g)) for value, g in groupby(sequence)]
    return np.mean([length for _, length in rle])

def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred, normalize=True)

def balanced_accuracy(y_true, y_pred):
    if y_pred.ndim == 1:
        return balanced_accuracy_score(y_true, y_pred)
    else:
        return balanced_accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

def accuracy_top_k(y_true, y_pred, K=5):
    if y_pred.ndim == 1:
        return 0
    else:
        order = np.argsort(y_pred, axis=1)
        correct = 0

        for i, sample in enumerate(np.argmax(y_true, axis=1)):
            if sample in order[i, -K:]:
                correct += 1
        return correct / len(y_true)

def compute_acc_acc5_f1_prec_by_class(y_true, y_pred, num_classes, categories):
    try:
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)

        labels_y = dict(zip(range(num_classes), categories))
        keys = np.unique(y_true)
        dic = dict((key,value) for key, value in labels_y.items() if key in keys)
        target_names = sorted(dic.values())
        print('...Creating classification report')
        dic_results = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        eval_acc = cm.diagonal()
        for i, key in enumerate(sorted(dic.keys())):
            name = labels_y[key]
            dic_results[name]['accuracy'] = eval_acc[i]

        result_class = pd.DataFrame(dic_results)
        result_class['macro avg'].iloc[-1] = result_class.iloc[-1][0:num_classes].mean()

        return result_class
    except Exception as e:
        raise e

def compute_acc_acc5_f1_prec_rec(y_true, y_pred):
    dic_model = {}
    dic_model['acc'] = accuracy(y_true, y_pred)
    dic_model['acc_top_K5'] = accuracy_top_k(y_true, y_pred, K=5)
    dic_model['balanced_accuracy'] = balanced_accuracy(y_true, y_pred)
    dic_model['precision_macro'] = precision_macro(y_true, y_pred)
    dic_model['recall_macro'] = recall_macro(y_true, y_pred)
    dic_model['f1-macro']= f1_macro(y_true, y_pred) 
    
    result = pd.DataFrame(dic_model, index=[0])

    return result



######################
# evaluation metrics #
######################
def mape(true, pred, sample_weight=None):
    """
    it is very, very slow when the shapes are different and the dataset is very large (e.g. > 1,000,000 rows).
    """
    if true.shape != pred.shape:
        true = true.reshape(pred.shape)
    return np.average( np.abs( (true - pred) / true ), weights=sample_weight)

def mape_xgb(true, pred, sample_weight=None):
    return 'error', mape(true, pred.get_label(), sample_weight)

def mse(true, pred, sample_weight=None):
    if true.shape[0] == 0:
        return None
    return mean_squared_error(true, pred, sample_weight)

def mse_xgb(true, pred, sample_weight=None):
    return 'error', mse(true, pred.get_label(), sample_weight)


class MetricsLogger:

    def __init__(self):
        self._df = pd.DataFrame({'method': [],
                                 'epoch': [],
                                 'dataset': [],
                                 'timestamp': [],
                                 'train_loss': [],
                                 'train_acc': [],
                                 'train_acc_top5': [],
                                 'train_f1_macro': [],
                                 'train_prec_macro': [],
                                 'train_rec_macro': [],
                                 'train_acc_up': [],
                                 'test_loss': [],
                                 'test_acc': [],
                                 'test_acc_top5': [],
                                 'test_f1_macro': [],
                                 'test_prec_macro': [],
                                 'test_rec_macro': [],
                                 'test_acc_up': []})

    def log(self, method, epoch, dataset, train_loss, train_acc,
            train_acc_top5, train_f1_macro, train_prec_macro, train_rec_macro,
            test_loss, test_acc, test_acc_top5, test_f1_macro,
            test_prec_macro, test_rec_macro):
        timestamp = datetime.now()

        if len(self._df) > 0:
            train_max_acc = self._df['train_acc'].max()
            test_max_acc = self._df['test_acc'].max()
        else:
            train_max_acc = 0
            test_max_acc = 0

        self._df = self._df.append({'method': method,
                                    'epoch': epoch,
                                    'dataset': dataset,
                                    'timestamp': timestamp,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'train_acc_top5': train_acc_top5,
                                    'train_f1_macro': train_f1_macro,
                                    'train_prec_macro': train_prec_macro,
                                    'train_rec_macro': train_rec_macro,
                                    'train_acc_up': 1 if train_acc > train_max_acc else 0,
                                    'test_loss': test_loss,
                                    'test_acc': test_acc,
                                    'test_acc_top5': test_acc_top5,
                                    'test_f1_macro': test_f1_macro,
                                    'test_prec_macro': test_prec_macro,
                                    'test_rec_macro': test_rec_macro,
                                    'test_acc_up': 1 if test_acc > test_max_acc else 0},
                                   ignore_index=True)

    def save(self, file):
        self._df.to_csv(file, index=False)

    def load(self, file):
        if os.path.isfile(file):
            self._df = pd.read_csv(file)
        else:
            print("WARNING: File '" + file + "' not found!")

        return self