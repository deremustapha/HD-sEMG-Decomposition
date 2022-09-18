# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 01:22:58 2022

@author: Dere Mustapha Deji
PProject: Decomposition of HD-sEMG
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback
import tensorflow as tf





x_sEMG = 'EMGs'
y_spikes = 'Spikes'


def load_train(path, train_ls, overlap, mu):
    
    for i, tr_ts in enumerate(train_ls):
        filename = "{}SG{}-WS120-ST{}.mat".format(path, tr_ts, overlap)
        dataset = loadmat(filename)
        
        if i == 0:
            data_stack = dataset[x_sEMG]
            label_stack = dataset[y_spikes]
        else:
            data_stack = np.row_stack((data_stack, dataset[x_sEMG]))
            label_stack = np.row_stack((label_stack, dataset[y_spikes]))
    
    
    must = np.zeros((label_stack.shape[0], len(mu)))  #(samples, mu)
    for m in mu:
        must[:,m] = label_stack[:,m]    
    
    must = list(must.T)
    return data_stack, must

def load_test(path, test_ls, overlap, mu):
    
    filename = "{}SG{}-WS120-ST{}.mat".format(path, test_ls, overlap)
    dataset = loadmat(filename)
    
    data = dataset[x_sEMG]
    label = dataset[y_spikes]
    
    must = np.zeros((label.shape[0], len(mu)))  #(samples, mu)
    for m in mu:
        must[:,m] = label[:,m]    # vst = list(must.T)
    
    must = list(must.T)
    return data, must


##############################
##############################
##############################


# Loss Function
# calculate f1 score
def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
    y_pred_binary = tf.where(y_pred>=0.5, 1., 0.)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_binary, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*((precision*recall)/(precision + recall + K.epsilon()))

# customized callback function to calculate averaged f1_score and accuracy across all outputs
class AccuracyCallback(Callback):
    def __init__(self, metric_name = 'accuracy'):
        super().__init__()
        self.metric_name = metric_name
        self.val_metric = []
        self.metric = []
        self.val_metric_mean = 0
        self.metric_mean = 0
        self.best_metric = 0
        
    def on_epoch_end(self, epoch, logs=None):
#         print('Accuracycallback')
        # extract values from logs
        self.val_metric = []
        self.metric = []
        for log_name, log_value in logs.items():
            if log_name.find(self.metric_name) != -1:
                if log_name.find('val') != -1:
                    self.val_metric.append(log_value)
                else:
                    self.metric.append(log_value)

        self.val_metric_mean = np.mean(self.val_metric)
        self.metric_mean = np.mean(self.metric)
        logs['val_{}'.format(self.metric_name)] = np.mean(self.val_metric)   # replace it with your metrics
        logs['{}'.format(self.metric_name)] = np.mean(self.metric)   # replace it with your metrics