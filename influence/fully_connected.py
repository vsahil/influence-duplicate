from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

# import numpy as np
# import pandas as pd
# from sklearn import linear_model, preprocessing, cluster
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.linalg as slin
# import scipy.sparse.linalg as sparselin
# import scipy.sparse as sparse 

import os.path
# import time
# import IPython
import tensorflow as tf
import math

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet

class Fully_Connected(GenericNeuralNet):

    def __init__(self, input_dim, hidden1_units, hidden2_units, weight_decay, **kwargs):
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.hidden1_units = hidden1_units
        self.hidden2_units = hidden2_units

        super(Fully_Connected, self).__init__(**kwargs)


    def hidden_layer_compute(self, input_data, input_neurons, output_neurons):
        weights = variable_with_weight_decay(
            'weights', 
            [input_neurons * output_neurons],
            stddev=2.0 / math.sqrt(float(input_neurons * output_neurons)),
            wd=self.weight_decay)

        biases = variable(
            'biases',
            [output_neurons],
            tf.constant_initializer(0.0))
        
        weights_reshaped = tf.reshape(weights, [input_neurons, output_neurons])
        hidden = tf.add(tf.matmul(input_data, weights_reshaped), biases)
        return hidden


    def get_all_params(self):
        all_params = []
        # for layer in ['h1_a', 'h1_c', 'h2_a', 'h2_c', 'h3_a', 'h3_c', 'softmax_linear']:        
        for layer in ['h1', 'h2', 'out']:
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
                all_params.append(temp_tensor)      
        return all_params        
        

    def retrain(self, num_steps, feed_dict):        

        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in range(num_steps):   
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)


    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def inference(self, input_x):                
        input_reshaped = tf.reshape(input_x, [-1, self.input_dim])

        # Hidden 1
        with tf.variable_scope('h1'):
            h1 = self.hidden_layer_compute(input_reshaped, self.input_dim, self.hidden1_units)
            
        # Hidden 2
        with tf.variable_scope('h2'):
            h2 = self.hidden_layer_compute(h1, self.hidden1_units, self.hidden2_units)
            
        # Shared layers / hidden 3
        with tf.variable_scope('out'):
            h3 = self.hidden_layer_compute(h2, self.hidden2_units, self.num_classes)
        
        logits = h3
        
        return logits
        
    
    def predictions(self, logits):
        preds = tf.argmax(logits, 1, name='preds')
        return preds