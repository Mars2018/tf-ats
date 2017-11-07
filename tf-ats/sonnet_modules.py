from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
#import itertools
import collections
import numpy as np
import tensorflow as tf
import sonnet as snt

from nlp.util.utils import interleave

'''
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/rnn_shakespeare.py
'''

''' simple sonnet wrapper for dropout'''
class Dropout(snt.AbstractModule):
    def __init__(self, keep_prob=None, name="dropout"):
        super(Dropout, self).__init__(name=name)
        self._keep_prob = keep_prob
        
        if keep_prob is None:
            with self._enter_variable_scope():
                self._keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    def _build(self, inputs):
        return tf.nn.dropout(inputs, keep_prob=self._keep_prob)
    
    @property
    def keep_prob(self):
        self._ensure_is_connected()
        return self._keep_prob
    
class MultiLSTM(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 use_skip_connections=False,
                 use_peepholes=False,
                 seq_len=None,
                 #initializer=None,
                 name="multi_lstm"):
        super(MultiLSTM, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self.use_skip_connections = use_skip_connections
        self.use_peepholes = use_peepholes
        self._seq_len = seq_len
        #self.initializer = initializer
        
        if use_skip_connections:
            self.dropout = 0.0
        
        with self._enter_variable_scope():
            
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])
            
            cell = snt.LSTM #RNN = RHN_Cell
#             use_peepholes=False,
#             use_batch_norm_h=False,
#             use_batch_norm_x=False,
#             use_batch_norm_c=False,
            
            self.subcores = [
                cell(self.rnn_size, 
                     forget_bias=self.forget_bias,
                     use_peepholes=self.use_peepholes,
                     #initializers=init_dict(self.initializer,
                     #                       cell.get_possible_initializer_keys()),
                     name="multi_lstm_subcore_{}".format(i))
                for i in range(self.num_layers)
            ]

    def _build(self, inputs):

#         ## SKIP CONNECTIONS ## rnn_shakespeare.py
#         if self.use_skip_connections:
#             skips = []
#             embed_dim = inputs.get_shape().as_list()[-1]
#             current_input_shape = embed_dim
#             for lstm in self.subcores:
#                 input_shape = tf.TensorShape([current_input_shape])
#                 skip = snt.SkipConnectionCore(
#                     lstm,
#                     input_shape=input_shape,
#                     name="skip_{}".format(lstm.module_name))
#                 skips.append(skip)
#                 # SkipConnectionCore concatenates the input with the output..
#                 # ...so the dimensionality increases with depth.
#                 current_input_shape += self.rnn_size
#             self.subcores = skips
        
        ## DROPOUT ##
        self._keep_prob = tf.placeholder_with_default(1.0-self.dropout, shape=())
        if self.dropout > 0.0:
            dropouts = [Dropout(keep_prob=self._keep_prob) for i in range(self.num_layers)]
            self.subcores = interleave(self.subcores, dropouts)
        
        if len(self.subcores) > 1:
            self.core = snt.DeepRNN(self.subcores, name="multi_lstm_core", skip_connections=self.use_skip_connections)
        else:
            self.core = self.subcores[0]
                
        self._initial_rnn_state = self.core.zero_state(self.batch_size, dtype=tf.float32)
        
        output, final_rnn_state = tf.nn.dynamic_rnn(self.core,
                                                    inputs,
                                                    dtype=tf.float32,
                                                    sequence_length=self._seq_len,
                                                    initial_state=self._initial_rnn_state)
        return output, final_rnn_state
    
    @property
    def keep_prob(self):
        self._ensure_is_connected()
        return self._keep_prob
    
    @property
    def seq_len(self):
        self._ensure_is_connected()
        return self._seq_len
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state
    
#     @property
#     def final_rnn_state(self):
#         self._ensure_is_connected()
#         return self.final_rnn_state