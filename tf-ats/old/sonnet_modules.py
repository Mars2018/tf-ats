from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import collections
import numpy as np
import tensorflow as tf
import sonnet as snt

#import LM_sonnet_modules as lm


from nlp.util.utils import interleave

'''
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/rnn_shakespeare.py
'''

class WordEmbed(snt.AbstractModule):
    def __init__(self, vocab_size=None, embed_dim=None, initial_matrix=None, trainable=True, name="word_embed"):
        super(WordEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# word_vocab.size
        self._embed_dim = embed_dim
        self._trainable = trainable
        if initial_matrix:
            self._vocab_size = initial_matrix.shape[0]
            self._embed_dim = initial_matrix.shape[1]
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._embedding = snt.Embed(vocab_size=self._vocab_size,
                                        embed_dim=self._embed_dim,
                                        trainable=self._trainable,
                                        name="internal_embed")
    
    # inputs shape = [batch_size, ?]
    # inputs = word_idx, output = input_embedded
    def _build(self, inputs):
        return self._embedding(inputs)

###############################################################################
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

###############################################################################

def rnn_unit(args):
    kwargs = {}
    if args.unit=='lstm':
        rnn = tf.nn.rnn_cell.LSTMCell
        kwargs = { 'reuse':False, 'forget_bias':args.forget_bias, 'state_is_tuple':True }
    elif args.unit=='gru':
        rnn = tf.nn.rnn_cell.GRUCell
        kwargs = { 'reuse':False }
    elif args.unit=='rwa':
        rnn = lm.RWA_Cell
    elif args.unit=='rhn':
        rnn = lm.RHN_Cell
    return rnn, kwargs

def create_rnn_cell(args):
    rnn, kwargs = rnn_unit(args)
    cell = rnn(args.rnn_size, **kwargs)
    #cell = tf.contrib.rnn.ResidualWrapper(cell)
    #cell = tf.contrib.rnn.HighwayWrapper(cell)
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=10, attn_size=100)
    if args.dropout>0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args._keep_prob)#variational_recurrent=True
    return cell

class DeepRNN(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 train_initial_state=False,
                 unit='lstm',
                 name="deep_rnn"):
        super(DeepRNN, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        self.train_initial_state = train_initial_state
        self.unit = unit
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0-self.dropout, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])

    def _build(self, inputs):
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell(self) for _ in range(self.num_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell(self)

        if self.train_initial_state:
            self._initial_rnn_state = cell.initial_state(self.batch_size, tf.float32, trainable=True)
        else:
            self._initial_rnn_state = cell.zero_state(self.batch_size, tf.float32)
        
        output, final_rnn_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    dtype=tf.float32,
                                                    sequence_length=self._seq_len,
                                                    initial_state=self._initial_rnn_state)
        return output, final_rnn_state
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state

###############################################################################

class DeepBiRNN(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 train_initial_state=False,
                 unit='lstm',
                 name="deep_bi_rnn"):
        super(DeepBiRNN, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        self.train_initial_state = train_initial_state
        self.unit = unit
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])

    def _build(self, inputs):
        cells_fw = [create_rnn_cell(self) for i in range(self.num_layers)]
        cells_bw = [create_rnn_cell(self) for i in range(self.num_layers)]
         
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                   sequence_length=self._seq_len,
                                                                                                   dtype=tf.float32,
                                                                                                   scope='BiMultiLSTM')
        return outputs, (output_state_fw, output_state_bw)
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob

###############################################################################

class MultiLSTM(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 use_skip_connections=False,
                 use_peepholes=False,
                 train_initial_state=False,
                 #initializer=None,
                 name="multi_lstm"):
        super(MultiLSTM, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        self.use_skip_connections = use_skip_connections
        self.use_peepholes = use_peepholes
        self.train_initial_state = train_initial_state
        #self.initializer = initializer
        
        if use_skip_connections:
            self.dropout = 0.0
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0-self.dropout, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])
            
            cell = snt.LSTM
            #cell = lm.RWA_Cell
            #cell = lm.RHN_Cell
            
            self.subcores = [
                cell(self.rnn_size, 
                    forget_bias=self.forget_bias,
                    use_peepholes=self.use_peepholes,
                    name="multi_lstm_subcore_{}".format(i+1),
                     
                     #initializers=init_dict(self.initializer,
                     #                       cell.get_possible_initializer_keys()),
                     #use_batch_norm_h=True, use_batch_norm_x=True, use_batch_norm_c=True, #is_training=self._is_training,
                     )
                for i in range(self.num_layers)
            ]

    def _build(self, inputs):
        
        ## DROPOUT ##
        if self.dropout > 0.0:
            dropouts = [Dropout(keep_prob=self._keep_prob) for i in range(self.num_layers)]
            self.subcores = interleave(self.subcores, dropouts)
        
        if len(self.subcores) > 1:
            self.core = snt.DeepRNN(self.subcores, name="multi_lstm_core", skip_connections=self.use_skip_connections)
        else:
            self.core = self.subcores[0]

        if self.train_initial_state:
            self._initial_rnn_state = self.core.initial_state(self.batch_size, tf.float32, trainable=True)
            #self._initial_rnn_state = snt.TrainableInitialState(self.core.initial_state(self.batch_size, tf.float32))()
        else:
            self._initial_rnn_state = self.core.zero_state(self.batch_size, tf.float32)
        
        output, final_rnn_state = tf.nn.dynamic_rnn(self.core,
                                                    inputs,
                                                    dtype=tf.float32,
                                                    sequence_length=self._seq_len,
                                                    initial_state=self._initial_rnn_state
                                                    )
    
        return output, final_rnn_state
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state


###############################################################################


class BiMultiLSTM(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 #initializer=None,
                 name="bi_multi_lstm"):
        super(BiMultiLSTM, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        #self.initializer = initializer
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])

    def _build(self, inputs):
        cell = snt.LSTM
         
        cells_fw = [cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_fw_{}".format(i+1))
                    for i in range(self.num_layers)]
        cells_bw = [cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_bw_{}".format(i+1))
                    for i in range(self.num_layers)]
         
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                   sequence_length=self._seq_len,
                                                                                                   dtype=tf.float32,
                                                                                                   scope='BiMultiLSTM')
        return outputs, (output_state_fw, output_state_bw)
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    
    





    ##################################################################################################
    ##### OLD STUFF ##########
    
    ###################################################
    # MultiLSTM skip connections
            
#     ## SKIP CONNECTIONS ## rnn_shakespeare.py
#     if self.use_skip_connections:
#         skips = []
#         embed_dim = inputs.get_shape().as_list()[-1]
#         current_input_shape = embed_dim
#         for lstm in self.subcores:
#             input_shape = tf.TensorShape([current_input_shape])
#             skip = snt.SkipConnectionCore(
#                 lstm,
#                 input_shape=input_shape,
#                 name="skip_{}".format(lstm.module_name))
#             skips.append(skip)
#             # SkipConnectionCore concatenates the input with the output..
#             # ...so the dimensionality increases with depth.
#             current_input_shape += self.rnn_size
#         self.subcores = skips

    ###################################################
    
    # old BiMultiLSTM build() fxn:
#     def _build(self, inputs):
#         output = inputs
#         cell = snt.LSTM
#          
#         for i in range(self.num_layers):
# #             lstm_fw = tf.nn.rnn_cell.LSTMCell(self.rnn_size, forget_bias=self.forget_bias)
# #             lstm_bw = tf.nn.rnn_cell.LSTMCell(self.rnn_size, forget_bias=self.forget_bias)
#             lstm_fw = cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_fw_{}".format(i+1))
#             lstm_bw = cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_bw_{}".format(i+1))
#      
#             _initial_state_fw = lstm_fw.zero_state(self.batch_size, tf.float32)
#             _initial_state_bw = lstm_bw.zero_state(self.batch_size, tf.float32)
#      
#             output, _states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, output, 
#                                                               initial_state_fw=_initial_state_fw,
#                                                               initial_state_bw=_initial_state_bw,
#                                                               sequence_length=self._seq_len,
#                                                               scope='BLSTM_'+str(i+1))
#             output = tf.concat(output, 2)
#         return output, _states