from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import itertools
import collections
import numpy as np
import tensorflow as tf
import sonnet as snt

# from nlp.util.data_reader import Dataset, load_vocab, seed_random
# from nlp.util.data_reader import get_loc, mkdirs

from nlp.readers.data_reader import Dataset, load_vocab
from nlp.util.utils import get_loc, mkdirs, seed_random

from nlp.rwa.RWACell import RWACell
from nlp.tensorflow_with_latest_papers import rnn_cell_modern
# from nlp.recurrent_highway_networks.rhn import RHNCell

tf.logging.set_verbosity(tf.logging.INFO)


def interleave(a,b):
    return list(itertools.chain.from_iterable(zip(a,b)))

def arrayfun(f,A):
    return list(map(f,A))

def isnum(a):
    try:
        float(repr(a))
        ans = True
    except:
        ans = False
    return ans

def init_dict(initializer, keys):
    if initializer!=None:
        if isnum(initializer):
            initializer = tf.constant_initializer(initializer)
        return {k: initializer for k in keys}
    return None

'''
MAKE DILATED CONVOLUTION FILTER/LAYER MODULE!

https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py
'''
class DCNN(snt.AbstractModule):
    def __init__(self, name="dcnn"):
        super(DCNN, self).__init__(name=name)
    def _build(self, inputs):
        return None



class CharEmbed(snt.AbstractModule):
    def __init__(self, vocab_size, embed_dim, max_word_length=None, initializer=None, trainable=True, name="char_embed"):
        super(CharEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# char_vocab.size
        self._embed_dim = embed_dim
        self._max_word_length = max_word_length
        self._initializers = init_dict(initializer, ['embeddings'])
        self._trainable = trainable
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._char_embedding = snt.Embed(vocab_size=self._vocab_size,
                                             embed_dim=self._embed_dim,
                                             trainable=self._trainable,
                                             name="internal_embed")
    
    # inputs shape = [batch_size, num_unroll_steps, max_word_length]
    # inputs = char ids, output = input_embedded
    def _build(self, inputs):
        output = self._char_embedding(inputs)
        
        max_word_length = self._max_word_length
        if max_word_length==None:
            max_word_length = tf.shape(inputs)[2]
            
        output = tf.reshape(output, [-1, max_word_length, self._embed_dim])
        
        self._clear_padding_op = tf.scatter_update(self._char_embedding.embeddings,
                                                   [0],
                                                   tf.constant(0.0, shape=[1, self._embed_dim]))
        return output
    
    @property
    def embeddings(self):
        self._ensure_is_connected()
        return self._char_embedding.embeddings
    
    @property
    def clear_padding_op(self):
        self._ensure_is_connected()
        return self._clear_padding_op
    
    def clear_padding(self, sess):
        sess.run(self.clear_padding_op)
      
    def initialize_to(self, sess, v):
        self._ensure_is_connected()
        sess.run(tf.assign(self.embeddings, v))


''' Time Delay Neural Network'''
class TDNN(snt.AbstractModule):
    def __init__(self, kernels, kernel_features, initializer=None, name="tdnn"):
        super(TDNN, self).__init__(name=name)
        self._kernels = kernels
        self._kernel_features = kernel_features
        self._initializers = init_dict(initializer, ['w','b'])
        assert len(self._kernels) == len(self._kernel_features), 'Kernel and Features must have the same size'
        
    def _build(self, inputs):
        #max_word_length = inputs.get_shape()[1]
        max_word_length = tf.shape(inputs)[1]#xxx1
        
        embed_size = inputs.get_shape()[-1]
        
        inputs = tf.expand_dims(inputs, 1)
        
        layers = []
        self.conv_layers = []
        for kernel_size, kernel_feature_size in zip(self._kernels, self._kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            ## [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv_fxn = snt.Conv2D(output_channels=kernel_feature_size,
                                  kernel_shape=[1,kernel_size],# ?? [kernel_size,1] ??
                                  initializers=self._initializers,
                                  padding='VALID')
            conv = conv_fxn(inputs)
            
            self.conv_layers.append(conv_fxn)



            ## [batch_size x 1 x 1 x kernel_feature_size]
            
#             pool = tf.nn.max_pool(tf.tanh(conv), 
#                                   ksize= [1,1,reduced_length,1], 
#                                   strides= [1,1,1,1],
#                                   padding= 'VALID')

            # https://stackoverflow.com/questions/43574076/tensorflow-maxpool-with-dynamic-ksize#xxx2
            pool = tf.reduce_max(tf.tanh(conv),
                                 axis=2,
                                 keep_dims=True
                                 )
            
            
            
            
            layers.append(tf.squeeze(pool, [1, 2]))
            
        if len(self._kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
        return output
    
    def initialize_conv_layer_to(self, sess, i, w, b):
        self._ensure_is_connected()
        sess.run(tf.assign(self.conv_layers[i].w, w))
        sess.run(tf.assign(self.conv_layers[i].b, b))

class Highway(snt.AbstractModule):
    def __init__(self, output_size, 
                 num_layers=1,
                 bias=-2.0,
                 f=tf.nn.relu,
                 initializer=None,
                 name="highway"):
        super(Highway, self).__init__(name=name)
        self._output_size = output_size
        self._num_layers = num_layers
        self._bias = bias
        self._f = f
        self._initializers = init_dict(initializer, ['w','b'])
        
    def _build(self, inputs):
        self.lin_g, self.lin_t = [],[]
        for idx in range(self._num_layers):
            lin_in_g = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_g")
            lin_in_t = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_t")
            
            self.lin_g.append(lin_in_g)
            self.lin_t.append(lin_in_t)
            
            g = self._f(lin_in_g(inputs))
            t = tf.sigmoid(lin_in_t(inputs) + self._bias)

            output = t * g + (1. - t) * inputs
            inputs = output
            
        return output
    
    def initialize_lin_layers_to(self, sess, Lg, Lt):
        self._ensure_is_connected()
        i=0
        for g, t in zip(Lg, Lt):
            sess.run(tf.assign(self.lin_g[i].w, g[0]))
            sess.run(tf.assign(self.lin_g[i].b, g[1]))
            sess.run(tf.assign(self.lin_t[i].w, t[0]))
            sess.run(tf.assign(self.lin_t[i].b, t[1]))
            i+=1

''' simple sonnet wrapper for dropout'''
class Dropout(snt.AbstractModule):
    def __init__(self, dropout_prob, name="dropout"):
        super(Dropout, self).__init__(name=name)
        self._dropout_prob = dropout_prob
    
    def _build(self, inputs):
        return tf.nn.dropout(inputs, keep_prob=1.-self._dropout_prob)
    
'''
193    modules/basic_rnn.py
30     modules/gated_rnn.py
255    examples/rnn_shakespeare.py    
'''
class DeepLSTM(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_rnn_layers=1, 
                 num_unroll_steps=35, 
                 batch_size=20, 
                 dropout=0.0,
                 initializer=None,
                 forget_bias=0.0,
                 name="deep_lstm"):
        super(DeepLSTM, self).__init__(name=name)
        self._rnn_size = rnn_size
        self._num_rnn_layers = num_rnn_layers
        self._num_unroll_steps = num_unroll_steps
        self._batch_size = batch_size
        self._dropout = dropout
        self._initializer = initializer
        self._forget_bias = forget_bias
        
        with self._enter_variable_scope():

            RNN = snt.LSTM
            #RNN = RHN_Cell
            
            self._lstms = [
                RNN(self._rnn_size, forget_bias=self._forget_bias,
                         initializers=init_dict(self._initializer,
                                                RNN.get_possible_initializer_keys()),
                         name="lstm_{}".format(i))
                for i in range(self._num_rnn_layers)
            ]
            if self._dropout > 0.0:
                dropouts = [Dropout(dropout_prob=self._dropout) for i in range(self._num_rnn_layers)]
                self._lstms = interleave(self._lstms, dropouts)
            
            if len(self._lstms) > 1:
                self._core = snt.DeepRNN(self._lstms, name="deep_lstm_core", skip_connections=False)
            else:
                self._core = self._lstms[0]

    def _build(self, inputs):
        d1 = inputs.get_shape().as_list()[1]
        
        self._initial_rnn_state = self._core.initial_state(self._batch_size)
        
        input_cnn = tf.reshape(inputs, [self._batch_size, self._num_unroll_steps, d1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, self._num_unroll_steps, 1)]
        
        outputs, self._final_rnn_state = tf.contrib.rnn.static_rnn(
            cell=self._core, 
            inputs=input_cnn2,
            initial_state=self._initial_rnn_state,
            dtype=tf.float32)
        
        return outputs, self._final_rnn_state
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        return self._final_rnn_state

class OutputWord(snt.AbstractModule):
    def __init__(self, output_size,
                 initializer=None,
                 name="output_word"):
        super(OutputWord, self).__init__(name=name)
        self._output_size = output_size# word_vocab.size
        self._initializer = initializer
        with self._enter_variable_scope():
            self._output_module = snt.Linear(self._output_size,
                                             initializers=init_dict(self._initializer, ['w','b']),
                                             name="output_word_embedding")
    
    def _build(self, inputs):
        input_sequence = tf.stack(inputs)
        batch_output_module = snt.BatchApply(self._output_module)
        output_sequence_logits = batch_output_module(input_sequence)
        return output_sequence_logits
    
    def initialize_lin_layer_to(self, sess, w, b):
        self._ensure_is_connected()
        sess.run(tf.assign(self._output_module.w, w))
        sess.run(tf.assign(self._output_module.b, b))
        
class LanguageModel(snt.AbstractModule):
    def __init__(self, char_vocab_size, word_vocab_size,
                 initializer=None,
                 char_embed_size=15,
                 batch_size=20,
                 num_highway_layers=2,
                 num_rnn_layers=2,
                 rnn_size=650,
                 max_word_length=65,
                 kernels         = [ 1,   2,   3,   4,   5,   6,   7],
                 kernel_features = [50, 100, 150, 200, 200, 200, 200],
                 num_unroll_steps=35,
                 dropout=0.0,
                 name="language_model"):
        super(LanguageModel, self).__init__(name=name)
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._initializer = initializer
        self._char_embed_size = char_embed_size
        self._batch_size = batch_size
        self._num_highway_layers = num_highway_layers
        self._num_rnn_layers = num_rnn_layers
        self._rnn_size = rnn_size
        self._max_word_length = max_word_length
        self._kernels = kernels
        self._kernel_features = kernel_features
        self._num_unroll_steps = num_unroll_steps
        self._dropout = dropout
        
        with self._enter_variable_scope():
            self._char_embed_module = CharEmbed(vocab_size=self._char_vocab_size, 
                                                embed_dim=self._char_embed_size, 
                                                max_word_length=self._max_word_length,
                                                initializer=initializer)
            self._char_cnn_module = TDNN(kernels=self._kernels,
                                         kernel_features=self._kernel_features,
                                         initializer=initializer)
            if self._num_highway_layers>0:
                self._highway_module = Highway(output_size=sum(kernel_features),
                                               num_layers=self._num_highway_layers,
                                               initializer=initializer)
            self._lstm_module = DeepLSTM(rnn_size=self._rnn_size,
                                         num_rnn_layers=self._num_rnn_layers,
                                         num_unroll_steps=self._num_unroll_steps,
                                         batch_size=self._batch_size,
                                         dropout=self._dropout,
                                         initializer=initializer
                                         )
            self._output_word_module = OutputWord(output_size=self._word_vocab_size,
                                                  initializer=initializer)
            
    def _build(self, inputs):
        input_embedded = self._char_embed_module(inputs)
        input_cnn = self._char_cnn_module(input_embedded)
        if self._num_highway_layers>0:
            input_cnn = self._highway_module(input_cnn)
        rnn_outputs, final_rnn_state = self._lstm_module(input_cnn)
        output_sequence_logits = self._output_word_module(rnn_outputs)
        return output_sequence_logits, final_rnn_state
    
    @property
    def clear_padding_op(self):
        self._ensure_is_connected()
        return self._char_embed_module.clear_padding_op
    
    @property
    def initial_rnn_state(self):
        return self._lstm_module.initial_rnn_state
    
    @property
    def final_rnn_state(self):
        return self._lstm_module.final_rnn_state


#---------------------------------------------------------------------------------
''' EXPERIMENTAL.....'''
#---------------------------------------------------------------------------------
class RHN_Cell(snt.RNNCore):
    def __init__(self, output_size,
                 num_layers=5, 
                 use_inputs_on_each_layer=False,
                 use_kronecker_reparameterization=False,
                 initializers=None,
                 name="rhn_cell"):
        super(RHN_Cell, self).__init__(name=name)
        self._output_size = output_size
        self._num_layers=num_layers
        self._use_inputs_on_each_layer=use_inputs_on_each_layer
        self._use_kronecker_reparameterization=use_kronecker_reparameterization
    
    @classmethod
    def get_possible_initializer_keys():
        return None
    
    @property
    def output_size(self):
        """Returns a description of the output size, without batch dimension."""
        return tf.TensorShape([self._output_size])
        
    def _build(self, inputs, state):
        cell = rnn_cell_modern.HighwayRNNCell(self._output_size, 
                                              num_highway_layers=self._num_layers,
                                              use_inputs_on_each_layer=self._use_inputs_on_each_layer,
                                              use_kronecker_reparameterization=self._use_kronecker_reparameterization)
#         cell = RHNCell(num_units=self._output_size,
#                        in_size=self._in_size,
#                        is_training=True,
#                        depth=self._num_layers)
        return cell(inputs, state)

class RWA_Cell(snt.RNNCore):
    def __init__(self, num_units,
                 name="rwa_cell"):
        super(RWA_Cell, self).__init__(name=name)
        self._num_units = num_units
    
    @classmethod
    def get_possible_initializer_keys():
        return None
    
    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units, self._num_units, self._num_units)
        
    def _build(self, inputs, state):
        cell = RWACell(self._num_units)
        return cell(inputs, state)