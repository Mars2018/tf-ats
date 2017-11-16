from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse
import pprint
import pickle as pk
import numpy as np
import tensorflow as tf
import sonnet as snt

from nlp.util import config
from nlp.util import utils as U
from nlp.util.utils import tic, toc, qwk, make_abs

from nlp.readers.vocab import Vocab
from nlp.readers.text_reader import GlobReader, TextParser, FieldParser, EssayBatcher, REGEX_NUM

import options
from attention import attention
import nlp.tf_tools.sonnet_modules as sm
#import LM_sonnet_modules as lm


''' get config '''    
parser = options.get_parser()
config_file = 'config/ats.conf'
argv=[]# override config file here
#argv.append('--seed'); argv.append('289681027')
FLAGS = config.get_config(parser=parser, config_file=config_file, argv=argv)
FLAGS.chkpt_dir = make_abs(FLAGS.chkpt_dir)
FLAGS.rand_seed = U.seed_random(FLAGS.rand_seed)
pprint.pprint(FLAGS)

''' setup checkpoint directory '''
if not os.path.exists(FLAGS.chkpt_dir):
    U.mkdirs(FLAGS.chkpt_dir)
    print('Created checkpoint directory', FLAGS.chkpt_dir)
config.save_local_config(FLAGS)

#mode = FLAGS.run_mode
batch_size = FLAGS.batch_size
pid = FLAGS.item_id
essay_file = os.path.join(FLAGS.data_dir, '{0}', '{0}.txt.clean.tok').format(pid)
embed_file = FLAGS.embed_path.format(FLAGS.embed_dim)

''' load Glove word embeddings, along with word vocab '''
embed_matrix, word_vocab = Vocab.load_word_embeddings(embed_file, essay_file, min_freq=FLAGS.min_word_count)
print(embed_matrix.shape)

''' create essay reader, parser, & batcher '''
reader =  GlobReader(essay_file, chunk_size=10000, regex=REGEX_NUM, shuf=True, rng=U.rng)
text_parser = TextParser(word_vocab=word_vocab)

fields = {0:'id', 1:'y', -1:text_parser}
field_parser = FieldParser(fields, reader=reader)
    
#batcher = EssayBatcher(reader=field_parser, batch_size=batch_size, trim_words=True)
#batcher = EssayBatcher(reader=field_parser, batch_size=batch_size, max_text_length=FLAGS.max_text_length)

batcher = EssayBatcher(reader=field_parser,
                       batch_size=FLAGS.batch_size,
                       max_text_length=FLAGS.max_text_length, 
                       trim_words=FLAGS.trim_words)

###############################
'''
https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
https://github.com/yvesx/tf-rnn-pub
( https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/ )
https://danijar.com/variable-sequence-lengths-in-tensorflow/

'''
####################

''' DEFINE REGRESSION MODEL '''
with tf.variable_scope("Model", 
                       initializer=tf.truncated_normal_initializer(seed=FLAGS.rand_seed, 
                                                                   stddev=0.05, 
                                                                   dtype=tf.float32)
                       ) as scope:

    ''' PLACEHOLDERS '''
    inputs = tf.placeholder(tf.int32,
                            shape=[batch_size, None],
                            name="inputs")
    
    targets = tf.placeholder(tf.float32,
                             shape=[batch_size, 1],
                             name="targets")
    
    ''' EMBEDDING LAYER '''
    ## OLD ##
    #E = tf.get_variable(name="E", shape=embed_matrix.shape, initializer=tf.constant_initializer(embed_matrix), trainable=True)# TRUE/FALSE ???
    #rnn_inputs = tf.nn.embedding_lookup(E, inputs)
    ## SONNET ##
    embed_module = snt.Embed(existing_vocab=embed_matrix, trainable=True)
    rnn_inputs = embed_module(inputs)
    
    ####################################################################################
    
    RNN = sm.DeepRNN#sm.MultiLSTM
    if FLAGS.bidirectional:
        RNN = sm.DeepBiRNN#sm.BiMultiLSTM
    
    rnn = RNN(rnn_size=FLAGS.rnn_dim,
              num_layers=FLAGS.rnn_layers,
              batch_size=FLAGS.batch_size,
              dropout=FLAGS.dropout,
              train_initial_state=FLAGS.train_initial_state,
              unit=FLAGS.rnn_unit,
#               use_skip_connections=FLAGS.skip_connections,
#               use_peepholes=FLAGS.peepholes
              )
    rnn_output, final_state = rnn(rnn_inputs)
    keep_prob = rnn.keep_prob
    seq_len = rnn.seq_len
    dim = rnn_output.get_shape().as_list()[-1]
    
    #############################################################################################
    
    ''' RNN AGGREGATION? '''
    if FLAGS.att_size>0:
        output, alphas = attention(rnn_output, FLAGS.att_size, return_alphas=True)
    elif FLAGS.mean_pool: ## use mean pooled rnn states
        output = tf.reduce_mean(rnn_output, axis=1)
    else: ## otherwise just use final rnn state
        output = tf.gather_nd(rnn_output, tf.stack([tf.range(batch_size), seq_len-1], axis=1))
    
    ## final dropout layer (??) ##
    if FLAGS.dropout>0:
        drop = sm.Dropout(keep_prob)
        output = drop(output)
        
    #############################################################################################
    
    ''' DENSE LAYER '''
    #dim = output.get_shape().as_list()[-1]
    init_bias = 0.0
    with tf.variable_scope('s1'):
        W = tf.get_variable('W', [dim, 1])
        b = tf.get_variable('b', [1], initializer=tf.constant_initializer(init_bias))
    output = tf.matmul(output, W) + b
    
    #############################################################################################
    
    ''' OUTPUT ACTIVATION (sigmoid?  tanh?) ''' 
    preds = tf.nn.tanh(output)
#     preds = tf.nn.sigmoid(output)
    
    #############################################################################################
    
    ''' LOSS FUNCTION (MSE) '''
    train_loss = tf.losses.mean_squared_error(targets, preds)
    train_qwk = qwk(targets, preds)
    
    learning_rate = tf.get_variable(
        "learning_rate",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(FLAGS.learning_rate),
        trainable=False)
    
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
    
    ''' TRAIN OPS '''
    if FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer
    elif FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer
    else:
        opt = tf.train.GradientDescentOptimizer
    optimizer = opt(learning_rate)

    tvars = tf.trainable_variables(); [print(var) for var in tvars]
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), FLAGS.max_grad_norm)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    ## experimental ##
    #mu = tf.cast(tf.reduce_mean(seq_len), tf.float32)
    #grads, global_norm = tf.clip_by_global_norm(tf.gradients(train_loss * mu, tvars), FLAGS.max_grad_norm)
    #grads, global_norm = tf.clip_by_global_norm( [mu * x for x in tf.gradients(train_loss, tvars)], FLAGS.max_grad_norm)
    
############################################################################

''' TRAINING SESSION '''
with tf.Session() as sess:
    tf.set_random_seed(FLAGS.rand_seed)
    tf.global_variables_initializer().run()
    
    valid_batches, valid_ids = [],None
    epochs= FLAGS.epochs
    epoch=0
    while epoch < epochs:
        epoch += 1
        tic()
        losses, qwks, P, Y, batch = [],[],[],[],0
        for b in batcher.batch_stream(stop=True, skip_ids=valid_ids, w=True, c=False):
            batch+=1
            if epoch==1 and U.rng.rand()<FLAGS.valid_cut:
                valid_batches.append(b)
                continue
            
            feed_dict = { inputs: b.w, targets: b.y, seq_len: b.s, keep_prob: 1.0-FLAGS.dropout }
            fetches = [train_loss, train_op, train_qwk, preds]
            
            loss, _, kappa, p = sess.run(
                fetches,
                feed_dict)
            
            losses.append(loss)
            qwks.append(kappa)
            word_count = batcher.word_count(reset=False)
            sec = toc(reset=False)
            wps = int(word_count/sec)
            P.extend(np.squeeze(p)); Y.extend(np.squeeze(b.y)); kappa = U.nkappa(Y,P)
            
            if batch % FLAGS.print_every == 0:
                sys.stdout.write('\tqwk={0:0.3g}'.format(kappa))
                sys.stdout.write('|mse={0:0.3g}'.format(loss))
                sys.stdout.write('|wps={0}'.format(wps))
                sys.stdout.flush()
            
        if epoch==1:
            valid_ids = set([id for b in valid_batches for id in b.id])
            print('\n{} VALID-BATCHES ({} VALID-IDS)'.format(len(valid_batches),len(valid_ids)))
            #vid=list(valid_ids); vid.sort(); print(vid)
        
        word_count = batcher.word_count()
        sec = toc()
        wps = int(word_count/sec)
        QWK = U.nkappa(Y,P)#QWK = np.mean(qwks)
        
        print('')
        print('\nEpoch {0}\tTRAIN Loss : {1:0.4}\tTRAIN Kappa : {2:0.4}\t{3:0.2g}min|{4}wps\n'.format(epoch, np.mean(losses), QWK, float(sec)/60., wps))
        
        ''' VALIDATION '''
        losses, qwks, P, Y, batch = [],[],[],[],0
        for b in valid_batches:
            batch+=1
            feed_dict = { inputs: b.w, targets: b.y, seq_len: b.s, keep_prob: 1.0 }
            loss, kappa, p = sess.run( [train_loss, train_qwk, preds], feed_dict)
            losses.append(loss)
            qwks.append(kappa)
            P.extend(np.squeeze(p)); Y.extend(np.squeeze(b.y)); kappa = U.nkappa(Y,P)
            
            if batch % FLAGS.print_every == 0:
                sys.stdout.write('\tmse={0:0.4}'.format(loss))
                sys.stdout.write('|qwk={0:0.4}\n'.format(kappa))
                sys.stdout.flush()
                
        QWK = U.nkappa(Y,P)#QWK = np.mean(qwks)
        print('')
        print('Epoch {0}\tVALID Loss : {1:0.4}\tVALID Kappa : {2:0.4}\n'.format(epoch, np.mean(losses), QWK))



###############################################################################################################################
######### OLD STUFF ##############################

#     ''' BUILD RNN '''
#     seq_len = tf.placeholder(tf.int32, [batch_size])
#     keep_prob = tf.placeholder_with_default(1.0, shape=())
#     def create_rnn_cell():
#         cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.rnn_dim, state_is_tuple=True, forget_bias=0.0, reuse=False)
#         #cell = tf.nn.rnn_cell.LSTMCell(FLAGS.rnn_dim, state_is_tuple=True, forget_bias=0.0, reuse=False, use_peepholes=True)
#         #############
#         ## dropout ##
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)#==1.-FLAGS.dropout
#         return cell
#              
#     if FLAGS.rnn_layers > 1:
#         cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(FLAGS.rnn_layers)], state_is_tuple=True)
#     else:
#         cell = create_rnn_cell()
#      
#     initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)
#     output, final_state = tf.nn.dynamic_rnn(cell,
#                                             rnn_inputs,
#                                             dtype=tf.float32,
#                                             sequence_length=seq_len,
#                                             initial_state=initial_rnn_state)
##############################
#     def length(sequence):
#         used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
#         length = tf.reduce_sum(used, 1)
#         length = tf.cast(length, tf.int32)
#         return length
#     seq_len = length(rnn_inputs)