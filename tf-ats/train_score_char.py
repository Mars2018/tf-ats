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
from nlp.util.utils import tic, toc, qwk, make_abs, memoize

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

FLAGS.kernel_widths = eval(eval(FLAGS.kernel_widths))
FLAGS.kernel_features = eval(eval(FLAGS.kernel_features))

''' setup checkpoint directory '''
if not os.path.exists(FLAGS.chkpt_dir):
    U.mkdirs(FLAGS.chkpt_dir)
    print('Created checkpoint directory', FLAGS.chkpt_dir)
config.save_local_config(FLAGS)

#mode = FLAGS.run_mode
batch_size = FLAGS.batch_size
pid = FLAGS.item_id
essay_file = os.path.join(FLAGS.data_dir, '{0}', '{0}.txt.clean.tok').format(pid)

# ''' load Glove word embeddings, along with word vocab '''
# embed_file = FLAGS.embed_path.format(FLAGS.embed_dim)
# embed_matrix, word_vocab = Vocab.load_word_embeddings(embed_file, essay_file, min_freq=FLAGS.min_word_count)
# print(embed_matrix.shape)

vocab_file = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)

''' create essay reader, parser, & batcher '''
reader =  GlobReader(essay_file, chunk_size=10000, regex=REGEX_NUM, shuf=True, rng=U.rng)
text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length)

fields = {0:'id', 1:'y', -1:text_parser}
field_parser = FieldParser(fields, reader=reader)

batcher = EssayBatcher(reader=field_parser,
                       max_word_length=max_word_length,
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
    #inputs = tf.placeholder(tf.int32, shape=[batch_size, None], name="inputs")#                    <-- WORD
    inputs = tf.placeholder(tf.int32, shape=[batch_size, None, max_word_length], name="inputs")#    <-- CHAR
    num_unroll_steps = tf.shape(inputs)[1]
    
    targets = tf.placeholder(tf.float32,
                             shape=[batch_size, 1],
                             name="targets")
    
    ''' EMBEDDING LAYER '''
    ## char embedding ##
    char_embed = sm.CharEmbed(vocab_size=char_vocab.size,
                              embed_dim=FLAGS.char_embed_size, 
                              max_word_length=max_word_length,
                              #trainable=False,
                              name='char_embed_b')
    input_embedded = char_embed(inputs)
    
    ## tdnn ##
    tdnn = sm.TDNN(FLAGS.kernel_widths, FLAGS.kernel_features, initializer=0, name='TDNN')
    input_cnn = tdnn(input_embedded)
    
    ## highway ?? ##
    if FLAGS.num_highway_layers > 0:
        highway = sm.Highway(output_size=input_cnn.get_shape()[-1], num_layers=FLAGS.num_highway_layers)
        input_cnn = highway(input_cnn)
    
    ## reshape ##
    d1 = input_cnn.get_shape().as_list()[1]# 550 == sum(kernel_features)
    print('d1=={}'.format(d1))
    input_cnn = tf.reshape(input_cnn, [FLAGS.batch_size, num_unroll_steps, d1])
    
    ####################################################################################
    
    ''' RNN '''
    RNN = sm.DeepRNN#sm.MultiLSTM
    if FLAGS.bidirectional:
        RNN = sm.DeepBiRNN#sm.BiMultiLSTM
    rnn = RNN(rnn_size=FLAGS.rnn_dim,
              num_layers=FLAGS.rnn_layers,
              batch_size=FLAGS.batch_size,
              dropout=FLAGS.dropout,
              train_initial_state=FLAGS.train_initial_state,
              unit=FLAGS.rnn_unit
              )
    rnn_output, final_state = rnn(input_cnn)
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
    with tf.variable_scope('dense_1'):
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
    
    def training_op(notrain=[]):
        tvars = [v for v in tf.trainable_variables() if v.name.split('/')[1] not in notrain]
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), FLAGS.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        
    train_op_1 = training_op(['char_embed_b', 'TDNN'])
    train_op_2 = training_op(['char_embed_b'])
    train_op_3 = training_op()
    #@memoize
    def train_op(epoch):
        if epoch>4:#2
            return train_op_2#training_op(['char_embed_b'])
        if epoch>8:#6
            return train_op_3#training_op()
        return train_op_1#training_op(['char_embed_b', 'TDNN'])

############################################################################

graph = tf.get_default_graph()
names = []
nodes = graph.as_graph_def().node
for n in nodes:
    if 'Variable' in n.op:
        if n.name.startswith('Model/char_embed_b') or n.name.startswith('Model/TDNN'):
            names.append(n.name)
            print(n.name) #; print('=======================================')
                
############################################################################

''' TRAINING SESSION '''
graph = tf.get_default_graph()

with graph.as_default(), tf.Session() as sess:
    tf.set_random_seed(FLAGS.rand_seed)
    tf.global_variables_initializer().run()
    
    ########################################################################
    
    ## RESTORE FROM LM !!!!!!
    variables_to_restore = [var for var in tf.global_variables()
                            if var.name.startswith('Model/char_embed_b')
                            or var.name.startswith('Model/TDNN')]
    [print(var.name) for var in variables_to_restore]
    
    #####
#     for var in variables_to_restore:
#         if var.name.startswith('Model/TDNN/conv_2d/w'):
#             v1 = var; break
#     print(v1.eval())#;sys.exit(0)
    
    #####
    
    saver = tf.train.Saver(variables_to_restore)
    restore_chkpt = tf.train.latest_checkpoint(FLAGS.char_embed_chkpt)
    saver.restore(sess, restore_chkpt)
    
    #print(v1.eval())#;sys.exit(0)
    ########################################
    
    valid_batches, valid_ids, best_val_qwk = [],None,0
    epochs, epoch = FLAGS.epochs, 0
    while epoch < epochs:
        epoch+=1
        tic()
        print('==================================\tEPOCH {}\t\t=================================='.format(epoch))
        ##########################################################################
        ''' TRAINING '''
        losses, qwks, P, Y, batch = [],[],[],[],0
        for b in batcher.batch_stream(stop=True, skip_ids=valid_ids, c=True, w=False):
            batch+=1
            if epoch==1 and U.rng.rand()<FLAGS.valid_cut:
                valid_batches.append(b)
                continue
            
            feed_dict = { inputs: b.c, targets: b.y, seq_len: b.s, keep_prob: 1.0-FLAGS.dropout }
            fetches = [train_loss, train_op(epoch), train_qwk, preds]
            
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
        
        ## POST TRAIN EPOCH ##    
        if epoch==1:
            valid_ids = set([id for b in valid_batches for id in b.id])
            print('\n{} VALID-BATCHES ({} VALID-IDS)'.format(len(valid_batches),len(valid_ids)))
            #vid=list(valid_ids); vid.sort(); print(vid)
        
        word_count = batcher.word_count()
        sec = toc()
        wps = int(word_count/sec)
        QWK = U.nkappa(Y,P)#QWK = np.mean(qwks)
        train_msg = 'Epoch {0}\tTRAIN Loss : {1:0.4}\tTRAIN Kappa : {2:0.4}\t{3:0.2g}min|{4}wps'.format(epoch, np.mean(losses), QWK, float(sec)/60.0, wps)
        print('\n[CURRENT]\t' + train_msg)
        
        ##########################################################################
        ''' VALIDATION '''
        losses, qwks, P, Y, batch = [],[],[],[],0
        for b in valid_batches:
            batch+=1
            feed_dict = { inputs: b.c, targets: b.y, seq_len: b.s, keep_prob: 1.0 }
            loss, kappa, p = sess.run( [train_loss, train_qwk, preds], feed_dict)
            losses.append(loss)
            qwks.append(kappa)
            P.extend(np.squeeze(p)); Y.extend(np.squeeze(b.y)); kappa = U.nkappa(Y,P)
            
            if batch % FLAGS.print_every == 0:
                sys.stdout.write('\tmse={0:0.4}'.format(loss))
                sys.stdout.write('|qwk={0:0.4}\n'.format(kappa))
                sys.stdout.flush()
                
        QWK = U.nkappa(Y,P)
        val_msg = 'Epoch {0}\tVALID Loss : {1:0.4}\tVALID Kappa : {2:0.4}'.format(epoch, np.mean(losses), QWK)
        if QWK>best_val_qwk:
            best_val_qwk=QWK
            best_val_msg=val_msg
        print('[CURRENT]\t' + val_msg)
        print('[BEST]\t\t' + best_val_msg + '\n')
