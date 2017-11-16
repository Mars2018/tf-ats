from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import pprint
import numpy as np
import pickle as pk
import tensorflow as tf

from nlp.util import config, ets_reader
from nlp.util import utils as U
from nlp.util.w2vEmbReader import W2VEmbReader


import options

def make_abs(path):
    return os.path.abspath(path)

''' get config '''    
parser = options.get_parser()
config_file = 'config/mode.conf'
argv=[]# override config file here
FLAGS = config.get_config(parser=parser, config_file=config_file, argv=argv)
FLAGS.chkpt_dir = make_abs(FLAGS.chkpt_dir)
FLAGS.data_dir = os.path.join(FLAGS.data_dir, FLAGS.item_id)
pprint.pprint(FLAGS)

''' setup checkpoint directory '''
if not os.path.exists(FLAGS.chkpt_dir):
    U.mkdirs(FLAGS.chkpt_dir)
    print('Created checkpoint directory', FLAGS.chkpt_dir)
config.save_local_config(FLAGS)

# ''' setup logger (???) '''
# U.set_logger(FLAGS.chkpt_dir)

''' random seed '''
rand_seed = U.seed_random(FLAGS.seed)

pid = FLAGS.item_id
mode = FLAGS.run_mode

''' load GLOVE word embeddings '''
emb_words = None
if not FLAGS.skip_emb_preload:
    #logger.info('Loading embedding vocabulary...')
    emb_reader = W2VEmbReader(FLAGS.glove_path, emb_dim=FLAGS.emb_dim)
    emb_words = emb_reader.load_words()

vocab_path = None
abs_vocab_file = os.path.join(FLAGS.chkpt_dir, 'vocab.pkl')
if mode=='test':
    vocab_path = abs_vocab_file
    
''' load essay data '''
train_df, dev_df, test_df, vocab, overal_maxlen = ets_reader.get_mode_data( FLAGS.data_dir,
                                                                            dev_split=0.1,
                                                                            emb_words=emb_words,
                                                                            vocab_path=vocab_path,
                                                                            seed=rand_seed)
train_x = train_df['text'].values;  train_y = train_df['yint'].values.astype('float32')
dev_x = dev_df['text'].values;      dev_y = dev_df['yint'].values.astype('float32')
test_x = test_df['text'].values;    test_y = test_df['yint'].values.astype('float32')

''' dump vocab '''
if mode=='train':
    with open(abs_vocab_file, 'wb') as vocab_file:
        pk.dump(vocab, vocab_file)














# if __name__ == "__main__":