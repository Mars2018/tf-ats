import os
import configargparse
import pprint

from nlp.util import config

def get_parser(default_config=None):
    p = configargparse.ArgParser()#default_config_files=['config/model.conf'])
    
#     if default_config==None:
#         default_config = 'config/model.conf'
        
    p.add('--config', required=False, is_config_file=True, default=default_config, help='config file path')
    
    p.add("-id", "--item_id", type=str, required=False, help="itemID or modeID")
    p.add("--data_dir", type=str, required=False, help="The path to the data directory")
    p.add("--chkpt_dir", type=str, required=False, help="The path to the checkpoint directory")
    
    p.add("--embed_path", type=str, required=False, help="The path to the glove embeddings")
    p.add("-e", "--embed_dim", type=int, default=50, help="Embeddings dimension (default=50)")
    p.add("--min_word_count", type=int, default=2, help="Min word frequency")
    
    p.add("-b", "--batch_size", type=int, default=32, help="Batch size (default=32)")
    p.add("--learning_rate", type=float, default=0.001, help="")
    p.add("--dropout", type=float, default=0.0, help="The dropout probability. To disable, give a negative number (default=0.5)")
    p.add("--max_grad_norm", type=float, default=1000.0, required=False, help="")
    p.add("--epochs", type=int, default=100, help="Number of epochs (default=50)")
    p.add("--optimizer", type=str, default='adam', required=False, help="optimizer")
    p.add("--valid_cut", type=float, default=0.1, required=False, help="")
    
    p.add("-r", "--rnn_dim", type=int, default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
    p.add("-u", "--rnn_unit", type=str, default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
    p.add("-rl","--rnn_layers", type=int, default=2, help='number of layers in the LSTM')#2
    p.add("--bidirectional", action='store_true', help="")

    p.add("--mean_pool", action='store_true', help="")
    p.add("--skip_connections", action='store_true', help="")
    p.add("--peepholes", action='store_true', help="")
    
    p.add("--rand_seed", type=int, default=0, help="")
    
    
#     p.add("--maxlen", type=int, default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
#     p.add("--aggregation", type=str, default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
#     p.add("--num_highway_layers", type=int, default=0, help="Number of highway layers")
#     p.add("-a", "--algorithm", type=str, default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
#     p.add("-l", "--loss", type=str, default='kappa', help="Loss function (mse|kappa|mae) (default=kappa)")
#     p.add("-v", "--vocab_size", type=int, default=4000, help="Vocab size (default=4000)")
#     p.add("-c", "--cnn_dim", type=int, default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
#     p.add("-w", "--cnn_window_size", type=int, default=3, help="CNN window size. (default=3)")
#     p.add("--stack", type=int, default=1, help="how deep to stack core RNN")
#     p.add("--skip_emb_preload", action='store_true', help="Skip preloading embeddings")
#     p.add("--tokenize_old", action='store_true', help="use old tokenizer")
#     p.add("--run_mode", type=str, default='train', required=False, help="train/valid")
#     p.add("--vocab_path", type=str, help="(Optional) The path to the existing vocab file (*.pkl)")
#     p.add("--skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")

#     dir_path = os.path.dirname(os.path.realpath(__file__))# get path of this file
#     p.add("--code_dir", type=str, default=dir_path, required=False, help="The path to the code directory")
    
    return p


if __name__ == "__main__":
    
    parser = get_parser()
    
    config_file = None
    #config_file = 'chkpt/mod1_650-20/model.conf'
    #config_file = 'config/ats.conf'
    config_file = 'config/mode.conf'
    
    FLAGS = config.get_config(config_file=config_file, parser=parser)
    
    pprint.pprint(FLAGS)
