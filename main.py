import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import numpy as np
import pickle
import pprint
import tensorflow.python.platform
# from keras.preprocessing import sequence
from dataloader import DataManager
import pdb
import models
import utils
import time
import argparse

parser = argparse.ArgumentParser(description='SENTGEN: Generator pretraining and generating')

data_dir = '/home/rohanjain/data/sentgen/'

parser.add_argument('--training_file', default=os.path.join(data_dir, 'RE/train.txt'), type=str, help='training_file')
parser.add_argument('--word2vec_file', default=os.path.join(data_dir, 'vec.txt'), type=str, help='where should we save')
parser.add_argument('--relation2id_file', default=os.path.join(data_dir, 'RE/relation2id.txt'), type=str, help='where should we save')
parser.add_argument('--entity2id_file', default=os.path.join(data_dir, 'RE/entity2id.txt'), type=str, help='where should we save')
parser.add_argument('--vocab2id_file', default='./vocab2id.txt', type=str, help='where should we save')
parser.add_argument('--outfile', default='./generated_sentences.txt', type=str, help='where should we save generated sentences')

parser.add_argument('--model_path', default='./models/', type=str, help='where should we save')
parser.add_argument('--batch_size', default=256, type=int, help='batch_size for each iterations')
# parser.add_argument('--rel_embed_dim', default=120, type=int, help='word embedding size')
parser.add_argument('--embed_dim', default=50, type=int, help='word embedding size')
parser.add_argument('--hidden_dim', default=120, type=int, help='hidden size')
parser.add_argument('--sequence_length', default=100, type=int, help='max length of sentence')
parser.add_argument('--n_epochs', default=250, type=int, help='how many epochs are we going to train')
parser.add_argument('--vocab_size', default=5003, type=int, help='total vocabulary size including PAS, UNK and START')
parser.add_argument('--num_rel', default=53, type=int, help='total number of relation including UNK')
parser.add_argument('--learning_rate', '--lr', default=0.001, type=float, help='learning rate for adam')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for adam')
parser.add_argument('--ss_threshold', default=0.5, type=float, help='scheduled sampling threshold prob')
parser.add_argument('--e','--evaluate', dest='evaluate', action='store_true', help='To train or to evaluate/sample')
parser.add_argument('--prepro','--preprocessed', '--load_preprocessed_training_data', dest='preprocessed', action='store_true', help='whether to preprocess raw training data')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

def main(_):    
    global args, best_prec1
    conf = parser.parse_args()
    with tf.Session() as sess:
        model = models.SentenceGenerator(sess, conf)
        datam=DataManager(conf)   
        if not conf.evaluate:         
            if conf.preprocessed:
                start_time = time.time()
                training_data, relation2id, word2id = datam.load_training_data_for_Gen()
                print("Time taken to load  and clean training data: {}".format(time.time()-start_time))
                utils.pickle_write([training_data, relation2id, word2id], './training_data.pkl')
            else:
                print('Loading preprocessed training data...')
                training_data, relation2id, word2id = utils.pickle_read('./training_data.pkl')
                assert conf.num_rel == len(relation2id.keys())
                assert conf.vocab_size == len(word2id.keys())
                datam.set_relationword_id(relation2id, word2id)
            model.build_model()
            print('Model built successful')
            model.train(datam, training_data)            
        else:
            start_time = time.time()
            testing_data, relation2id, word2id = datam.load_testing_data_for_Gen()
            print("Time taken to load and clean testing data: {}".format(time.time()-start_time))
            model.build_generator()
            # model.test(datam, testing_data)
        
    '''
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    gen_data_loader.create_batches(positive_file)
    '''
    ######
    
    
    
if __name__ == '__main__':
    tf.app.run()