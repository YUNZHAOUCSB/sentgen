import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import tensorflow as tf
import numpy as np
import pickle
import pprint
import tensorflow.python.platform
from keras.preprocessing import sequence
# from dataloader import *
from dataloader import Gen_data_loader, Dis_data_loader
from dataloader import DataManager



flags = tf.app.flags
pp = pprint.PrettyPrinter().pprint
'''
tf.app.flags.DEFINE_string('input_ques_h5', './data_prepro.h5', 'path to the h5file containing the preprocessed dataset')
tf.app.flags.DEFINE_string('input_json', './data_prepro.json', 'path to the json file containing additional info and vocab')
tf.app.flags.DEFINE_string('model_path', './models/', 'where should we save')
tf.app.flags.DEFINE_string('vgg_path', './vgg16.tfmodel', 'momentum for adam')
tf.app.flags.DEFINE_string('gpu_fraction', '2/3', 'define the gpu fraction used')
tf.app.flags.DEFINE_string('test_image_path', './assets/demo.jpg', 'the image you want to generate question')
tf.app.flags.DEFINE_string('test_model_path', './models/model-250', 'model we saved')

tf.app.flags.DEFINE_integer('batch_size', 256, 'tch_size for each iterations')
tf.app.flags.DEFINE_integer('dim_embed', 50, 'word embedding size')
tf.app.flags.DEFINE_integer('dim_hidden', 100, 'hidden size')
tf.app.flags.DEFINE_integer('maxlen', 26, 'max length of sentence')
tf.app.flags.DEFINE_integer('n_epochs', 250, 'how many epochs are we going to train')
tf.app.flags.DEFINE_float('learning_rate', '0.001', 'learning rate for adam')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum for adam')
tf.app.flags.DEFINE_boolean('is_train', 'True', 'momentum for adam')

attrs = flags.FLAGS
'''

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
# positive_file = 'save/real_data.txt'
training_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000
#vocab_size = 5000 # should be set in dataloader

conf['word2vec_file'] = os.path.join(data_dir, 'vec.txt')
conf['relation2id_file'] = os.path.join(data_dir, 'RE/relation2id.txt')
conf['entity2id_file'] = os.path.join(data_dir, 'RE/entity2id.txt')
conf['sequence_length'] = 100
conf['vocab2id_file'] = './vocab2id.txt'
conf['training_file'] = os.path.join(data_dir, 'RE/train.txt')

def main(_):
    conf = attrs.__dict__['__flags']
    pp(conf)

    with tf.Session() as sess:
        model = model.SentenceGenerator(sess, conf) # change the arguments

        if conf.is_train:
            datam=Datamanager(conf)
            # training_data=datam.load_training_data_for_Gen(training_file, relation2id_file, word2id_file)
            model.build_model(datam.vocab_size)
            model.train(sess, datam)
        '''    
        else:
            model.build_generator()
            model.test(test_image_path=conf.test_image_path, model_path=conf.test_model_path, maxlen=26)
        '''
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