import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
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

flags = tf.app.flags
pp = pprint.PrettyPrinter().pprint

data_dir = '/home/rohanjain/data/sentgen/'
# tf.app.flags.DEFINE_string('data_dir', '/home/rohanjain/data/sentgen/', 'where should we save')
# pdb.set_trace()
tf.app.flags.DEFINE_string('training_file', os.path.join(data_dir, 'RE/train.txt'), 'where should we save')
tf.app.flags.DEFINE_string('word2vec_file', os.path.join(data_dir, 'vec.txt'), 'where should we save')
tf.app.flags.DEFINE_string('relation2id_file', os.path.join(data_dir, 'RE/relation2id.txt'), 'where should we save')
tf.app.flags.DEFINE_string('entity2id_file', os.path.join(data_dir, 'RE/entity2id.txt'), 'where should we save')
tf.app.flags.DEFINE_string('vocab2id_file', './vocab2id.txt', 'where should we save')


tf.app.flags.DEFINE_string('model_path', './models/', 'where should we save')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch_size for each iterations')
tf.app.flags.DEFINE_integer('rel_embed_dim', 100, 'word embedding size')
tf.app.flags.DEFINE_integer('embed_dim', 50, 'word embedding size')
tf.app.flags.DEFINE_integer('hidden_dim', 100, 'hidden size')
tf.app.flags.DEFINE_integer('sequence_length', 100, 'max length of sentence')
tf.app.flags.DEFINE_integer('n_epochs', 250, 'how many epochs are we going to train')
tf.app.flags.DEFINE_float('learning_rate', '0.001', 'learning rate for adam')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum for adam')
tf.app.flags.DEFINE_float('ss_threshold', 0.5, 'scheduled sampling threshold prob')
tf.app.flags.DEFINE_boolean('is_train', 'True', 'momentum for adam')

attrs = flags.FLAGS

def main(_):
    conf = attrs.__dict__['__flags']
    pp(conf)

    with tf.Session() as sess:
        # model = models.SentenceGenerator(sess, conf) # change the arguments

        if conf['is_train']:
            datam=DataManager(conf)
            
            start_time = time.time()
            training_data=datam.load_training_data_for_Gen()
            print("Time taken to load  and clean data: {}".format(time.time()-start_time))
            # utils.pickle_write(training_data, './training_data.pkl')
            
            # print('loading previously prepared training data')
            # training_data = utils.pickle_read('./training_data.pkl')
            
            model = models.SentenceGenerator(sess, conf, datam.vocab_size, datam.num_rel)
            model.build_model()
            
            model.train(sess, datam, training_data)
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