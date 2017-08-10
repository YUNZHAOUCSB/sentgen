import os
import tensorflow as tf
import numpy as np
import tensorflow.python.platform
from keras.preprocessing import sequence
from utils import batch_iter
import random
random.seed=8
import pdb

class SentenceGenerator(object):
    def __init__(self, sess, conf, vocab_size, num_rel, start_token=0,
                 learning_rate=0.01, reward_gamma=0.95):
        self.sess = sess
        self.num_epochs = conf['n_epochs']
        # self.num_emb = conf['num_emb']
        self.batch_size = conf['batch_size']
        self.rel_emb_dim = conf['rel_embed_dim']
        self.emb_dim = conf['embed_dim']
        self.hidden_dim = conf['hidden_dim']
        self.lstm_state_size = conf['hidden_dim'] # lstm state size is same as hidden dimension in our recurrent unit
        self.sequence_length = conf['sequence_length']
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.g_params = []
        # self.vocab_size =  #########Set by data manager. Sent when calling build_model#######
        self.n_lstm_steps = conf['sequence_length'] #########!!!!!!!#######
        self.learning_rate = conf['learning_rate']
        self.ss_threshold = conf['ss_threshold']
        self.vocab_size = vocab_size
        self.num_rel = num_rel
        
        # with tf.variable_scope('generator'):
        self.rel_embeddings = tf.Variable(self.init_matrix([self.num_rel, self.rel_emb_dim])) # Relation embedding
        self.g_params.append(self.rel_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
        # h_tm1 mean h_{t-1}
        self.g_output_unit = self.create_output_unit(self.g_params, self.vocab_size, 'output_token')  # maps h_t to o_t (output token logits)
        self.g_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
        self.g_params.append(self.g_embeddings)
        self.g_output_unit_entity1 = self.create_output_unit(self.g_params, self.n_lstm_steps, 'output_entity1')
        self.g_output_unit_entity2 = self.create_output_unit(self.g_params, self.n_lstm_steps,'output_entity2')
        pdb.set_trace()
                
    def build_model(self):
        ######
        self.rel = tf.placeholder(tf.int32, shape=[self.batch_size]) # Batch of relation seeds
        self.sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps]) # labels
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        self.p1 = tf.placeholder(tf.int32, [self.batch_size]) #p1_locations
        self.p2 = tf.placeholder(tf.int32, [self.batch_size]) #p2_locations
        loss = 0.0
        for i in range(self.n_lstm_steps):
            print("LSTM step i = ", i)
            if i==0:
                # Initial states
                c_prev = tf.zeros([self.batch_size, self.lstm_state_size]) 
                h_tm1 = tf.nn.embedding_lookup(self.rel_embeddings, self.rel) # + self.bemb # (batch_size, relation_emb_siz)
                print('c_prev shape: ', c_prev.get_shape())
                print('h_tm1 shape: ', h_tm1.get_shape())
                # pdb.set_trace()
                h_tm1 = tf.stack([h_tm1, c_prev])
                x_t = tf.nn.embedding_lookup(self.g_embeddings,self.start_token) # start token embedding
            else:
                # tf.get_variable_scope().reuse_variables()
                h_tm1 = h_t
                ###### Scheduled sampling #########
                if random.random()>self.ss_threshold:
                    x_t = x_tp1
                else:
                    x_t = tf.nn.embedding_lookup(self.g_embeddings,self.sentence[:,i-1])
                ###################################
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t) # logits  
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob,1),[self.batch_size]),tf.int32)
            x_tp1=tf.nn.embedding_lookup(self.g_embeddings,next_token)
                
            # ground truth
            if i >= 0:
                '''
                labels = tf.expand_dims(self.sentence[:, i], 1) # (batch_size,1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size,1)
                # concated = tf.concat(1, [indices, labels]) # (batch_size,2)
                concated = tf.concat([indices, labels], 1) # (batch_size,2)
                onehot_labels = tf.sparse_to_dense(
                    concated, tf.pack([self.batch_size, self.vocab_size]), 1.0, 0.0) 
                '''
                labels = self.sentence[:, i] # (batch_size)
                onehot_labels = tf.one_hot(labels, self.vocab_size)
                # pdb.set_trace()
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=o_t, labels = onehot_labels)
                cross_entropy = cross_entropy * self.mask[:,i]                             

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss
        # self.loss = loss / tf.reduce_sum(self.mask[:,1:])
        self.loss = loss / tf.reduce_sum(self.mask)
        o_e1_t = self.g_output_unit_entity1(h_t)
        o_e2_t = self.g_output_unit_entity2(h_t)
        # Calculate cross entropy losses for the two entities and add it to self.loss
        ##### e1 loss #######
        '''
        labels_e1 = tf.expand_dims(self.p1, 1) # (batch_size,1)
        indices_e1 = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size,1)
        # concated = tf.concat(1, [indices, labels]) # (batch_size,2)
        concated_e1 = tf.concat([indices_e1, labels_e1], 1) # (batch_size,2)
        onehot_labels_e1 = tf.sparse_to_dense(
            concated_e1, tf.pack([self.batch_size, self.n_lstm_steps]), 1.0, 0.0) 
        '''
        labels_e1 = self.p1
        onehot_labels_e1 = tf.one_hot(labels_e1, self.n_lstm_steps)
        cross_entropy_e1 = tf.nn.softmax_cross_entropy_with_logits(logits = o_e1_t, labels = onehot_labels_e1)
        self.loss += tf.reduce_sum(cross_entropy_e1)
        ##### e2 loss #######
        '''
        labels_e2 = tf.expand_dims(self.p2, 1) # (batch_size,1)
        indices_e2 = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size,1)
        # concated = tf.concat(1, [indices, labels]) # (batch_size,2)
        concated_e2 = tf.concat([indices_e2, labels_e2], 1) # (batch_size,2)
        onehot_labels_e2 = tf.sparse_to_dense(
            concated_e2, tf.pack([self.batch_size, self.n_lstm_steps]), 1.0, 0.0) 
        '''
        labels_e2 = self.p2
        onehot_labels_e2 = tf.one_hot(labels_e2, self.n_lstm_steps)
        cross_entropy_e2 = tf.nn.softmax_cross_entropy_with_logits(logits = o_e2_t, labels = onehot_labels_e2)
        self.loss += tf.reduce_sum(cross_entropy_e2)
        
    def train(self, datam, training_data):
        # index = np.arange(self.num_train)
        # np.random.shuffle(index)
        # training_data=datam.load_training_data_for_Gen(conf['training_file'], conf['relation2id_file'], conf['vocab2id_file'])
        
        sess = self.sess
        
        self.saver = tf.train.Saver(max_to_keep=50)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        sess.run(tf.global_variables_initializer())
        
        
        for epoch in range(self.n_epochs):
            # counter = 0
            # Load relevant batch data and set it to feed_dict
            # batch_itrr=
            for batch_data in batch_iter(training_data):
                ## Load batch data
                relations = generate_x(batch_data)
                p1, p2 = generate_p(batch_data)
                sentences, masks = generate_y(batch_data)
                ##
                feed_dict ={self.rel: relations,
                           self.sentence: sentences,
                           self.mask: masks,
                           self.p1: p1,
                           self.p2: p2}
                loss = tf.sess.run(self.loss, feed_dict)
            if epoch % 5 == 0:
                print('epoch', epoch, 'loss', loss)
            
    
    def generate_x(self, training_data):
        return [data.relation_idx for data in training_data]
    
    def generate_p(self, training_data):
        return ([data.entity1p for data in training_data], [data.entity2p for data in training_data])
    
    def generate_y(self, training_data):
        
        return 
        
    #### Utility functions ####    
    def init_matrix(self, shape, **kwargs):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
            # pdb.set_trace()
            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) + 
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params, outsize, scope_name):
        with tf.variable_scope(scope_name, reuse=True):
            # Wo = tf.Variable(self.init_matrix([self.hidden_dim, outsize]), name="Wo")
            # bo = tf.Variable(self.init_matrix([outsize]), name = "bo")
            Wo = tf.get_variable(name="Wo", shape=[self.hidden_dim, outsize], initializer=self.init_matrix)
            bo = tf.get_variable(name="bo", shape=[outsize], initializer=self.init_matrix)
        params.extend([Wo, bo])

        def unit(hidden_memory_tuple):
            with tf.variable_scope(scope_name):
                Wo = tf.get_variable(name="Wo")
                bo = tf.get_variable(name = "bo")
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, Wo) + bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit