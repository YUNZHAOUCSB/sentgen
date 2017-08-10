import numpy as np
import nltk
import itertools
# https://stackoverflow.com/questions/4142151/how-to-import-the-class-within-the-same-directory-or-sub-directory
from utils_training_gen import Relation, Sentence, Training_sample
import re
from nltk.tokenize import wordpunct_tokenize
import pdb
import pandas as pd


class DataManager:
    def __init__(self, conf):
        self.wordvector_dim = 50 # should be 50. should be read from vec.txt
        self.sequence_length = conf['sequence_length']
        self.word2index = {}
        self.index2vector = []
        self.relations = {}
        
        self.training_file = conf['training_file']
        self.relation2id_file = conf['relation2id_file']
        self.entity2id_file = conf['entity2id_file']
        self.word2vec_file = conf['word2vec_file']
        self.vocab2id_file = conf['vocab2id_file']

    def load_training_data_for_Gen(self):
        print("Start loading training data for Generator.")
        training_data = []
        #########################################################################################
        #calculate the entity locations from Sentence object
        #########################################################################################
        def get_entity_loc(e1,e2,d):
            # e1 = d.entity1
            # e2 = d.entity2
            # words = d.split()
            words = d
            l1 = 0
            l2 = 0
            for i, w in enumerate(words):
                if w == e1:
                    l1 = i
                if w == e2:
                    l2 = i
            return (l1,l2)        
        #########################################################################################
        training_data = list(open(self.training_file).readlines())
        training_data = [s.split() for s in training_data]
        # training_data = [wordpunct_tokenize(s) for s in training_data]
        ############### load relations ##################
        relation_data = list(open(self.relation2id_file).readlines())
        relation_data = [s.split() for s in relation_data]
        self.relation2id = {}
        for relation in relation_data:
            self.relation2id[relation[0]]=relation[1]
        print("RelationTotal: "+str(len(self.relation2id)))
        self.num_rel = len(self.relation2id)
        ############### load words ######################
        word_id = list(open(self.vocab2id_file).readlines())
        word_id = [s.split() for s in word_id]        
        self.word2id={}
        for word in word_id:
            self.word2id[word[0]]=word[1]
        self.vocab_size = len(self.word2id.keys())
        ##############   Get words2index for each training sample    ####################
        return_data=[]
        for data in training_data:
            ## Get the positions of entities
            entity1p,entity2p=get_entity_loc(data[2], data[3], data[5:-1])
            ## Relation
            if data[4] not in self.relation2id.keys():
                relation_idx = self.relation2id["NA"]
            else:
                relation_idx = self.relation2id[data[4]]    
            #Get the word_idx
            words2index=[]
            # for word in clean_str(str(data[5:-1])).split():
            for word in data[5:-1]:
                if word not in self.word2id.keys():
                    word = 'UNK'
                words_idx=self.word2id[word]
                words2index.append(words_idx)            
            words2index=[int(word) for word in words2index]           
            # Finally !
            training_sample = Training_sample(relation_idx,words2index,entity1p,entity2p)
            return_data.append(training_sample)
        # return return_data.__iter__()
        return self.clean_data(return_data)

    def clean_data(self, data):
        print("Start cleaning training data for Generator.")
        num = self.sequence_length
        seq_len = [len(item.words_idx) for item in data]
        ent1p = [item.entity1p for item in data]
        ent2p = [item.entity2p for item in data]
        seq_data = pd.DataFrame([seq_len, ent1p, ent2p]).T
        seq_data.columns=['sentence_length', 'entity1_pos', 'entity2_pos']
        data_toberemoved = seq_data.loc[(seq_data.sentence_length>num)&((seq_data.entity1_pos>num)|(seq_data.entity2_pos>num))]
        print('***')
        print(len(set(data_toberemoved.index)))
        print(len(data))
        filtered_data = [i for j, i in enumerate(data) if j not in set(data_toberemoved.index)]
        print("Done loading and cleaning training data for Generator.")
        return filtered_data
    
    def load_training_data(self, entity2id_file, filename="../data/RE/train.txt", distant_supervision=True):
        """
        load training data from file
        Used by make_vocab
        """
        f = open(entity2id_file, "a") # USELESS right now
        print("Start loading training data.")
        print("====================")
        training_data = list(open(filename).readlines())
        training_data = [s.split() for s in training_data]
        # training_data = [wordpunct_tokenize(s) for s in training_data]
        for data in training_data:
            entity1 = data[2]
            entity2 = data[3]
            if data[4] not in self.relations:
                relation = self.relations["NA"]
            else:
                relation = self.relations[data[4]]
            s = Sentence(entity1,
                         entity2,
                         relation,
                         data[5:-1])
            self.training_data.append(s)
        return self.training_data
    
            
    def load_testing_data(self, filename="../data/RE/test.txt"):
        """
        load training data from file
        Haven't used this till now
        """
        print("Start loading testing data.")
        print("====================")
        testing_data = list(open(filename).readlines())
        testing_data = [s.split() for s in testing_data]
        #for data in testing_data:
        for data in testing_data:
            entity1 = data[2]
            entity2 = data[3]
            if data[4] not in self.relations:
                relation = self.relations["NA"]
            else:
                relation = self.relations[data[4]]
            s = Sentence(entity1,
                         entity2,
                         relation,
                         data[5:-1])
            if data[0]+"\t"+data[1] not in self.bags_test:
                self.bags_test[entity1+" "+entity2] = [s]
            else:
                self.bags_test[entity1+" "+entity2].append(s)
        return self.bags_test

    def relation_analyze(self):
        for r in self.relations:
            print(r+": "+str(self.relations[r].number))

    def batch_iter(self, data, batch_size, shuffle=True):
        """
        Generates a batch iterator for a dataset. Do we need to shuffle again here?
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        #Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index == end_index:
                continue
            else:
                yield shuffled_data[start_index:end_index]    
    def generate_x(self, data):
        return [elem.relation_idx for elem in data]
    
    def generate_p(self, data):
        return ([elem.entity1p for elem in data], [elem.entity2p for elem in data])
    def generate_y(self, data):
        ##################
        seq_len = [len(item.words_idx) for item in data] # useful for creating mask
        seq = [item.words_idx for item in data]
        y = self.padding(seq)
        mask = self.mask(y)                
        ##################
        return (y, mask)
    
    def word2num(self, words):
        return [words2index[w] for w in words]
    
    ########## Misc stuff ##########
    ################################
    def mask(self, padded_seq):
        updated_vectors = []
        for seq in padded_seq:
            new_seq = [0 if x == self.word2id['PAD'] else 1 for x in seq]
            updated_vectors.append(seq)
        return updated_vectors
    
    def padding(self, vectors):
        updated_vectors = []
        for vector in vectors:
            a = self.sequence_length-len(vector)
            if a > 0:
                # front = a/2
                front = 0
                back = a-front
                pad_token_id = self.word2id['PAD']
                # front_vec = [np.zeros for i in range(front)]
                front_vec = []
                # back_vec = [np.zeros(self.wordvector_dim) for i in range(back)]
                back_vec = [pad_token_id]*back
                vectors = front_vec + vector + back_vec
            else:
                vector = vector[:self.sequence_length] # TRUNCATING
            updated_vectors.append(vector)
        return updated_vectors
    
    def batch_iter_old(self, data, batch_size, num_epochs, shuffle=False):
        """
        Generates a batch iterator for a dataset. Do we need to shuffle again here?
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            #Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if start_index == end_index:
                    continue
                else:
                    yield shuffled_data[start_index:end_index]
                    #return shuffled_data[start_index:end_index]
                    
    def generate_x_old(self, data):
        x = []
        for d in data:
            v = []
            words = d.words
            e1 = d.entity1
            e2 = d.entity2
            for i, w in enumerate(words):
                if w not in self.word2index:
                    tmp = self.index2vector[0]
                else:
                    tmp = self.index2vector[self.word2index[w]]
                v.append(tmp)
            vectors = self.padding(v)
            x.append(vectors)
        return x

    def generate_y_old(self, data):
        return [d.relation.vector for d in data] # what is relation.vector --> one hot representation

    def generate_p_old(self, data):
        p1 = []
        p2 = []
        for d in data:
            p11 = []
            p22 = []
            e1 = d.entity1
            e2 = d.entity2
            words = d.words
            l1 = 0
            l2 = 0
            for i, w in enumerate(words):
                if w == e1:
                    l1 = i
                if w == e2:
                    l2 = i
            for i, w in enumerate(words):
                a = i-l1
                b = i-l2
                if a > 30:
                    a = 30
                if b > 30:
                    b = 30
                if a < -30:
                    a = -30
                if b < -30:
                    b = -30
                p11.append(a+31)
                p22.append(b+31)
            a = self.sequence_length-len(p11)
            if a > 0:
                # front = a/2
                front = a//2
                back = a-front
                front_vec = [0 for i in range(front)]
                back_vec = [0 for i in range(back)]
                p11 = front_vec + p11 + back_vec
                p22 = front_vec + p22 + back_vec
            else:
                p11 = p11[:self.sequence_length]
                p22 = p22[:self.sequence_length]
            p1.append(p11)
            p2.append(p22)
        return p1, p2
    
    def padding_old(self, vectors):
        a = self.sequence_length-len(vectors)
        if a > 0:
            # front = a/2
            front = a//2
            back = a-front
            front_vec = [np.zeros(self.wordvector_dim) for i in range(front)]
            back_vec = [np.zeros(self.wordvector_dim) for i in range(back)]
            vectors = front_vec + vectors + back_vec
        else:
            vectors = vectors[:self.sequence_length] # TRUNCATING
        return vectors


def __init__():
    return 0
