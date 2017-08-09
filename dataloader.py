import numpy as np
import nltk
import itertools
# https://stackoverflow.com/questions/4142151/how-to-import-the-class-within-the-same-directory-or-sub-directory
from utils_training_gen import Relation, Sentence, Training_sample
import re
from nltk.tokenize import wordpunct_tokenize
import pdb


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
        
        self.load_relations()
        self.load_word2vec()
        
        

    def load_word2vec(self):
        #load word2vec from file
        #Two data structure: word2index, index2vector
        # wordvector = list(open("../data/vector1.txt", "r").readlines())
        wordvector = list(open(self.word2vec_file, "r").readlines())
        wordvector = [s.split() for s in wordvector]
        ###### New line added to remove the first line from word2vec file
        _ = wordvector.pop(0)
        ######
        self.wordvector_dim = len(wordvector[0])-1
        self.word2index["UNK"] = 0
        #################
        self.index2vector.append(np.zeros(self.wordvector_dim)) ### Might want to put random
        #################
        index = 1
        for vec in wordvector:
            a = np.zeros(self.wordvector_dim)
            for i in range(self.wordvector_dim):
                a[i] = float(vec[i+1])
            self.word2index[vec[0]] = index
            self.index2vector.append(a)
            index += 1

        print("WordTotal: ", len(self.index2vector))
        print("Word dimension: ", self.wordvector_dim)

    def load_relations(self):
        #load relation from file
        # relation_data = list(open("../data/RE/relation2id.txt").readlines())
        relation_data = list(open(self.relation2id_file).readlines())
        relation_data = [s.split() for s in relation_data]
        for relation in relation_data:
            r = Relation(relation[0], int(relation[1]))
            self.relations[relation[0]] = r
        for r in self.relations:
            self.relations[r].generate_vector(len(self.relations)) # Generates one-hot representation
        print("RelationTotal: "+str(len(self.relations)))

    def load_training_data_for_Gen(self):
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
        print("Start loading training data for Generator.")
        print("====================")
        training_data = list(open(self.training_file).readlines())
        training_data = [s.split() for s in training_data]
        # training_data = [wordpunct_tokenize(s) for s in training_data]
        relation_data = list(open(self.relation2id_file).readlines())
        relation_data = [s.split() for s in relation_data]
        word_id = list(open(self.vocab2id_file).readlines())
        word_id = [s.split() for s in word_id]
        relation2id = {}
        for relation in relation_data:
            relation2id[relation[0]]=relation[1]
        word2id={}
        for word in word_id:
            word2id[word[0]]=word[1]
        return_data=[]
        for data in training_data:
            ## Get the positions of entities
            entity1p,entity2p=get_entity_loc(data[2], data[3], data[5:-1])
            ## Relation
            if data[4] not in relation2id.keys():
                relation_idx = relation2id["NA"]
            else:
                relation_idx = relation2id[data[4]]    
            #Get the word_idx
            words2index=[]
            # for word in clean_str(str(data[5:-1])).split():
            for word in data[5:-1]:
                if word not in word2id.keys():
                    word = 'UNK'
                words_idx=word2id[word]
                words2index.append(words_idx)            
            words2index=[int(word) for word in words2index]
            #words2index=self.padding(words2index)
            # Finally !
            training_sample = Training_sample(relation_idx,words2index,entity1p,entity2p)
            return_data.append(training_sample)
        # return return_data.__iter__()
        return return_data

    
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
    def generate_y(self, data)
        ##################
        
        ##################
        return (y, padding)
    
    def word2num(self, words):
        return [words2index[w] for w in words]
    
    ########## Misc stuff ##########
    ################################
    def padding(self, vectors):
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
