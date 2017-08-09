import numpy as np
import nltk
import itertools


class Gen_data_loader():
    def __init__(self, sequence_length, word2vec_file, relation2id_file):
        self.wordvector_dim = 0
        self.sequence_length = sequence_length
        self.word2index = {}
        self.index2vector = []
        self.relations = {}
        # self.bags_train = {}
        self.training_data = []
        # self.bags_test = {}
        self.load_relations(relation2id_file) # Will set self.relations
        self.load_word2vec(word2vec_file) # Will set self.word2index & self.index2vector

    def load_word2vec(self, word2vec_file):
        #load word2vec from file
        #Two data structure: word2index, index2vector
        # wordvector = list(open("../data/vector1.txt", "r").readlines())
        wordvector = list(open(word2vec_file, "r").readlines())
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

    def load_relations(self, relation2id_file):
        #load relation from file
        # relation_data = list(open("../data/RE/relation2id.txt").readlines())
        relation_data = list(open(relation2id_file).readlines())
        relation_data = [s.split() for s in relation_data]
        for relation in relation_data:
            r = Relation(relation[0], int(relation[1]))
            self.relations[relation[0]] = r
        for r in self.relations:
            self.relations[r].generate_vector(len(self.relations)) # Generates one-hot representation
        print("RelationTotal: "+str(len(self.relations)))

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class old_Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
        

class Sentence(object):
    def __init__(self, e1, e2, r, w):
        self.entity1 = e1
        self.entity2 = e2
        self.relation = r
        self.words = w        
    def __repr__(self):
        return('Sentence(entity1={0}| entity2={1}| relation={2}| words={3})'.format(
            self.entity1, self.entity2, self.relation, ' '.join(self.words)))
class Relation(object):
    def __init__(self, name_, id_):
        self.id = id_
        self.name = name_
        self.number = 0

    def generate_vector(self, relationTotal):
        v = np.zeros(relationTotal)
        v[self.id] = 1
        self.vector = v

    def add_one(self):
        self.number += 1

    def __repr__(self):
        return('Relation(id={0}| name={1}| number={2})'.format(self.id, self.name, self.number))
    
