import numpy as np

class Training_sample(object):
    def __init__(self, relation_idx, words_idx, entity1p, entity2p):
        self.relation_idx = relation_idx
        self.words_idx = words_idx 
        self.entity1p = entity1p
        self.entity2p = entity2p        
    def __repr__(self):
        return('Training_sample(relation_idx={0}| words_idx={1}| entity1p={2}| entity2p={3})'.format(
            self.relation_idx, self.words_idx, self.entity1p, self.entity2p))
    
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