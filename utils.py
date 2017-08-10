import numpy as np
import re
import h5py
import json
import pickle
# import cPickle as pickle

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z_]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r",", "", string)
    return string.strip().lower()

def batch_iter(data, batch_size, shuffle=True):
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
            yield list(shuffled_data[start_index:end_index])
            
def hdf5_write(arr,label,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset(label, data=arr, dtype=arr.dtype)
    
def hdf5_read(label,infile):
    with h5py.File(infile,"r") as f:
        arr = f[label][:]
    return arr

# https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch06s02.html
def json_read(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def json_write(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)
        
def pickle_read(infile):
    with open(infile, "rb") as f:
        data = pickle.load(f)
    return data
        
def pickle_write(data, outfile):
    with open(outfile, "wb") as f:
        pickle.dump(data, f)
        
