import os
import pickle

def load_pkl(file_path):
    assert os.path.exists(file_path)
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
        
    return data
    
def save_pkl(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)
        
        
def del_dir(dir_path):
    os.removedirs(dir_path)
    
def create_dir(dir_path, cover=False):
    if cover and os.path.exists(dir_path):
        os.removedirs(dir_path)
        os.makedirs(dir_path)