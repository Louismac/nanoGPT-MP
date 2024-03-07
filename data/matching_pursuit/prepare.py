from matching_pursuit import get_dictionary, get_run_name
from matching_pursuit_data import preprocess_data_embedding
import numpy as np
import os

num_atoms = 100
dictionary_size = 10000
file_name = "Wiley_10.wav"
output_name = "wiley"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
dictionary_size = len(dictionary[0])
cache_name = get_run_name(output_name, chunk_size, dictionary_size, num_atoms)
data, sparse = preprocess_data_embedding(file_name, 
                                sr = sr, num_atoms=num_atoms,
                                chunk_size=chunk_size, hop_length=hop_length, 
                                dictionary=dictionary, name=output_name)

# create the train and test splits
print("data", data.shape)
print("sparse", sparse.shape, sparse)
n = len(data)
train_data = np.vstack([data[:int(n*0.9)] for i in range(100)])
val_data = np.vstack([data[int(n*0.9):] for i in range(100)])
print("train_data x", train_data.shape, train_data)
train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)
train_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'train_x.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'val_x.bin'))

train_data = np.vstack([sparse[:int(n*0.9)] for i in range(100)])
val_data = np.vstack([sparse[int(n*0.9):] for i in range(100)])
train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)
train_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'train_y.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'val_y.bin'))
print("train_data y", train_data.shape)
