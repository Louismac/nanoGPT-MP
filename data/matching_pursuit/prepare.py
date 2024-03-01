from matching_pursuit import get_dictionary
from matching_pursuit_data import preprocess_data_embedding
import numpy as np
import os

num_atoms = 20
dictionary_size = 5000
file_name = "Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
dictionary_size = len(dictionary[0])
data = preprocess_data_embedding(file_name, 
                                sr = sr, num_atoms=num_atoms,
                                chunk_size=chunk_size, hop_length=hop_length, 
                                dictionary=dictionary)
# create the train and test splits
print("data", data.shape)
n = len(data)
train_data = np.vstack([data[:int(n*0.9)] for i in range(100)])
print("train_data", train_data.shape)
val_data = np.vstack([data[int(n*0.9):] for i in range(100)])

# encode both to integer
print(f"train has {len(train_data):,} tokens")
print(f"val has {len(val_data):,} tokens")
print("train_data", train_data.max())
print("val_data", val_data.max())

# export to bin files (gets flattened)
train_data = np.array(train_data, dtype=np.int32)
val_data = np.array(val_data, dtype=np.int32)
print("train_data", train_data.max())
print("val_data", val_data.max())
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
