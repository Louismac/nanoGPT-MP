import numpy as np
import os
import csv
import sys
from matching_pursuit import get_run_name
np.set_printoptions(precision=5, suppress=True, threshold=sys.maxsize)
num_atoms = 100
chunk_size = 1024
dictionary_size = 512
hop_length = 512
cache_name = get_run_name("taylor_1024", chunk_size, dictionary_size, num_atoms)
# path = "/Users/lmccallum/Documents/nanoGPT-MP/taylor"
# path = "test/"
path = "/home/louis/Documents/notebooks/fastmpwithmultigabor/libltfat/examples/multigabormp/taylor/converted_wavs/1024"
files = os.listdir(path)
data = []
for p in files:
    if p.endswith(".csv"): 
        print(p)
        p = os.path.join(path, p)
        with open(p, 'r') as f:
            reader = csv.reader(f)
            n_features = 3
            def get_frame(atoms):
                frame = np.zeros((num_atoms, n_features))
                max_count = len(atoms)//n_features
                if max_count > num_atoms:
                    max_count = num_atoms
                for atom in range(max_count):
                    for f in range(n_features):
                        index = (atom*n_features)+f
                        val = atoms[index]
                        frame[atom,f] = val
                return frame
            #filter out mostly silence frames
            min_atoms = 5
            song_data = np.array([get_frame(atoms) for atoms in reader if (len(atoms)-1)//n_features>min_atoms])
            data.append(song_data)
data = np.vstack(data)           

# create the train and test splits
print("data", data.shape)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
train_data = np.array(train_data, dtype=np.float32)
val_data = np.array(val_data, dtype=np.float32)
train_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), cache_name, 'val.bin'))
