from matching_pursuit import reconstruct_from_embedding_chunks, get_dictionary, get_run_name, preprocess_data_embedding
from datetime import datetime
import soundfile as sf
import torch
import numpy as np
import os
config = {}
config["num_atoms"] = 20
config["dictionary_size"] = 1024
config["chunk_size"] = 2048
config["name"] = "taylor_vocals"
device = 'cuda'
device_type = "cuda"
config["num_features"] = 3
config["logit_loss"] = False
config["conv_input"] = True
block_size = 128
batch_size = 64*10
dataset = 'matching_pursuit'

cache_path = get_run_name(config["name"], config["chunk_size"], config["dictionary_size"], config["num_atoms"])
# cache_path = os.path.join("data", dataset, cache_path)

def get_sparse(y):
    #sparse
    #deinterleave
    y = torch.cat((y[:,:,::3], y[:,:,1::3],y[:,:,2::3]), dim=2)
    indices = y[:,:,:config["num_atoms"]].long()
    coeff = y[:,:,config["num_atoms"]:]
    b, s, a = indices.shape
    sparse = torch.zeros(b, s, config["dictionary_size"], dtype=torch.float32, device=device)
    for i in range(a):
        #DIM, INDICES, VALUES
        sparse.scatter_add_(2, indices[:, :, i:i+1], torch.ones_like(indices[:, :, i:i+1], dtype=torch.float32))
    # print("indices", indices.cpu().numpy())
    # print("sparse", sparse.cpu().numpy())
    sparse.clamp_(max=1)
    #ADD IN COEFF
    sparse = torch.cat([sparse.float(), coeff], dim=-1)
    return sparse

# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(cache_path, 'train.bin'), dtype=np.float32, mode='r')
    else:
        data = np.memmap(os.path.join(cache_path, 'val.bin'), dtype=np.float32, mode='r')
    saved_atoms = 100
    num_features = (saved_atoms*3)
    # print("num_frames", len(data)//num_features)
    data = data.reshape(len(data)//num_features, saved_atoms, 3)
    data = data[:,:config["num_atoms"],:]
    # print("data", data.shape)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.float32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.float32)) for i in ix]) 
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    #batch, seq, num_atoms, num_feaures
    x = x[:,:,:,:config["num_features"]]
    #normalise into the range 0-1 (from -pi - pi)
    x[:,:,:,2::3] += torch.pi
    x[:,:,:,2::3] /= (2*torch.pi)
    if config["conv_input"]:
        #normalise into the range 0-1 (0 - dictionary_size)
        x[:,:,:,::3] /= (config["dictionary_size"])
    else :
        #flatten [100x3 into 300]
        x = x.view(x.size(0), x.size(1), -1)
        #deinterleave
        x = torch.cat((x[:,:,::3], x[:,:,1::3],x[:,:,2::3]), dim=2)

    y = y[:,:,:,:config["num_features"]]
    #normalise into the range 0-1 (from -pi - pi)
    y[:,:,:,2::3] += torch.pi
    y[:,:,:,2::3] /= (2*torch.pi)

    if config["logit_loss"]:
        #flatten [100x3 into 300]
        y = y.view(y.size(0), y.size(1), -1)
        #get sparse + end to end mags and phases
        y = get_sparse(y)
    else:
        #normalise into the range 0-1 (0 - dictionary_size)
        y[:,:,:,0] /= (config["dictionary_size"])

    return x, y

X,y = get_batch("train")

labels = X[:,:,:,::3].view(-1)
labels = torch.floor(labels*config["dictionary_size"])
# Get unique labels and their counts
unique_labels, counts = torch.unique(labels, return_counts=True)
# Sort labels and reindex counts
sorted_indices = torch.argsort(counts)
sorted_labels = unique_labels[sorted_indices]
sorted_counts = counts[sorted_indices]

# Display sorted class distribution
for label, count in zip(sorted_labels.tolist(), sorted_counts.tolist()):
    print(f'Class {label}: {count} occurrences')


