from matching_pursuit import reconstruct_from_embedding_chunks, get_dictionary, get_run_name, preprocess_data_embedding
from datetime import datetime
import soundfile as sf
import torch
import numpy as np
import sys
np.set_printoptions(precision=5, suppress=True, threshold=sys.maxsize) 
import os
config = {}
config["num_atoms"] = 20
config["dictionary_size"] = 1024
config["chunk_size"] = 2048
config["name"] = "cello"
config["curric_steps"] = 1
device = 'cuda'
device_type = "cuda"
config["num_features"] = 3
config["logit_loss"] = True
config["conv_input"] = True
config["mag_buckets"] = 50
block_size = 128
batch_size = 64*10
dataset = 'matching_pursuit'

cache_path = get_run_name(config["name"], config["chunk_size"], config["dictionary_size"], config["num_atoms"])
# cache_path = os.path.join("data", dataset, cache_path)
edges = torch.zeros(0)
def encode_indices(indices, mags):
    return indices * config["mag_buckets"] + mags

def decode_indices(encoded_indices):
    indices  = encoded_indices // config["mag_buckets"]
    mags = encoded_indices % config["mag_buckets"]
    return indices, mags

def get_edges(mags):
    sorted_tensor, _ = torch.sort(mags.reshape(-1))
    total_elements = sorted_tensor.numel()
    quantile_edges = torch.linspace(0, total_elements - 1, steps=config["mag_buckets"] + 1)[1:-1].long()
    return sorted_tensor[quantile_edges]

def bucket_mags(mags):
    global edges
    if edges.numel()==0:
        edges = get_edges(mags).unsqueeze(0).unsqueeze(0)
    bucket_indices = (mags.unsqueeze(-1) >= edges).long().sum(dim=-1)
    return bucket_indices

def get_sparse(y):
    #sparse
    indices = y[:,:,:config["num_atoms"]].long()
    mags = y[:,:,config["num_atoms"]:]
    
    mags = bucket_mags(mags)
    # print(mags.cpu().numpy())
    b, s, a = indices.shape
    combined = encode_indices(indices, mags)
    sparse = torch.zeros(b, s, config["dictionary_size"]*config["mag_buckets"], dtype=torch.float32, device=device)
    for i in range(a):
        #DIM, INDICES, VALUES
        sparse.scatter_add_(2, combined[:, :, i:i+1], torch.ones_like(combined[:, :, i:i+1], dtype=torch.float32))
    sparse.clamp_(max=1)
    print("sparse", sparse.shape)
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
    #drop phase
    data = data[:,:config["num_atoms"],:]

    # print("data", data.shape)
    ix = torch.randint(len(data) - block_size - config["curric_steps"], (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.float32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+block_size:i+block_size+config["curric_steps"]]).astype(np.float32)) for i in ix]) 
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    #drop phase
    x = x[:,:,:,:2]
    if config["conv_input"]:
        #normalise into the range 0-1 (0 - dictionary_size)
        x[:,:,:,0] /= (config["dictionary_size"])
    else :
        #flatten [100x2 into 200]
        x = x.reshape(x.size(0), x.size(1), -1)
        #deinterleave
        indices = x[:,:,::2]
        mags = x[:,:,1::2]
        mags = bucket_mags(mags)
        x = encode_indices(indices, mags)

    y = y[:,:,:,:2]

    if config["logit_loss"]:
        #flatten [100x2 into 200]
        y = y.reshape(y.size(0), y.size(1), -1)
        #deinterleave
        y = torch.cat((y[:,:,::2], y[:,:,1::2]), dim=2)
        y = get_sparse(y)
    else:
        #normalise into the range 0-1 (0 - dictionary_size)
        y[:,:,:,0] /= (config["dictionary_size"])
    return x, y

# for i in range(10):
#     X,y = get_batch("train")
#     print(edges.cpu().numpy())
#     edges = torch.zeros(0)


# # mags = X[:,:,:,1::3]
# target_loop_loss = torch.mean(torch.abs(X[:,-1,:,1]-X[:,-2,:,1]), dim = (0))
# X,y = get_batch("train")
# loop_loss = torch.mean(torch.abs(X[:,-1,:,1]-X[:,-2,:,1]), dim = (0))
# print(torch.mean(torch.abs(target_loop_loss - loop_loss)))

# labels = X[:,:,:,::3].view(-1)
# labels = torch.floor(labels*config["dictionary_size"])
# # Get unique labels and their counts
# unique_labels, counts = torch.unique(labels, return_counts=True)
# # Sort labels and reindex counts
# sorted_indices = torch.argsort(counts)
# sorted_labels = unique_labels[sorted_indices]
# sorted_counts = counts[sorted_indices]

# # Display sorted class distribution
# for label, count in zip(sorted_labels.tolist(), sorted_counts.tolist()):
#     print(f'Class {label}: {count} occurrences')


