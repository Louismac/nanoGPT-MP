"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_mp import GPTConfig, GPT
import numpy as np
import sys
np.set_printoptions(precision=5, suppress=True, threshold=sys.maxsize) 
from data.matching_pursuit.matching_pursuit import get_run_name, get_dictionary
from data.matching_pursuit.matching_pursuit import reconstruct_from_embedding_chunks
from datetime import datetime
import soundfile as sf

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out_mp' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 3 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
dataset = 'matching_pursuit'
checkpoint_path = "data/matching_pursuit/taylor_1024_1024_512_50/28-Mar-2024-13-22-55/ckpt.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]
cache_path = get_run_name(config["name"], config["chunk_size"], config["dictionary_size"], config["num_atoms"])
cache_path = os.path.join("data", dataset, cache_path)  
print("resume from checkpoint", config)

mag_edges = torch.zeros(0)
phase_edges = torch.zeros(0)

def encode_indices_3d(indices, mags, phases):
    composite_index = ((indices * config["mag_buckets"] + mags) * config["phase_buckets"]) + phases
    return composite_index

def decode_indices_3d(encoded_indices):
    phases = encoded_indices % config["phase_buckets"]
    intermediate_index = encoded_indices // config["phase_buckets"]
    indices = intermediate_index // config["mag_buckets"]
    mags = intermediate_index % config["mag_buckets"]
    return indices, mags, phases

def get_edges(vals, num_buckets):
    sorted_tensor, _ = torch.sort(vals.reshape(-1))
    total_elements = sorted_tensor.numel()
    quantile_edges = torch.linspace(0, total_elements - 1, steps=num_buckets + 1)[1:-1].long()
    return sorted_tensor[quantile_edges].to(vals.device)

def set_edges(mags, phases):
    global phase_edges
    if phase_edges.numel()==0:
        phase_edges = get_edges(phases, config["phase_buckets"]).unsqueeze(0).unsqueeze(0)
    global mag_edges
    if mag_edges.numel()==0:
        mag_edges = get_edges(mags, config["mag_buckets"]).unsqueeze(0).unsqueeze(0)

def bucket(vals, edges):
    bucket_indices = (vals.unsqueeze(-1) >= edges).long().sum(dim=-1)
    return bucket_indices

def unbucket(bucket_indices, edges):
    edges = edges.view(-1).to(device)
    bucket_indices = bucket_indices.to(device)
    midpoints = torch.zeros(len(edges) + 1).to(device)
    lower_bound = edges[0] - (edges[1] - edges[0]) 
    midpoints[0] = (lower_bound + edges[0]) / 2
    for i in range(1, len(edges)):
        midpoints[i] = (edges[i-1] + edges[i]) / 2
    midpoints[-1] = edges[-1] + (edges[-1] - edges[-2]) / 2
    approx_vals = midpoints[bucket_indices]

    return approx_vals

# poor man's data loader
def get_batch(split):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

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

    # print("data", data.shape)
    ix = torch.randint(len(data) - block_size - config["curric_steps"], (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.float32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+config["curric_steps"]:i+config["curric_steps"]+block_size]).astype(np.float32)) for i in ix]) 
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    if config["conv_input"]:
        #normalise into the range 0-1 (0 - dictionary_size)
        x[:,:,:,0] /= (config["dictionary_size"])
    else :
        #split based on biggest mags
        mags = x[:,:,:,1]
        _, indices = mags.sort(descending=True)
        selected_indices = indices[:, :, :config["num_atoms"]]
        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1, 1).expand(-1, block_size, config["num_atoms"], 3)
        seq_indices = torch.arange(block_size).view(1, block_size, 1, 1).expand(batch_size, -1, config["num_atoms"], 3)
        selected_indices = selected_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        x = x[batch_indices, seq_indices, selected_indices, torch.arange(3)]
        #flatten [100x3 into 300]
        x = x.reshape(x.size(0), x.size(1), -1)
        #deinterleave
        indices = x[:,:,::3]
        mags = x[:,:,1::3]
        phases = x[:,:,2::3]
        set_edges(mags, phases)
        mags = bucket(mags, mag_edges)
        phases = bucket(phases, phase_edges)
        x = encode_indices_3d(indices, mags, phases)

    return x

# model
if init_from == 'resume':
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    gptconf.logit_loss = config["logit_loss"]
    gptconf.num_features = config["num_features"]
    gptconf.conv_input = config["conv_input"] 
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

print("model loaded")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)


x = get_batch("train")
print("x[0]")
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            pick = np.random.randint(len(x))
            input = x[pick].unsqueeze(0)
            y = model.generate(input, max_new_tokens, temperature=temperature, top_k=top_k)
            y = y.squeeze(0)
            indices, mags, phases = decode_indices_3d(y)
            mags = unbucket(mags.long(), mag_edges)
            phases = unbucket(phases.long(), phase_edges)
            print("y",y.shape)
            print("indices",indices.shape)
            # interleaved = (torch.rand((y.shape[0],y.shape[1]*3), device=y.device)*(2*torch.pi))-torch.pi
            interleaved = torch.zeros((y.shape[0],y.shape[1]*3), device=y.device)
            idx = torch.arange(interleaved.shape[1]) % 3
            interleaved[:,idx == 0] = indices
            interleaved[:,idx == 1] = mags
            interleaved[:,idx == 2] = phases
            #we get indexes and mags and phases, libltfat wants sparse complex coefficients 
            np.savetxt("output" + str(k) + ".csv", interleaved.cpu().numpy(), delimiter=',',fmt='%.6f')
            
