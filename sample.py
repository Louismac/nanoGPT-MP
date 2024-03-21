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
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

num_atoms = 100
file_name = "taylor_vocals"
name = "taylor_vocals"
chunk_size = 2048
hop_length = chunk_size//2
sr = 44100
dictionary_size = 10000
dataset = 'matching_pursuit'
batch_size = 64
block_size = 256 
logit_loss = False
num_features = 3
conv_input = False

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("config", config)
cache_path = get_run_name(config["name"], config["chunk_size"], config["dictionary_size"], config["num_atoms"])
cache_path = os.path.join("data", dataset, cache_path)

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt_conv_input_mse_loss_2.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("resume from checkpoint")
    gptconf = GPTConfig(**checkpoint['model_args'])
    gptconf.logit_loss = logit_loss 
    gptconf.num_features = num_features 
    gptconf.conv_input = conv_input 
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

x,y = get_batch("train")
print("x[0]", x[0].shape)
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            pick = np.random.randint(len(x))
            y = model.generate(x[pick].unsqueeze(0), max_new_tokens, temperature=temperature, top_k=top_k)
            y = y.squeeze(0)
            #trim off seed
            y = y[config["block_size"]:]
            print(y.shape)
            if not config["conv_input"]:
                #embed input 
                # full indexes + mags + phases end to end 
                interleaved = torch.empty_like(y)
                indices = torch.arange(y.shape[1]) % 3
                for i in range(3):
                    n = config["num_atoms"]
                    interleaved[:,indices==i] = y[:,i*n:(i+1)*n]
                y = interleaved
            else:
                #conv input (2d)
                # normalised indexes in a 2D tensor
                interleaved = torch.zeros((y.shape[0],y.shape[1]*y.shape[2]), device=y.device)
                indices = torch.arange(interleaved.shape[1]) % 3
                for i in range(3):
                    n = config["num_atoms"]
                    interleaved[:,indices==i] = y[:,:,i]
                y = interleaved
                #rediscretize to index (from 0-1)
                y[:,::3] = torch.floor(y[:,::3]*config["dictionary_size"])
            #unnormalise phase
            y[:,2::3] = (y[:,2::3]*(2*torch.pi))-torch.pi
            #we get indexes and mags and phases, libltfat wants sparse complex coefficients 
            np.savetxt("output" + str(k) + ".csv", y.cpu().numpy(), delimiter=',',fmt='%.5f')
            
