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
from data.matching_pursuit.matching_pursuit import get_run_name, get_dictionary
from data.matching_pursuit.matching_pursuit import reconstruct_from_embedding_chunks
from datetime import datetime
import soundfile as sf


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out_mp' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.7 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 5000 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

num_atoms = 100
file_name = "taylor_songs"
name = "taylor_songs"
chunk_size = 2048
hop_length = chunk_size//4
sr = 22050
dictionary_size = 10000
dataset = 'matching_pursuit'
batch_size = 64
block_size = 256 

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print("config", config)
cache_path = get_run_name(config["name"], config["chunk_size"], config["dictionary_size"], config["num_atoms"])
cache_path = os.path.join("data", dataset, cache_path)
# poor man's data loader
# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(cache_path, 'train.bin'), dtype=np.float32, mode='r')
        # sparse = load_npz(os.path.join(cache_path, 'train_y.bin.npz'))
    else:
        data = np.memmap(os.path.join(cache_path, 'val.bin'), dtype=np.float32, mode='r')
        # sparse = load_npz(os.path.join(cache_path, 'val_y.bin.npz'))
    num_features = (config["num_atoms"]*2)
    data = data.reshape(len(data)//num_features, num_features)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.float32)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.float32)) for i in ix]) 
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    #y = torch.stack([torch.from_numpy((sparse[i+1:i+1+block_size]).toarray().astype(np.float32)) for i in ix]) 
    
    return x
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("resume from checkpoint")
    gptconf = GPTConfig(**checkpoint['model_args'])
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

dictionary = get_dictionary(chunk_size=config["chunk_size"], max_freq=10000, 
                                        sr=config["sr"], dictionary_size=config["dictionary_size"])
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            x = get_batch("train")
            seed = x[0].unsqueeze(0)
            print("Seed", seed.shape)
            y = model.generate(seed, max_new_tokens, temperature=temperature, top_k=top_k)
            audio = reconstruct_from_embedding_chunks(y.squeeze(0), 
                                                      dictionary=dictionary, 
                                                      chunk_size=config["chunk_size"], 
                                                      hop_length=config["hop_length"]).cpu().numpy()
            print(len(audio))
            timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
            # # WRITE AUDIO
            name = config["name"]
            name = f"{name}_{timestampStr}.wav"
            if not os.path.exists(os.path.join(cache_path, "audio")):
                os.mkdir(os.path.join(cache_path, "audio"))
            sf.write(os.path.join(cache_path, "audio", name), audio, config["sr"])
