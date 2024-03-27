out_dir = 'out-mp'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'mp'
wandb_run_name = 'mini-gpt'

#mp stuff 
logit_loss = True
conv_input = True
num_atoms = 80
num_features = 3
name = "cello"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary_size = chunk_size//2

dataset = 'matching_pursuit'
gradient_accumulation_steps = 1
curric_steps = 31
batch_size = 64
block_size = 256 
#batch = 31*32*512 = ~0.5M?
n_layer = 12
n_head = 12
n_embd = 300
dropout = 0.1

learning_rate = 1e-4 
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95
