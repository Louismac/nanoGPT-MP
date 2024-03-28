out_dir = 'out-mp'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'mp'
wandb_run_name = 'mini-gpt'

#mp stuff 
logit_loss = True
conv_input = False
num_atoms = 20
num_features = 3
name = "cello"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary_size = chunk_size//2

dataset = 'matching_pursuit'
gradient_accumulation_steps = 1
curric_steps = 1
batch_size = 16
block_size = 128 
#batch = 31*32*
n_layer = 6
n_head = 6
n_embd = n_head*30
dropout = 0.3

#keep this high, make sure to have a short warmup set
learning_rate = 1e-2

max_iters = 2000
lr_decay_iters = max_iters
min_lr = 1e-3 # learning_rate / 10 usually
beta2 = 0.95
warmup_iters = 100