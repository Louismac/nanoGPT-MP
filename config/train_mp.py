out_dir = 'out-mp'
eval_interval = 2000 # keep frequent because we'll overfit
eval_iters = 2
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'mp'
wandb_run_name = 'mini-gpt'

#mp stuff 
logit_loss = True
conv_input = False
num_atoms = 50
num_features = 3
name = "taylor_1024"
chunk_size = 1024
hop_length = 512
sr = 44100
dictionary_size = 512
phase_buckets = 4
mag_buckets = 50

dataset = 'matching_pursuit'
gradient_accumulation_steps = 1
curric_steps = 1
batch_size = 32
block_size = 128 
#batch = 31*32*
n_layer = 6
n_head = 6
n_embd = n_head*30
dropout = 0.3

#keep this high, make sure to have a short warmup set
learning_rate = 1e-2

max_iters = 10000
lr_decay_iters = max_iters
min_lr = 1e-2 # learning_rate / 10 usually
beta2 = 0.95
warmup_iters = 100