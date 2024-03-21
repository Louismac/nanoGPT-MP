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
logit_loss = False
conv_input = True
num_atoms = 20
num_features = 3
name = "taylor_vocals"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary_size = chunk_size//2

dataset = 'matching_pursuit'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 12*10
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
