out_dir = 'out-mp'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'mp'
wandb_run_name = 'mini-gpt'

#mp stuff 
num_atoms = 100
file_name = "taylor_songs"
name = "taylor_songs"
chunk_size = 2048
hop_length = chunk_size//4
sr = 22050
dictionary_size = 10000


dataset = 'matching_pursuit'
gradient_accumulation_steps = 32
batch_size = 2
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 12
n_embd = 384*2
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = 15000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
