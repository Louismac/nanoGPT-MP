Fork of nanoGPT to work with Matching Pursuit sequences 

Use `prepare.py` in `data/matching_pursuit` to make the dataset. First time you prepare with new data or settings this will take abit of time because we need to encode the matching pursuit. This is cached so will be quicker in future runs.

Then update `config/train_mp.py` with your dataset settings (dictionary_size, num_atoms etc...)

Then run `python3 train_mp.py config/train_mp.py`

Generation code to come