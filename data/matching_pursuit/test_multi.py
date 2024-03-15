import matching_pursuit as mp
from datetime import datetime
import soundfile as sf
from scipy.signal import get_window
import numpy as np
import os
import torch
import librosa

dictionary_size = 10000
#can be directory or file
file_name = "test_short.wav"
output_name = "test_short"
file_name = "taylor_vocals/vocals_betty.wav"
output_name = "betty"
# file_name = "Wiley_10.wav"
# output_name = "wiley"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
num_atoms= 500

window_type = "hann"
hann1 = get_window(window_type, chunk_size)
hann2 = get_window(window_type, chunk_size//2)
hann3 = get_window(window_type, chunk_size//4)
hann4 = get_window(window_type, chunk_size*2)
params_list = [
    (hann1, 4, 16),
    (hann2, 4, 16), 
    (hann3, 4, 16),
    (hann4, 4, 16),
]
multi_gabor_dictionary = mp.generate_multi_gabor_dictionary(params_list)
print(multi_gabor_dictionary.shape)
# dictionary = mp.get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
dictionary = multi_gabor_dictionary
dictionary_size = len(dictionary[0])
cache_name = mp.get_run_name(output_name, chunk_size*2, dictionary_size, num_atoms)

data = mp.preprocess_data_embedding(file_name, 
                                sr = sr, num_atoms=num_atoms,
                                chunk_size=chunk_size*2, hop_length=hop_length*2, 
                                dictionary=dictionary, name=output_name, trim = 0.1)

# create the train and test splits
print("data", data.shape)
audio = mp.reconstruct_from_embedding_chunks(data, dictionary, chunk_size*2, hop_length*2)
audio = torch.real(audio).cpu().numpy()
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
name = f"{output_name}_{timestampStr}.wav"
path = os.path.join("recon_audio", name)
# # WRITE AUDIO
sf.write(path, audio, 44100)
