from matching_pursuit_data import MatchingPursuitDataset
from matching_pursuit_data import preprocess_data_embedding
from matching_pursuit import reconstruct_from_embedding_chunks, get_dictionary
from datetime import datetime
import soundfile as sf
import cupy as cp

sequence_length = 40
num_atoms = 20
dictionary_size = 10000
file_name = "Wiley_10.wav"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
x_frames, y_frames = preprocess_data_embedding(file_name, 
                                        sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary)

# x_frames, y_frames, cmax, cmin = preprocess_data_normalised(file_name,sequence_length=sequence_length, 
#                                         sr = sr, num_atoms=num_atoms,
#                                         chunk_size=chunk_size, hop_length=hop_length, 
#                                         dictionary=dictionary)

dataset = MatchingPursuitDataset(x_frames, y_frames)

audio = reconstruct_from_embedding_chunks(y_frames, dictionary, chunk_size, hop_length)
# audio = reconstruct_from_normalised_chunks(y_frames, dictionary, chunk_size, hop_length, cmax, cmin)

print(y_frames.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
output_name = "wiley"
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)
