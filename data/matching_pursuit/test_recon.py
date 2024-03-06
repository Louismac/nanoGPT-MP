from matching_pursuit_data import MatchingPursuitDataset
from matching_pursuit_data import preprocess_data_embedding, get_sequences
from matching_pursuit import reconstruct_from_embedding_chunks, get_dictionary
from datetime import datetime
import soundfile as sf

sequence_length = 40
num_atoms = 20
dictionary_size = 10000
file_name = "vocals_betty.wav"
output_name = "betty"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
data = preprocess_data_embedding(file_name, sr = sr, num_atoms=num_atoms,
                                        chunk_size=chunk_size, hop_length=hop_length, 
                                        dictionary=dictionary, name = output_name)
x_frames, y_frames = get_sequences(data, sequence_length)

# x_frames, y_frames, cmax, cmin = preprocess_data_normalised(file_name,sequence_length=sequence_length, 
#                                         sr = sr, num_atoms=num_atoms,
#                                         chunk_size=chunk_size, hop_length=hop_length, 
#                                         dictionary=dictionary)

dataset = MatchingPursuitDataset(x_frames, y_frames)
print(y_frames, y_frames.shape)
audio = reconstruct_from_embedding_chunks(y_frames, dictionary, chunk_size, hop_length)
# audio = reconstruct_from_normalised_chunks(y_frames, dictionary, chunk_size, hop_length, cmax, cmin)

print(y_frames.shape, len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

# # WRITE AUDIO
sf.write(f"{output_name}_{timestampStr}.wav", audio, 44100)
