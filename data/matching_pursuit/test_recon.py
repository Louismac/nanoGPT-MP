from matching_pursuit import reconstruct_from_embedding_chunks, get_dictionary, get_run_name, preprocess_data_embedding
from datetime import datetime
import soundfile as sf

dictionary_size = 10000
#can be directory or file
file_name = "test_short.wav"
output_name = "test_short"
# file_name = "Wiley_10.wav"
# output_name = "wiley"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
num_atoms= 600
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=10000, sr=sr, dictionary_size=dictionary_size)
dictionary_size = len(dictionary[0])
cache_name = get_run_name(output_name, chunk_size, dictionary_size, num_atoms)

data = preprocess_data_embedding(file_name, 
                                sr = sr, num_atoms=num_atoms,
                                chunk_size=chunk_size, hop_length=hop_length, 
                                dictionary=dictionary, name=output_name)

# create the train and test splits
print("data", data.shape)
audio = reconstruct_from_embedding_chunks(data, dictionary, chunk_size, hop_length).cpu().numpy()
# audio = reconstruct_from_normalised_chunks(y_frames, dictionary, chunk_size, hop_length, cmax, cmin)

print(len(audio))
timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
name = f"{output_name}_{timestampStr}.wav"
path = os.path.join("recon_audio", name)
# # WRITE AUDIO
sf.write(path, audio, 44100)
