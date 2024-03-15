from matching_pursuit import get_dictionary, get_run_name, preprocess_data_embedding
import torch

dictionary_size = 10000
#can be directory or file
file_name = "august.mp3"
output_name = "august"
# file_name = "Wiley_10.wav"
# output_name = "wiley"
chunk_size = 2048
hop_length = chunk_size//4
sr = 44100
num_atoms=100
dictionary = get_dictionary(chunk_size=chunk_size, max_freq=20000, sr=sr, dictionary_size=dictionary_size)
dictionary_size = len(dictionary[0])
cache_name = get_run_name(output_name, chunk_size, dictionary_size, num_atoms)

data = preprocess_data_embedding(file_name, 
                                sr = sr, num_atoms=num_atoms,
                                chunk_size=chunk_size, hop_length=hop_length, 
                                dictionary=dictionary, name=output_name)

labels = data[:,:num_atoms]
# Get unique labels and their counts
unique_labels, counts = torch.unique(labels, return_counts=True)
# Sort labels and reindex counts
sorted_indices = torch.argsort(counts)
sorted_labels = unique_labels[sorted_indices]
sorted_counts = counts[sorted_indices]

# Display sorted class distribution
for label, count in zip(sorted_labels.tolist(), sorted_counts.tolist()):
    print(f'Class {label}: {count} occurrences')


