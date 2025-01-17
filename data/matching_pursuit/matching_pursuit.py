import numpy as np
import numpy as np
from os.path import exists, join, isdir
from os import mkdir, listdir
from scipy.signal import get_window
import torch
import sys
import librosa
import scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "mps")

def read_audio(path, sr=44100):
    print("reading audio")
    #search folder
    x = [0]
    if not isdir(path):
        x, sr = librosa.load(path, sr=sr) 
    else:
        files = listdir(path)
        x = np.array([0])
        for file in files:
            if not ".DS" in file:
                audio, sr, = librosa.load(join(path, file), sr = sr)
                x = np.concatenate((x, audio))
    print("loaded audio", len(x)/sr)
    return torch.tensor(x, device=device).float()

def preprocess_data_embedding(path, chunk_size=2048, hop_length=1024, sr=44100, 
                       num_atoms=100, dictionary=None, name=""):
    x = read_audio(path, sr)
    #Get chunks (this takes time!)
    print("process_in_chunks")
    data = process_in_chunks(x, 
                            dictionary, 
                            hop_length=hop_length,
                            chunk_size=chunk_size, 
                            iterations=num_atoms, name = name)
    print("data", data.shape)
    return data

def get_run_name(name, chunk_size, dictionary_size, num_atoms):
    dir = name + "_" + str(chunk_size) + "_" + str(dictionary_size) + "_" + str(num_atoms)
    return dir

def process_in_chunks(signal, dictionary, chunk_size=2048, hop_length = 1024,
                       window_type='hann', iterations = 100, name=""):
    cached_path = get_run_name(name, chunk_size, len(dictionary[0]), iterations)
    cached_path = join(cached_path,"cached_chunks.pt")
    if exists(cached_path):
        chunks_info = torch.load(cached_path)
        chunks_info = chunks_info.to(device).float()
        print("loaded from cache", chunks_info.shape)
        return chunks_info

    window = torch.tensor(get_window(window_type, chunk_size), device=device, dtype=torch.float32)
    
    stop = len(signal) - chunk_size + 1
    num_chunks = (stop//hop_length)+1
    chunks_info = torch.zeros(num_chunks, iterations*2, device=device)
    for i, start in enumerate(range(0, stop, hop_length)):
        end = start + chunk_size
        chunk = signal[start:end]
        windowed_chunk = chunk * window
        chunks_info[i] = matching_pursuit(windowed_chunk, dictionary, iterations) 
        sys.stdout.write("\r{} frames out of {} ({:.2f}%)".format(start//chunk_size, stop//chunk_size, start/stop*100))
        sys.stdout.flush()
    torch.save(chunks_info, cached_path)
    return torch.tensor(chunks_info, device=device)

def matching_pursuit(signal, dictionary, iterations=20):

    residual = signal.clone().float()
    output = torch.zeros(iterations*2, device=device)

    for i in range(iterations):
        correlations = torch.matmul(dictionary.T, residual.view(-1, 1))  
        best_atom_index = torch.argmax(correlations.abs())  
        best_coefficient = correlations[best_atom_index] 
        if not torch.isinf(best_coefficient):
            residual = residual - (best_coefficient * dictionary[:, best_atom_index])  
            output[i] = best_atom_index
            output[i+iterations] = best_coefficient
        else:
            break
    return output

def get_embedding_atoms(chunk_info, num_atoms):
    return chunk_info[:num_atoms], chunk_info[num_atoms:]

def reconstruct_from_embedding_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024):
    num_atoms = len(chunks_info[0])//2
    return reconstruct_from_chunks(chunks_info, dictionary, chunk_size, hop_length, 
                                   get_embedding_atoms,  num_atoms)

def reconstruct_signal(atom_indices, coefficients, dictionary):
    reconstructed_signal = torch.zeros(dictionary.shape[0], device=device)
    for index, coeff in zip(atom_indices, coefficients):
        reconstructed_signal += coeff * dictionary[:, int(index)]
    return reconstructed_signal

def reconstruct_from_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024, unpack_func=lambda x: x, *args):
    
    signal_length = (len(chunks_info) * (hop_length))+chunk_size
    reconstructed_signal = torch.zeros(signal_length, device=device)
    weight_sum = torch.zeros(signal_length, device=device)  
    dictionary_size = len(dictionary[0])
    start = 0
    end = chunk_size
    
    for chunk_info in chunks_info:
        
        atom_indices, coefficients = unpack_func(chunk_info, *args)
        atom_indices = torch.clamp(atom_indices, min=0, max=dictionary_size-1)
        chunk_reconstruction = reconstruct_signal(atom_indices, coefficients, dictionary) 
        reconstructed_signal[start:end] += chunk_reconstruction
        weight_sum[start:end] += 1  
        start += hop_length
        end += hop_length
        sys.stdout.write("\r{} frames out of {} ({:.2f}%)".format(start//chunk_size, signal_length//chunk_size, start/signal_length*100))
        sys.stdout.flush()

    overlap_areas = weight_sum > 1  
    reconstructed_signal[overlap_areas] /= weight_sum[overlap_areas]
    return reconstructed_signal

def generate_gabor_atom(length, freq, sigma, sr, phase=0):
    # Adjust time vector to be in seconds 
    t = np.linspace(-1, 1, length) * (length / sr)
    gaussian = np.exp(-0.5 * (t / sigma) ** 2)
    sinusoid = np.cos(2 * np.pi * freq * t + phase)
    return gaussian * sinusoid

def create_gabor_dictionary(length, freqs, sigmas, sr, phases=[0]):
    atoms = []
    for freq in freqs:
        for sigma in sigmas:
            for phase in phases:
                atom = generate_gabor_atom(length, freq, sigma, sr, phase)
                atoms.append(atom)
    return np.array(atoms).T  # Each column is an atom

def get_dictionary(chunk_size=2048, dictionary_size=10000, 
                   min_freq=30, max_freq=20000, sr=44100,
                   sigmas=[0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5]):
    freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), dictionary_size // len(sigmas))
    dictionary = create_gabor_dictionary(chunk_size, freqs, sigmas, sr)
    gen_size = dictionary.shape[1]
    pad_size = dictionary_size-gen_size
    padding = np.random.random((chunk_size,pad_size))
    print("padding", padding.shape)
    dictionary = np.hstack((dictionary, padding))
    dictionary = dictionary.astype(np.float32)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    print("dictionary", dictionary.shape)
    return torch.tensor(dictionary, device=device)
