import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
from os.path import exists, join
from os import mkdir
from scipy.signal import get_window
import torch
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_run_name(name, chunk_size, dictionary_size, num_atoms):
    dir = name + "_" + str(chunk_size) + "_" + str(dictionary_size) + "_" + str(num_atoms)
    if not exists(dir):
        mkdir(dir)
    return dir

def process_in_chunks(signal, dictionary, chunk_size=2048, hop_length = 1024,
                       window_type='hann', iterations = 100, name=""):
    cached_path = get_run_name(name, chunk_size, len(dictionary[0]), iterations)
    cached_path = join(cached_path,"cached_chunks.pt")
    if exists(cached_path):
        chunks_info = torch.load(cached_path)
        chunks_info = torch.tensor(chunks_info, device=device).float()
        print("loaded from cache", chunks_info.shape)
        return chunks_info

    window = torch.tensor(get_window(window_type, chunk_size), device=device).float()
    
    stop = len(signal) - chunk_size + 1
    chunks_info = torch.zeros((stop//hop_length)+1, iterations*2, device=device)
    print(0, stop, hop_length)
    for i, start in enumerate(range(0, stop, hop_length)):
        end = start + chunk_size
        chunk = signal[start:end]
        windowed_chunk = chunk * window
        atom_indices, coefficients = matching_pursuit(windowed_chunk, dictionary, iterations) 
        chunks_info[i] = torch.cat((atom_indices, coefficients))
        sys.stdout.write("\r{} out of {} ({}\%)".format(start, stop, np.round(start/stop, 4)*100))
        sys.stdout.flush()
    torch.save(chunks_info, cached_path)
    return torch.tensor(chunks_info, device=device)

def matching_pursuit(signal, dictionary, iterations=20):

    residual = signal.clone().float()
    atom_indices = torch.zeros(iterations, device=device)
    coefficients = torch.zeros(iterations, device=device)

    for i in range(iterations):
        correlations = torch.matmul(dictionary.T, residual.view(-1, 1))  
        best_atom_index = torch.argmax(correlations.abs())  
        best_coefficient = correlations[best_atom_index] 
        if not torch.isinf(best_coefficient):
            residual = residual - (best_coefficient * dictionary[:, best_atom_index])  
            atom_indices[i] = best_atom_index
            coefficients[i] = best_coefficient
        else:
            break
    return atom_indices, coefficients

def get_embedding_atoms(chunk_info, num_atoms):
    chunk_info = chunk_info.detach().numpy()
    return chunk_info[:num_atoms], chunk_info[num_atoms:]

def reconstruct_from_embedding_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024):
    num_atoms = len(chunks_info[0])//2
    return reconstruct_from_chunks(chunks_info, dictionary, chunk_size, hop_length, 
                                   get_embedding_atoms,  num_atoms)

def reconstruct_signal(atom_indices, coefficients, dictionary):
    reconstructed_signal = torch.zeros(dictionary.shape[0])
    for index, coeff in zip(atom_indices, coefficients):
        reconstructed_signal += coeff * dictionary[:, int(index)]
    return reconstructed_signal

def reconstruct_from_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024, unpack_func=lambda x: x, *args):
    
    signal_length = (len(chunks_info) * (hop_length))+chunk_size
    reconstructed_signal = torch.zeros(signal_length)
    weight_sum = torch.zeros(signal_length)  
    
    start = 0
    end = chunk_size
    
    for chunk_info in chunks_info:
        
        atom_indices, coefficients = unpack_func(chunk_info, *args)
        chunk_reconstruction = reconstruct_signal(atom_indices, coefficients, dictionary) 
        reconstructed_signal[start:end] += chunk_reconstruction
        weight_sum[start:end] += 1  
        start += hop_length
        end += hop_length

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
    dictionary = dictionary.astype(np.float64)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    print("dictionary", dictionary.shape)
    return torch.tensor(dictionary, device=device).float()
