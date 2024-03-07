import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
from os.path import exists, join
from os import mkdir
from scipy.signal import get_window
import torch
import sys

def get_run_name(name, chunk_size, dictionary_size, num_atoms):
    dir = name + "_" + str(chunk_size) + "_" + str(dictionary_size) + "_" + str(num_atoms)
    if not exists(dir):
        mkdir(dir)
    return dir

def process_in_chunks(signal, dictionary, chunk_size=2048, hop_length = 1024,
                       window_type='hann', iterations = 100, name=""):
    cached_path = get_run_name(name, chunk_size, len(dictionary[0]), iterations)
    cached_path = join(cached_path,"cached_chunks.npy")
    if exists(cached_path):
        chunks_info = np.load(cached_path)
        chunks_info = torch.tensor(chunks_info).float()
        print("loaded from cache", chunks_info.shape)
        return chunks_info

    window = torch.tensor(get_window(window_type, chunk_size)).float()
    
    chunks_info = []
    stop = len(signal) - chunk_size + 1
    print(0, stop, hop_length)
    for start in range(0, stop, hop_length):
        end = start + chunk_size
        chunk = signal[start:end]
        windowed_chunk = chunk * window
        _, atom_indices, coefficients = matching_pursuit(windowed_chunk, dictionary, iterations) 
        chunk_info = atom_indices + coefficients
        chunks_info.append(chunk_info)
        sys.stdout.write("\r{} out of {}...".format(start, stop))
        sys.stdout.flush()
    np.save(cached_path, chunks_info)
    return torch.tensor(chunks_info)

def matching_pursuit(signal, dictionary, iterations=20):

    residual = signal.clone().float()
    reconstruction = torch.zeros_like(signal)
    atom_indices = []
    coefficients = []

    for _ in range(iterations):
        correlations = torch.matmul(dictionary.T, residual.view(-1, 1))  
        best_atom_index = torch.argmax(np.abs(correlations))  
        best_coefficient = correlations[best_atom_index] 
        if not np.isinf(best_coefficient):
            reconstruction += best_coefficient * dictionary[:, best_atom_index] 
            residual = residual - (best_coefficient * dictionary[:, best_atom_index])  
            atom_indices.append(best_atom_index.item())
            coefficients.append(best_coefficient.item())
        else:
            break
    return reconstruction.detach().numpy(), atom_indices, coefficients

def get_unnormalised_atoms(chunk_info, num_atoms, dictionary_size, cmax, cmin):
    atom_indices = chunk_info[:num_atoms].detach().numpy()
    atom_indices = np.array(np.ceil(atom_indices*dictionary_size), dtype=np.int32)-1
    
    coefficients = chunk_info[num_atoms:].detach().numpy()
    coefficients = (coefficients * (cmax - cmin)) + cmin
    return np.array(atom_indices), np.array(coefficients)

def get_dense_atoms(chunk_info):
    nonzero_indexes = np.nonzero(chunk_info)
    nonzero_indexes = nonzero_indexes.detach().numpy().ravel()
    nonzero_values = chunk_info[nonzero_indexes]
    return nonzero_indexes, nonzero_values.detach().numpy()

def get_embedding_atoms(chunk_info, num_atoms):
    chunk_info = chunk_info.detach().numpy()
    return chunk_info[:num_atoms], chunk_info[num_atoms:]

def reconstruct_from_sparse_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024):
    return reconstruct_from_chunks(chunks_info, dictionary, chunk_size, hop_length, get_dense_atoms)

def reconstruct_from_embedding_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024):
    num_atoms = len(chunks_info[0])//2
    return reconstruct_from_chunks(chunks_info, dictionary, chunk_size, hop_length, 
                                   get_embedding_atoms,  num_atoms)

def reconstruct_from_normalised_chunks(chunks_info, dictionary, chunk_size=2048, hop_length=1024, cmax=1, cmin=0):
    num_atoms = len(chunks_info[0])//2
    dictionary_size = len(dictionary[0])
    return reconstruct_from_chunks(chunks_info, dictionary, chunk_size, hop_length, 
                                   get_unnormalised_atoms, num_atoms, dictionary_size, cmax, cmin)

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
    return torch.tensor(dictionary).float()
