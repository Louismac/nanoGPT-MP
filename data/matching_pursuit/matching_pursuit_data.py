# Recurrent Neural Network
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa
from os.path import isdir
from os import listdir
from matching_pursuit import process_in_chunks
np.set_printoptions(suppress=True)

def read_audio(path, sr=44100):
    #search folder
    x = [0]
    if not isdir(path):
        x, sr = librosa.load(path, sr=sr) 
    else:
        files = listdir(path)
        x = np.array([0])
        for file in files:
            if not ".DS" in file:
                audio, sr, = librosa.load(path + file, sr = sr)
                x = np.concatenate((x, audio))
    x = np.array(x, dtype=np.float32)
    return x

def get_sequences(chunks_info, sequence_length):
    #chunks info is num_frames x (num_atoms * 2)
    start = 0
    end = len(chunks_info) - sequence_length - 1 
    step = 1
    x_frames = []
    y_frames = []

    #Split into sequences
    for i in range(start, end, step):
        x = chunks_info[i:i + sequence_length]
        y = chunks_info[i + sequence_length]
        x_frames.append(torch.tensor(x))
        y_frames.append(torch.tensor(y))

    #Swap so items x features x sequence
    x_frames = torch.stack(x_frames).transpose(1,2)
    y_frames = torch.stack(y_frames)

    return x_frames, y_frames

def preprocess_data_embedding(path, chunk_size=2048, hop_length=1024, sr=44100, 
                       num_atoms=100, dictionary=None):
    x = read_audio(path, sr)
    #Get chunks (this takes time!)
    _,chunks_info = process_in_chunks(x, 
                                    dictionary, 
                                    hop_length=hop_length,
                                    chunk_size=chunk_size, 
                                    iterations=num_atoms)
    return chunks_info

class MatchingPursuitDataset(Dataset):
    def __init__(self, x_frames, y_frames):
        self.x_frames = x_frames.float()
        self.y_frames = y_frames.float()

    def __len__(self):
        return self.x_frames.shape[0]  # Number of frames

    def __getitem__(self, idx):
        return self.x_frames[idx], self.y_frames[idx]
    
class MultiClassCoeffiecentLoss(nn.Module):
    def __init__(self, num_atoms, num_categories):
        super(MultiClassCoeffiecentLoss, self).__init__()
        self.num_atoms = num_atoms
        self.num_categories = num_categories
        self.softmax_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_indices, predicted_coefficients, targets):
        true_indices = targets[:,:self.num_categories]
        true_coefficients = targets[:,self.num_categories:]
        #compares probs against binary (num_categories vs num_categories)
        indices_loss = self.softmax_loss(predicted_indices, true_indices)
        coefficients_loss = self.mse_loss(predicted_coefficients, true_coefficients)
        total_loss = indices_loss + coefficients_loss
        return total_loss

class RNNEmbeddingModel(nn.Module):
    def __init__(self, num_categories, num_atoms,embedding_dim, hidden_size, num_layers):
        super(RNNEmbeddingModel, self).__init__()
        self.num_categories = num_categories
        self.num_atoms = num_atoms
        self.batch_norm = nn.BatchNorm1d(embedding_dim+1)
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.lstm = nn.LSTM((embedding_dim+1)*num_atoms, hidden_size, num_layers, batch_first=True)
        #this is num_categories for the indices and num_atoms for the coeffients
        self.linear1 = nn.Linear(hidden_size, num_atoms) 
        self.linear2 = nn.Linear(hidden_size, num_categories)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        indices = x[:, :self.num_atoms].long()
        coefficients = x[:, self.num_atoms:].float()
        embedded_indices = self.embedding(indices)
        concatenated_inputs = torch.cat((embedded_indices, coefficients.unsqueeze(-1)), dim=-1)
        reshaped_inputs = concatenated_inputs.view(-1, concatenated_inputs.size(-1))
        x = self.batch_norm(reshaped_inputs)
        x = x.view(concatenated_inputs.size())
        x = x.view(x.size(0), x.size(2), -1)
        # LSTM expects [batch, seq_len, features]
        x, _ = self.lstm(x)
        output_coefficients = self.linear1(x[:, -1, :])
        output_indices = self.linear2(x[:, -1, :])
        # output_indices = self.sig(output_indices)
        return output_indices, output_coefficients
    
