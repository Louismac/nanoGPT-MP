import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from matching_pursuit_data import MultiClassCoeffiecentLoss, RNNEmbeddingModel
from matching_pursuit_data import MatchingPursuitDataset
from matching_pursuit import get_dictionary
from datetime import datetime
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
x_frames, y_frames = get_sequences(data, sequence_length, num_atoms, dictionary_size)

dataset = MatchingPursuitDataset(x_frames, y_frames)
print(x_frames.shape, y_frames.shape)
# Create a DataLoader
batch_size = 64  # Define your batch size
shuffle = True   # Shuffle the data every epoch

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
# # Model parameters
learning_rate = 0.001
amount_epochs = 100
weight_decay = 0.0001

loss_function = MultiClassCoeffiecentLoss(num_atoms, dictionary_size)
model = RNNEmbeddingModel(num_categories=dictionary_size, num_atoms=num_atoms, embedding_dim=40,
                          hidden_size=128, num_layers=2) 
opt = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(amount_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        opt.zero_grad()
        predicted_indices, predicted_coefficients = model(inputs)
        loss = loss_function(predicted_indices, predicted_coefficients, targets)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{amount_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    running_loss = 0.0

timestampStr = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
torch.save(model.state_dict(), f"model_weights_{timestampStr}.pth")

