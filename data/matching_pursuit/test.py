import torch
import torch.nn as nn
import numpy as np
import sys
np.set_printoptions(precision=5, suppress=True, threshold=sys.maxsize)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

class CustomEmbedding(nn.Module):
    def __init__(self, in_channels=3, embedding_size=64):
        super(CustomEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, embedding_size, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x shape: (batch_size, block_size, num_atoms, 3)
        # Reshape x to: (batch_size*block_size, 3, num_atoms) for Conv1d
        batch_size, block_size, num_atoms, _ = x.size()
        x = x.view(batch_size * block_size, 3, num_atoms)  # Combine batch and block for batched processing
        x = self.conv1(x)
        # Aggregate features across atoms, reshape back to include block_size
        x = x.sum(dim=2).view(batch_size, block_size, -1)  #
        return x

# # Example usage
# batch_size = 64
# block_size = 256
# seq_len = 100
# embedding = 100
# model = CustomEmbedding(embedding_size=embedding)
# input_triplets = torch.rand(batch_size, block_size, seq_len, 3)  # Random example data
# embedded = model(input_triplets)
# print(embedded.shape)
a = torch.arange(1*8*12).reshape((1,8,12))
print(a.cpu().numpy())
even = a[:,:,::2]
odd = a[:,:,1::2]
b = torch.cat((even,odd), dim=2)
print(b.cpu().numpy())
