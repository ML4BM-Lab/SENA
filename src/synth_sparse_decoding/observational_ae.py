import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def load_simple_dataset(n, k, mean, variance):
    """
    Generates a dataset consisting of n samples, each containing k points from a 2D Gaussian distribution.
    
    Parameters:
    - n: The number of samples to generate.
    - k: The number of points in each sample.
    - mean: A list or array of length 2 specifying the mean of the Gaussian distribution.
    - variance: A list or array of length 2 specifying the diagonal elements of the covariance matrix.
    
    Returns:
    - A numpy array of shape (n, k, 2) containing the generated samples.
    """
    # Ensure the mean and variance are correctly shaped
    mean = np.array(mean)
    assert mean.shape == (2,), "Mean must be a 2-element array."
    
    variance = np.array(variance)
    assert variance.shape == (2,), "Variance must be a 2-element array."
    
    # Create a diagonal covariance matrix from the variance
    covariance_matrix = np.diag(variance)
    
    # Generate the samples
    samples = np.random.multivariate_normal(mean, covariance_matrix, (n, k//mean.shape[0]))
    
    return samples.reshape(n,-1)

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(input_size, latent_size)
        self.relu = nn.ReLU()
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        h = self.encoder(x)
        x = self.relu(h)
        x = self.decoder(x)
        return x, h


# Model Parameters
input_size = 10
latent_size = 3

# Model, Loss Function, and Optimizer
model = Autoencoder(input_size, latent_size).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Data Loading
dataset = []
for mean in [(5,5), (0,0), (-5,-5)]:
    dataset.append(load_simple_dataset(n = 1000, k = input_size, mean = mean, variance = [0.05, 0.05]))
train_dataset = np.vstack(dataset)
train_loader = DataLoader(torch.tensor(train_dataset).float(), batch_size=128, shuffle=True)

# Training Loop
epochs = 100
for epoch in range(epochs):
    for data in train_loader:
        optimizer.zero_grad()
        output, _ = model(data.cuda())
        loss = criterion(output, data.cuda())
        loss.backward()
        optimizer.step()

    if not epoch%5:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        for i,experiment in enumerate(dataset):
            _, h = model(torch.tensor(experiment[0:128,:]).float().cuda())
            print(f'latent space {i} -> h: {h.detach().cpu().numpy().mean(axis=0)}')


# Add any additional training, validation, testing, saving model as needed.

