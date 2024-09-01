import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os
import src.utils as ut

## understanding KL in vae https://kvfrans.com/deriving-the-kl/

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        ## latent space
        self.fcmu = nn.Linear(hidden_dim, latent_dim)  # Mean μ
        self.fcstd = nn.Linear(hidden_dim, latent_dim)  # Log variance σ^2
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):

        ## mlp
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))

        #compute mu and std
        mu, std = self.fcmu(h2), self.fcstd(h2)

        return mu, std
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        h4 = F.leaky_relu(self.fc4(h3))
        h5 = self.fc5(h4)
        return h5
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE+KLD, MSE, KLD

# Example usage
N = 1000  # Number of samples
X, latent_data = ut.generate_synthetic_data(N)
print(f"mean of latent features: {latent_data.mean(axis=0)}")

# Init model, optimizer and dataset
device = 'cuda'
model = VAE(X.shape[1], X.shape[1], latent_data.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader = DataLoader(X, batch_size=64, shuffle=True)

# Training loop
epochs = 400
nbatch = len(train_loader)
for epoch in range(epochs):

    train_loss, train_mse, train_kld = 0,0,0
    for data in train_loader:
        
        x = data.to(device).float()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(x)
        loss, loss_mse, loss_kld = model.vae_loss(recon_batch, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_mse += loss_mse.item()
        train_kld += loss_kld.item()
        optimizer.step()
    
    if not epoch%10:
        print(f"Epoch: {epoch} Average loss: {train_loss / nbatch:.4f}, Average MSE: {train_mse / nbatch:.4f}, Average KLD: {train_kld / nbatch:.4f}")

## plot latent
ut.plot_orig_space(model, X)
ut.plot_latent_space(model, X, latent_data)