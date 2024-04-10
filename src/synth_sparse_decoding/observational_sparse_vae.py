import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import src.utils as ut
import importlib
from sklearn import datasets
import pandas as pd
import imageio
importlib.reload(ut)

class SparseVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SparseVAE, self).__init__()
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU())

        ## latent space
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)  # Mean μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance σ^2
        
        # Decoder
        self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, input_dim))     

        ## mse loss
        self.mseloss = nn.MSELoss()
        self.maeloss = nn.L1Loss()
        
    def encode(self, x):

        ## mlp
        h = self.encoder(x)

        #compute mu and std
        mu, std = self.fc_mean(h), self.fc_logvar(h)

        return mu, std
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps*std
    
    def decode(self, z):
        return self.generator(z)
    
    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar):

        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = self.mseloss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + 0.5*KLD, MSE, KLD

# Example usage
N = 1000  # Number of samples
X, Z = ut.generate_synthetic_data(N, input_dim = 5, latent_dim = 2)
#X = StandardScaler().fit_transform(X)

# Init model, optimizer and dataset
device = 'cuda'
model = SparseVAE(X.shape[1], X.shape[1]*10, Z.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_loader = DataLoader(X, batch_size=64, shuffle=False)

# Training loop
epochs = 150
nbatch = len(train_loader)
for epoch in range(epochs):

    train_loss, train_mse, train_kld = 0, 0, 0
    for data in train_loader:
        
        x = data.to(device).float()
        optimizer.zero_grad()
        x_pred, mu, logvar = model(x)
        loss, loss_mse, loss_kld = model.vae_loss(x_pred, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        #losses
        train_loss += loss.item()
        train_mse += loss_mse.item()
        train_kld += loss_kld.item()
    
    if not epoch%10:
        print(f"Epoch: {epoch} Average loss: {train_loss / nbatch:.4f}, Average MSE: {train_mse / nbatch:.4f}, Average KLD: {train_kld / nbatch:.4f}")
    

## plot latent
ut.plot_orig_space(train_loader, model)
ut.plot_latent_space(train_loader, model, Z)
ut.plot_vae_x_z_relationship(train_loader, model, Z)