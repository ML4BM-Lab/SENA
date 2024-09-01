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
importlib.reload(ut)

# Define the DNN
class RegressionDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionDNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.lrelu = nn.LeakyReLU()                             # Activation function
        self.layer2 = nn.Linear(hidden_size, output_size*4) # Hidden layer to output layer
        self.lrelu2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(output_size*4, output_size*2)
        self.lrelu3 = nn.LeakyReLU()
        self.layer4 = nn.Linear(output_size*2, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.lrelu(out)
        out = self.layer2(out)
        out = self.lrelu2(out)
        out = self.layer3(out)
        out = self.lrelu3(out)
        out = self.layer4(out)
        return out

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Example usage
device = 'cuda'
N = 2000  # Number of samples
X, Z = ut.generate_synthetic_data(N, input_dim = 5, latent_dim = 2)
print(f"Mean of the features' variance: {X.var(axis=0).mean()}")

## init the model
model = RegressionDNN(Z.shape[1], Z.shape[1], X.shape[1]).to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Using Adam optimizer

# Create dataset
dataset = CustomDataset(Z, X)
train_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Training loop
num_epochs = 250
for epoch in range(num_epochs):

    train_loss = 0
    for z,x in train_loader:
        
        # Forward pass
        outputs = model(z.cuda().float())
        loss = criterion(outputs, x.cuda().float())
        train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch % 10) == 0:
        print(f'Epoch {epoch}, Loss: {train_loss/len(train_loader)}')
 
##pplot
ut.plot_x_z_relationship(train_loader, model)