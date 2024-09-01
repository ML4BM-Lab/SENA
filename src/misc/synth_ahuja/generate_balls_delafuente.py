import numpy as np
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader
from pygame import gfxdraw, init
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import cv2
import os
from PIL import ImageColor
from functools import reduce


class Shapes(torch.utils.data.Dataset):
    
    def __init__(
        self,
        n_shapes: int = 1,
    ):
        super(Shapes, self).__init__()
        pygame.init()
        self.screen_dim = 64
        self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.n_shapes = n_shapes

    def __len__(self) -> int:
        # arbitrary since examples are generated online.
        return 20000

    def draw_shape(self, shape_type, color, position):

        # Set position based on the specified position
        if position == "down":
            x,y = 26,5
        elif position == "up":
            x,y = 26,50
        elif position == "left":
            x,y = 5,26
        elif position == "right":
            x,y = 50,26

        # Draw shape based on the specified shape type
        if shape_type == "triangle":
            side = 5
            points = [(x, y), (x+side, y-side), (x-side, y-side)]
            pygame.draw.polygon(self.surf, color, points)

        elif shape_type == "rectangle":
            side = 6
            rect = pygame.Rect(x, y, side, side)
            pygame.draw.rect(self.surf, color, rect)

        elif shape_type == "circle":
            radius = 4
            #center = (x + radius, y + radius)
            # pygame.draw.circle(self.surf, color, center, radius)
            gfxdraw.aacircle(
                self.surf, int(x), int(y), int(radius), ImageColor.getcolor(color, "RGB")
            )
            gfxdraw.filled_circle(
                self.surf, int(x), int(y), int(radius), ImageColor.getcolor(color, "RGB")
            )

    def __getitem__(self, item):
        raise NotImplemented()

class SparseShapes(Shapes):
    def __init__(
        self,
        n_shapes: int = 3,
        colors: list = ["#777acd", "#72a65a", "#c65896", "#c77442"],
        positions: list = ["up", "down", "left", "right"],
    ):
        super().__init__(n_shapes=n_shapes)
        self.colors = colors
        self.positions = positions

    def __getitem__(self, item):

        self.surf.fill((255, 255, 255))
        z = []

        for i in range(self.n_shapes):

            # Randomly select a shape, color, and position
            shape_type = np.random.choice(["triangle", "rectangle", "circle"])
            color = self.colors[i]
            position = self.positions[i]
            z.append((shape_type,color, position))

            # Generate the shape based on selected parameters
            self.draw_shape(shape_type, color, position)
        
        #get zs
        xi = 0
        
        ## get xo
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        xo = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
        return xi, z, xo

class SingleShape(Shapes):
    def __init__(
        self,
        shapes: list = ["triangle", "rectangle", "circle"],
        colors: list = ["#777acd", "#72a65a", "#c65896", "#c77442"],
        positions: list = ["up", "down", "left", "right"]
        
    ):
        super().__init__(n_shapes=1)
        self.shapes = shapes
        self.colors = colors
        self.positions = positions
        
    def generate_ci(self, shape, color, pos):

        #lets generate a conditioned input, which is 
        #a onehotencoder describing the characteristics of the object
        ci = np.zeros((len(self.shapes)+len(self.colors)+len(self.positions),))
        ci[self.shapes.index(shape)] = 1
        ci[len(self.shapes)+self.colors.index(color)] = 1
        ci[len(self.shapes)+len(self.colors)+self.positions.index(pos)] = 1
        return ci

    def __getitem__(self, item):

        self.surf.fill((255, 255, 255))

        # Randomly select a shape, color, and position
        shape_type = np.random.choice(self.shapes)
        color = np.random.choice(self.colors)
        position = np.random.choice(self.positions)
        label = (shape_type, color, position)

        # Generate the shape based on selected parameters
        self.draw_shape(shape_type, color, position)
        
        #get zs
        xi = np.random.normal(0, 1, self.screen_dim**2)
        xci = self.generate_ci(shape_type, color, position)
        
        ## get xo
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        xo = np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        
        return xi, xci, xo, label

class CustomDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]


"""
ENVIRONMENT 1
"""

## get sparseShapes
dataset = SingleShape()

## testing
# n = 10
# for i in range(n):
#     xi, ci, xo, label = dataset[i]
#     print(label)
#     cv2.imwrite(os.path.join('crl','env1_exp','figs',f'img{i}.png'), xo)


##generate dataset
n = 5000
input, cinput, output, labels = [], [], [], []

for i in range(n):
    xi, ci, xo, label = dataset[i]

    input.append(xi)
    cinput.append(ci)
    output.append(xo.flatten())
    labels.append(label)

## convert to dataset
input = torch.from_numpy(np.vstack(input))
cinput = torch.from_numpy(np.vstack(cinput))
output = torch.from_numpy(np.vstack(output))
labels = np.vstack(labels)

"""
GENERATE DATASET
"""
dataset = CustomDataset(input, cinput, output)
train_loader = DataLoader(dataset, batch_size=128, shuffle=False)

"""
INIT MODEL
"""

# Define the DNN
class RegularVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(RegularVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc31 = nn.Linear(hidden_dim, latent_dim)
        self.enc32 = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, output_dim)

    def encode(self, x):
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))

        return self.enc31(h2), self.enc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.dec1(z))
        h2 = F.relu(self.dec2(h3))
        return self.dec3(h2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def vae_loss(self, output, output_pred, mu, logvar):

        # Reconstruction loss
        recon_loss = F.mse_loss(output, output_pred, reduction='mean')

        # KL divergence loss
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + 2*kl_divergence

        return loss, recon_loss, kl_divergence

## define device
device = 'cuda'

## init the model
model = RegularVAE(input.shape[1]+cinput.shape[1], input.shape[1], cinput.shape[1], reduce(lambda x,y: x*y, output.shape[1:])).to(device)

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    
    train_loss = 0
    for xi, xc, output in train_loader:
        
        # Forward pass
        model_input = torch.cat((xi, xc),1).cuda().float()
        pred_output, mu, logvar = model(model_input)
        loss, mse, vae = model.vae_loss(output.cuda().float(), pred_output, mu, logvar)
        train_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch % 10) == 0:
        print(f'Epoch {epoch}, Loss: {train_loss/len(train_loader)}, mse: {mse}, vae: {vae}')


#plot last image
cv2.imwrite(os.path.join('crl','env1_exp','figs',f'lastimg.png'), output[0,:].reshape((64, 64, 3)).numpy())
cv2.imwrite(os.path.join('crl','env1_exp','figs',f'lastimg_pred.png'), pred_output[0,:].detach().cpu().reshape((64, 64, 3)).numpy())
