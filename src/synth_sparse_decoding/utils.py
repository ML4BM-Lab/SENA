import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import os
import imageio


def generate_synthetic_data(N, input_dim = 5, latent_dim = 2):

    # Input data generation
    cov_matrix = np.eye(latent_dim)
    cov_matrix[cov_matrix != 1] = 0.2
    means = np.zeros(latent_dim)
    
    Z = np.random.multivariate_normal(means, cov_matrix, N)
    
    ## input
    X = np.zeros((N, input_dim))
    X[:,0] = 3 * Z[:,0]
    X[:,1] = -1 * Z[:,1]
    X[:,2] = (1/2) * (Z[:,0] **2)
    X[:,3] = 4 * Z[:,1]
    X[:,4] = 6 * np.sin(Z[:, 1])

    # add noise
    X = X + 0.5 * np.random.randn(N, input_dim)

    return X, Z

def generate_deterministic_synth_data(N):

    # Input data generation
    Z = np.random.normal(0, 0.5, (N, 1))
    #cov_matrix = np.array([[1, 0.6],[0.6, 1]])
    #means = np.array([0,0])
    #Z = np.random.multivariate_normal(means, cov_matrix, N)
    
    # Function to calculate mean mu for the latent space
    a0 = 3 # x0
    a1 = -1/2 # x1
    #a2 = 1 # x2
    #a3 = 4 # x3

    ## generate X
    X = np.zeros((N, 2))
    X[:,0] = a0 * Z
    X[:,1] = a1 * Z
    
    return np.vstack(X), Z

"""
PLOTS
"""

def plot_latent_space(train_loader, model, Z):

    ##
    z_pred = []

    for x in train_loader:
        _, mu, std = model(x.cuda().float())
        z = model.reparameterize(mu,std)
        z_pred.append(z.detach().cpu().numpy())

    z_pred = np.vstack(z_pred)

    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(Z.shape[1], 1, figsize=(10, 10), constrained_layout=True)

    for i in range(Z.shape[1]): 
        
        zgt = Z[:,i]
        zpred = z_pred[:,i]

        # Fit a linear regression model
        lr = LinearRegression().fit(zpred.reshape(-1, 1), zgt)

        # Plotting
        ax = axes[i]
        #ax.hist(zpred, label='predicted distribution', bins = 50)
        #ax.hist(zgt, label = 'ground distribution', bins = 50, alpha = 0.8)
        ax.scatter(zpred, zgt, label = 'pred vs gt')
        ax.plot([zgt.min(), zgt.max()], [zgt.min(), zgt.max()], color='red', label='Linear Regression')
        ax.set_title(f'Latent Feature {i}')
        ax.set_xlabel(f'Z (estimated) - Dimension {i}')
        ax.set_ylabel(f'Z (ground truth) - Dimension {i}')
        ax.legend()
        
    plt.show()
    plt.savefig(os.path.join('figures','synth_data','vae','vae_latent_space.png'))

def plot_orig_space(train_loader, model):

    x_orig, x_pred = [], []

    for x in train_loader:
        x_orig.append(x)
        xpred, _, _ = model(x.cuda().float())
        x_pred.append(xpred.detach().cpu().numpy())

    x_pred = np.vstack(x_pred)
    x_orig = np.vstack(x_orig)

    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(x_pred.shape[1], 1, figsize=(10, 15), constrained_layout=True)

    for i in range(x_pred.shape[1]): 
        
        x_gt = x_orig[:,i]
        x_recons_ft = x_pred[:,i]

        # Fit a linear regression model
        #lr = LinearRegression().fit(x_recons_ft.reshape(-1,1), x_gt)
        
        # Plotting
        ax = axes[i]
        ax.scatter(x_recons_ft, x_gt, label='Data Points')
        ax.plot([x_gt.min(), x_gt.max()], [x_gt.min(), x_gt.max()], color='red', label='Linear Regression')
        ax.set_title(f'Latent Feature {i+1}')
        ax.set_xlabel(f'X (estimated) - Dimension {i+1}')
        ax.set_ylabel(f'X (ground truth) - Dimension {i+1}')
        ax.legend()
        
    plt.show()
    plt.savefig(os.path.join('figures','synth_data','vae','vae_reconstruction.png'))
 
def plot_vae_x_z_relationship(train_loader, model, Z):

    x_orig, x_pred, z_orig, z_pred = [], [], Z, []
   
    for x in train_loader:

        #get x orig
        x_orig.append(x)
       
        #generate x
        xpred, mu, logvar = model(x.cuda().float())
        x_pred.append(xpred.detach().cpu().numpy())

        #generate z
        z = model.reparameterize(mu,logvar)
        z_pred.append(z.detach().cpu().numpy())

    x_orig = np.vstack(x_orig)
    x_pred = np.vstack(x_pred)
    z_pred = np.vstack(z_pred)

    #z_pred = z_orig

    num_plots = x_orig.shape[1]

    # Create scatter plot
    plt.figure(figsize=(10, 6))  # Set figure size

    # Create subplots: one subplot for each feature
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3))

    # If there's only one feature, axs may not be an array, so we ensure it's iterable
    if num_plots == 1:
        axs = [axs]

    funct_dict = {0:0, 1: 1, 2:0, 3:1, 4:1}
    models_coeff_dict = {0:3, 1: -1, 2: 0.5, 3: 4, 4: 6}
    models_dict = {0: 'z_pred[:, funct_dict[i]]', 1: 'z_pred[:, funct_dict[i]]', 2: '(z_pred[:, funct_dict[i]])**2', 3: 'z_pred[:, funct_dict[i]]', 4: 'np.sin(z_pred[:, funct_dict[i]])'}
    model_funct_dict = {0: 'Z0', 1: 'Z1', 2: 'Z0^2', 3: 'Z1', 4: 'sin(Z1)'}
    gt_funct_dict = {0: '3*Z0', 1: '-1*Z1', 2: '1/2 * (Z0^2)', 3: '4*Z1', 4: '6*sin(Z1)'}
    plot_dict = {0:'zs', 1:'zs', 2: 'zs**2', 3: 'zs', 4: 'np.sin(zs)'}

    for i in range(num_plots):

        fit_model = LinearRegression(fit_intercept=False).fit((eval(models_dict[i])).reshape(-1,1), x_pred[:,i])
        
        axs[i].scatter(z_pred[:, funct_dict[i]], x_pred[:,i], color='blue')
        axs[i].set_title(f'Pred: {fit_model.coef_[0]:.4f} * {model_funct_dict[i]}, True: {gt_funct_dict[i]}')
        axs[i].set_xlabel(f'Z_{funct_dict[i]}')
        axs[i].set_ylabel(f'Xpred_{i}')

        zs = np.linspace(-3,3,30)
        xs = models_coeff_dict[i] * eval(plot_dict[i])
        axs[i].plot(zs, xs, '--', color='red')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show plot
    plt.show()
    plt.savefig(os.path.join('figures','synth_data','vae',f'vae_x_y_relationship.png'))


###
def plot_x_z_relationship(train_loader, model):

    z_orig, x_pred = [], []

    for z,_ in train_loader:
        z_orig.append(z)
        x_pred.append(model(z.cuda().float()).detach().cpu().numpy())

    z_orig = np.vstack(z_orig)
    x_pred = np.vstack(x_pred)

    num_plots = x_pred.shape[1]

    # Create scatter plot
    plt.figure(figsize=(10, 6))  # Set figure size

    # Create subplots: one subplot for each feature
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3))

    # If there's only one feature, axs may not be an array, so we ensure it's iterable
    if num_plots == 1:
        axs = [axs]

    funct_dict = {0:0, 1: 1, 2:0, 3:1, 4:1}
    models_coeff_dict = {0:3, 1: -1, 2: 0.5, 3: 4, 4: 6}
    models_dict = {0: 'z_orig[:, funct_dict[i]]', 1: 'z_orig[:, funct_dict[i]]', 2: '(z_orig[:, funct_dict[i]])**2', 3: 'z_orig[:, funct_dict[i]]', 4: 'np.sin(z_orig[:, funct_dict[i]])'}
    model_funct_dict = {0: 'Z0', 1: 'Z1', 2: 'Z0^2', 3: 'Z1', 4: 'sin(Z1)'}
    gt_funct_dict = {0: '3*Z0', 1: '-1*Z1', 2: '1/2 * (Z0^2)', 3: '4*Z1', 4: '6*sin(Z1)'}

    plot_dict = {0:'zs', 1:'zs', 2: 'zs**2', 3: 'zs', 4: 'np.sin(zs)'}

    for i in range(num_plots):

        fit_model = LinearRegression(fit_intercept=False).fit((eval(models_dict[i])).reshape(-1,1), (x_pred[:,i]).reshape(-1,1))
        
        axs[i].scatter(z_orig[:, funct_dict[i]], x_pred[:,i], color='blue')
        axs[i].set_title(f'Pred: {fit_model.coef_[0][0]:.4f} * {model_funct_dict[i]}, True: {gt_funct_dict[i]}')
        axs[i].set_xlabel(f'Z_{funct_dict[i]}')
        axs[i].set_ylabel(f'Xpred_{i}')

        zs = np.linspace(-3,3,30)
        xs = models_coeff_dict[i] * eval(plot_dict[i])
        axs[i].plot(zs, xs, '--', color='red')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Labeling
    plt.xlabel('Z original')
    plt.ylabel('Predicted X')

    # Show plot
    plt.show()
    plt.savefig(os.path.join('figures','synth_data','x_y_relationship.png'))

def generate_gif():

    # Folder containing images
    images_folder = os.path.join('figures','synth_data','vae')
    output_gif_path = os.path.join('figures','synth_data','vae_output.gif')

    # Sorting images by filename (assuming this gives them in the desired order)
    image_files = sorted([os.path.join(images_folder, file) for file in os.listdir(images_folder)])

    # Read images
    images = [imageio.imread(file) for file in image_files]

    # Write images to a gif
    imageio.mimsave(output_gif_path, images, fps=5) # fps is frames per second

    print(f"GIF saved at {output_gif_path}")
