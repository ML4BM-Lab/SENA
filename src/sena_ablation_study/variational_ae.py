import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sena_tools as st
import importlib
from scipy.stats import ttest_ind
import pandas as pd
import math as m
importlib.reload(st)
import os
import pickle
import sys
import torch.nn.functional as F
import time

class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Linear(input_size, latent_size)  # Mean for latent space
        self.encoder_var = nn.Linear(input_size, latent_size)  # Log variance for latent space
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x):
        # Get the mean and log variance from the encoder
        mean = self.encoder(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean, var):
        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * var)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        z = mean + eps * std  # Reparameterization trick
        return z

    def forward(self, x):

        # Encoding step: get mean and log variance
        mean, var = self.encode(x)
        
        # Reparameterization step: sample from latent space
        z = self.reparameterize(mean, var)
        
        # Decoding step: reconstruct the input
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var

class VAE2Layers(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE2Layers, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Linear(input_size, latent_size)
        self.encoder_mean = nn.Linear(latent_size, latent_size)  # Mean for latent space
        self.encoder_var = nn.Linear(latent_size, latent_size)  # Log variance for latent space
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x):
        # Get the mean and log variance from the encoder
        x = self.encoder(x)
        x = self.lrelu(x)
        mean = self.encoder_mean(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean, var):
        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * var)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        z = mean + eps * std  # Reparameterization trick
        return z

    def forward(self, x):

        # Encoding step: get mean and log variance
        mean, var = self.encode(x)
        
        # Reparameterization step: sample from latent space
        z = self.reparameterize(mean, var)
        
        # Decoding step: reconstruct the input
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var

class SENAVAE(nn.Module):
    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda', sp = 0):
        super(SENAVAE, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder 
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device, sp=sp)  # Mean for latent space
        self.encoder_var = nn.Linear(input_size, latent_size)  # Log variance for latent space
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x):
        # Get the mean and log variance from the encoder
        mean = self.encoder(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean, var):
        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * var)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        z = mean + eps * std  # Reparameterization trick
        return z

    def forward(self, x):

        # Encoding step: get mean and log variance
        mean, var = self.encode(x)
        
        # Reparameterization step: sample from latent space
        z = self.reparameterize(mean, var)
        
        # Decoding step: reconstruct the input
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var 

class SENADeltaVAE(nn.Module):
    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda', sp = 0):
        super(SENADeltaVAE, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        # Encoder 
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device, sp=sp)
        self.encoder_mean = nn.Linear(latent_size, latent_size)  # Mean for latent space
        self.encoder_var = nn.Linear(latent_size, latent_size)  # Log variance for latent space
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x):
        # Get the mean and log variance from the encoder
        x = self.encoder(x)
        x = self.lrelu(x)
        mean = self.encoder_mean(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean, var):
        # Sample from the latent space using the reparameterization trick
        std = torch.exp(0.5 * var)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal tensor
        z = mean + eps * std  # Reparameterization trick
        return z

    def forward(self, x):

        # Encoding step: get mean and log variance
        mean, var = self.encode(x)
        
        # Reparameterization step: sample from latent space
        z = self.reparameterize(mean, var)
        
        # Decoding step: reconstruct the input
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var 

def run_model(mode, seed, analysis = 'interpretability', beta=1):

    def vae_loss(reconstructed_x, x, mean, var, beta):
        # Reconstruction loss (MSE or Binary Cross-Entropy)
        reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
        
        logvar = torch.log(var)
        kl_divergence = torch.sum(mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)/x.shape[0]

        # Total VAE loss
        return reconstruction_loss, beta*kl_divergence

    ##measure time
    starttime = time.time()

    if mode == 'regular':

        if nlayers == 1:
            model = VAE(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')
        else:
            model = VAE2Layers(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

    elif mode[:4] == 'sena':

        if 'delta' in mode:
            sp_num = float(mode.split('_')[2])
            sp = eval(f'10**-{sp_num}') if sp_num > 0 else 0
            model = SENADeltaVAE(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict, sp=sp).to('cuda')

        else:
            sp_num = float(mode.split('_')[1])
            sp = eval(f'10**-{sp_num}') if sp_num > 0 else 0
            model = SENAVAE(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict, sp=sp).to('cuda')

    elif mode[:2] == 'l1':

        if nlayers == 1:
            model = VAE(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')
        else:
            model = VAE2Layers(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

        """l1 reg"""
        l1_lambda_num = float(mode.split('_')[1])
        l1_lambda = eval(f'1e-{int(l1_lambda_num)}')

    #define criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ##results
    results = []
    
    # Training Loop
    epochs = 250 if analysis == 'efficiency' else 250
    report_epoch = 2 if analysis == 'efficiency' else 10

    """train"""
    for epoch in range(epochs):

        epoch_train_mse, epoch_train_kl = [], []

        for batched_exp in train_loader:

            optimizer.zero_grad()
            reconstructed_x, mean, logvar = model(batched_exp.cuda())
            train_mse, train_KL = vae_loss(reconstructed_x, batched_exp.cuda(), mean, logvar, beta=beta)

            if mode[:2] != 'l1':
                total_loss = train_mse + train_KL
            else:
                #compute total_loss
                total_loss = train_mse + train_KL + l1_lambda * torch.norm(model.encoder.weight, p=1)
            
            total_loss.backward()
            optimizer.step()

            #add losses
            epoch_train_mse.append(train_mse.item().__float__())
            epoch_train_kl.append(train_KL.item().__float__())

        """metrics"""
        if not epoch%report_epoch:

            if analysis == 'interpretability':

                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
                summary_analysis_ep = st.compute_outlier_activation_analysis(ttest_df, adata, ptb_targets, mode = mode)
                summary_analysis_ep['epoch'] = epoch
                print(summary_analysis_ep['recall_at_100'].values[0])

            elif analysis == 'efficiency':
                
                with torch.no_grad():
                    reconstructed_x, mean, logvar = model(test_data.cuda())
                    test_mse, test_KL = vae_loss(reconstructed_x.detach().cpu(), 
                                                 test_data.cuda().detach().cpu(), mean.detach().cpu(),
                                                 logvar.detach().cpu(), beta=beta)
                
                #compute sparsity
                sparsity = 0
                
                # sparsity = st.compute_sparsity_contribution(model, test_data.cuda(), mode=mode, sparsity_th = 1e-5)
                summary_analysis_ep = pd.DataFrame({'epoch': epoch, 'train_mse': np.mean(epoch_train_mse),
                                                    'test_mse': test_mse.__float__(), 'train_KL': np.mean(epoch_train_kl),
                                                    'test_KL': test_KL.__float__(),
                                                    'mode': mode, 'sparsity':sparsity}, index = [0])

                print(f'Epoch {epoch+1}, TEST MSE Loss: {test_mse.__float__()}, TEST KL Loss: {test_KL.__float__()}')
            
            results.append(summary_analysis_ep)

        print(f'Epoch {epoch+1}, TRAIN MSE Loss: {np.mean(epoch_train_mse)}, TRAIN KL Loss: {np.mean(epoch_train_kl)}')
    
    #add seed
    results_df = pd.concat(results)
    results_df['seed'] = seed
    results_df['time'] = time.time() - starttime

    return results_df

if __name__ == '__main__':

    ##define inputs
    modeltype = sys.argv[1]
    analysis = sys.argv[2]
    dataset = sys.argv[3]
    num_gene_th = 5 if '_' not in dataset else int(dataset.split('_')[-1])
    nlayers = 1 if len(sys.argv) < 5 else sys.argv[4]
    beta = 1.0 if len(sys.argv) < 6 else float(sys.argv[5])
    
    #define seeds
    nseeds = 2 if analysis == 'efficiency' else 3

    #init 
    fpath = os.path.join('./../../result/ablation_study/',f'vae_{modeltype}')
    fname = os.path.join(fpath,f'vae_{modeltype}_ablation_{analysis}_{nlayers}layer_{dataset}_beta_{beta}')
    results_dict = {}

    #if folder does not exist, create it
    if not os.path.exists(fpath):
         os.makedirs(fpath)

    #seeds
    results = []
    for i in range(nseeds):

        # Load data
        if 'norman' in dataset:
            adata, ptb_targets, ptb_targets_ens, gos, rel_dict, gene_go_dict, ens_gene_dict = st.load_norman_2019_dataset(num_gene_th=num_gene_th)

        #split train/test
        train_data, test_data = train_test_split(torch.tensor(adata.X.todense()).float(), stratify = adata.obs['guide_ids'], test_size = 0.1)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        #run the model
        results.append(run_model(mode = modeltype, seed = i, analysis = analysis, beta=beta))
        
    ##build the dataframe and save
    if analysis != 'lcorr':
        results_df = pd.concat(results).reset_index(drop=True)
        print(results_df)
        results_df.to_csv(fname+'.tsv',sep='\t')
    else:
        with open(fname+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
