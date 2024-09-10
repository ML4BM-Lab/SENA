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
import time


class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Linear(input_size, latent_size)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

class Autoencoder2Layers(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder2Layers, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Linear(input_size, latent_size)
        self.delta = nn.Linear(latent_size, latent_size)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lrelu(x)
        x = self.delta(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

class SENA(nn.Module):

    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda', sp = 0):
        super(SENA, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device, sp=sp, bias=True)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

class SENADelta(nn.Module):

    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda', sp = 0):
        super(SENADelta, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device, sp=sp, bias=True)
        self.delta = nn.Linear(latent_size, latent_size)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.delta(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

def run_model(mode, seed, analysis, gene_go_dict, ens_gene_dict):

    ##measure time
    starttime = time.time()

    if mode == 'regular':
        if nlayers == 1:
            model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')
        else:
            model = Autoencoder2Layers(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

    elif mode[:4] == 'sena':

        if 'delta' in mode:
            sp_num = float(mode.split('_')[2])
            sp = eval(f'10**-{sp_num}') if sp_num > 0 else 0
            model = SENADelta(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict, sp=sp).to('cuda')

        else:
            sp_num = float(mode.split('_')[1])
            sp = eval(f'10**-{sp_num}') if sp_num > 0 else 0
            model = SENA(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict, sp=sp).to('cuda')

    elif mode[:2] == 'l1':

        if nlayers == 1:
            model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')
        else:
            model = Autoencoder2Layers(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

        """l1 reg"""
        l1_lambda_num = float(mode.split('_')[1])
        l1_lambda = eval(f'1e-{int(l1_lambda_num)}')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ##results
    results = []
    
    # Training Loop
    epochs = 250 if analysis == 'efficiency' else 250
    report_epoch = 2 if analysis == 'efficiency' else 10

    """train"""
    for epoch in range(epochs):

        epoch_train_mse = []
        for batched_exp in train_loader:
            optimizer.zero_grad()
            output = model(batched_exp.cuda())
            loss = criterion(output, batched_exp.cuda())

            if mode[:2] != 'l1':
                total_loss = loss
            else:
                #compute total_loss
                total_loss = loss + l1_lambda * torch.norm(model.encoder.weight, p=1)
            
            total_loss.backward()
            optimizer.step()
            epoch_train_mse.append(total_loss.item())

        """metrics"""
        if not epoch%report_epoch:

            if analysis == 'interpretability':
                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode, 
                                                    gene_go_dict=gene_go_dict, ensembl_genename_dict=ens_gene_dict, ptb_targets=ptb_targets)
                summary_analysis_ep = st.compute_outlier_activation_analysis(ttest_df, mode = mode)
                summary_analysis_ep['epoch'] = epoch

            elif analysis == 'efficiency':
                
                test_mse = torch.nn.functional.mse_loss(model(test_data.cuda()).detach().cpu(), test_data).__float__()
                sparsity = st.compute_sparsity_contribution(model, test_data.cuda(), mode=mode, sparsity_th = 1e-5)
                summary_analysis_ep = pd.DataFrame({'epoch': epoch, 'train_mse': np.mean(epoch_train_mse),
                                                    'test_mse': test_mse, 'mode': mode, 'sparsity':sparsity}, index = [0])

            elif analysis == 'lcorr':
                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
                summary_analysis_ep = st.compute_latent_correlation_analysis(model, adata, ptb_targets, gos, ttest_df)
                summary_analysis_ep['epoch'] = epoch
            
            results.append(summary_analysis_ep)

        print(f'Epoch {epoch+1}, Loss: {np.mean(epoch_train_mse)}')
    
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
    nlayers = 1 if len(sys.argv) < 5 else sys.argv[4]
    
    #define seeds
    nseeds = 2 if analysis == 'efficiency' else 3

    #init 
    fpath = os.path.join('./../../result/ablation_study',f'ae_{modeltype}')
    fname = os.path.join(fpath,f'autoencoder_{modeltype}_ablation_{analysis}_{nlayers}layer_{dataset}')
    results_dict = {}

    #if folder does not exist, create it
    if not os.path.exists(fpath):
         os.makedirs(fpath)

    #seeds
    results = []
    for i in range(nseeds):

        # Load data
        if dataset == 'norman':
            adata, ptb_targets, ptb_targets_ens, gos, rel_dict, gene_go_dict, ens_gene_dict = st.load_norman_2019_dataset()

        #split train/test
        dataset = torch.tensor(adata.X.todense()).float()
        train_data, test_data = train_test_split(dataset, stratify = adata.obs['guide_ids'], test_size = 0.1)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

        #run the model
        results.append(run_model(mode = modeltype, seed = i, analysis = analysis, gene_go_dict=gene_go_dict, ens_gene_dict=ens_gene_dict))
        
    ##build the dataframe and save
    if analysis != 'lcorr':
        results_df = pd.concat(results).reset_index(drop=True)
        print(results_df)
        results_df.to_csv(fname+'.tsv',sep='\t')
    else:
        with open(fname+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
