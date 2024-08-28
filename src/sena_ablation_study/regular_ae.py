import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

class SENA(nn.Module):

    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda', sp = 0):
        super(SENA, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device, sp=sp)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

def run_model(mode, seed, analysis = 'interpretability'):

    if mode == 'regular':

        if analysis == 'interpretability':

            model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

        elif analysis == 'efficiency':

            sena_model = SENA(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict).to('cuda')

            sena_trainable = sena_model.encoder.mask.sum().__float__() + (len(gos)*adata.X.shape[1] + sena_model.decoder.bias.shape[0])
            latent_size = m.ceil(sena_trainable / (2*adata.X.shape[1]+2)) #  sena_W / (2*input+2)]
 
            model = Autoencoder(input_size = adata.X.shape[1], latent_size = latent_size).to('cuda')
            #params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

    elif mode == 'regular_orig':
        model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

    elif mode[:4] == 'sena':
        
        sp_num = float(mode.split('_')[1])
        sp = eval(f'1e-{int(sp_num)}') if sp_num > 0 else 0
        model = SENA(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict, sp=sp).to('cuda')

    elif mode[:2] == 'l1':
        model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')

        """l1 reg"""
        l1_lambda_num = float(mode.split('_')[1])
        l1_lambda = eval(f'1e-{int(l1_lambda_num)}')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ##results
    results = []
    
    # Training Loop
    epochs = 100 if analysis == 'efficiency' else 50
    report_epoch = 2 if analysis == 'efficiency' else 5

    """train"""
    for epoch in range(epochs):

        epoch_mse = []
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
            epoch_mse.append(total_loss.item())

        """metrics"""
        if not epoch%report_epoch:
            if analysis == 'interpretability':
                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
                summary_analysis_ep = st.compute_outlier_activation_analysis(ttest_df, adata, ptb_targets, mode = mode)
                summary_analysis_ep['epoch'] = epoch
                summary_analysis_ep['mse'] = np.mean(epoch_mse)

            elif analysis == 'efficiency':
                
                if mode[:4] == 'sena':
                    sparsity = (((model.encoder.mask == 0).sum() / (model.encoder.mask.shape[0] * model.encoder.mask.shape[1]))).__float__()
                else:
                    #st.plot_weight_distribution(model, epoch, mode)
                    #lets compute sparsity to every weight that is |w| < 1e-4
                    # w_abs_mean = model.encoder.weight.abs().mean()
                    # sparse_weights = ((model.encoder.weight.abs() - w_abs_mean) <= 1e-4).sum()
                    sparse_weights = (model.encoder.weight.abs() <= 1e-4).sum()
                    sparsity = (sparse_weights / (model.encoder.weight.shape[0] * model.encoder.weight.shape[1])).__float__()

                summary_analysis_ep = pd.DataFrame({'epoch': epoch, 'mse': np.mean(epoch_mse), 'mode': mode, 'sparsity':sparsity}, index = [0])

            elif analysis == 'lcorr':
                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
                summary_analysis_ep = st.compute_latent_correlation_analysis(model, adata, ptb_targets, gos, ttest_df)
                summary_analysis_ep['epoch'] = epoch
            
            results.append(summary_analysis_ep)

        print(f'Epoch {epoch+1}, Loss: {np.mean(epoch_mse)}')
    
    #add seed
    results_df = pd.concat(results)
    results_df['seed'] = seed

    return results_df

if __name__ == '__main__':

    ##define inputs
    modeltype = sys.argv[1]
    analysis = sys.argv[2]
    subsample = sys.argv[3]

    #init 
    fpath = os.path.join('./../../result/ablation_study',f'ae_{modeltype}')
    fname = os.path.join(fpath,f'autoencoder_{modeltype}_ablation_{analysis}_1layer_{subsample}')
    results_dict = {}

    #if folder does not exist, create it
    if not os.path.exists(fpath):
         os.makedirs(fpath)

    #seeds
    results = []
    for i in range(5): #5

        # Load data
        adata, ptb_targets, ptb_targets_ens, gos, rel_dict = st.load_norman_2019_dataset(subsample=subsample)
        train_loader = DataLoader(torch.tensor(adata.X.todense()).float(), batch_size=128, shuffle=True)

        #run the model
        results.append(run_model(mode = modeltype, seed = i, analysis = analysis))
        
    ##build the dataframe and save
    if analysis != 'lcorr':
        results_df = pd.concat(results).reset_index(drop=True)
        print(results_df)
        results_df.to_csv(fname+'.tsv',sep='\t')
    else:
        with open(fname+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
