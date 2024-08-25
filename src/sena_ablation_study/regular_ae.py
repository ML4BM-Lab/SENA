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

    def __init__(self, input_size, latent_size, relation_dict, device = 'cuda'):
        super(SENA, self).__init__()

        # Activation Functions
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder = st.NetActivity_layer(input_size, latent_size, relation_dict, device = device)
       
        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.lrelu(x)
        x = self.decoder(x)
        return x

def run_model(mode, seed, analysis = 'interpretability', metric = 'recall'):

    if mode == 'regular':
        if type == 'interpretability':
            model = Autoencoder(input_size = adata.X.shape[1], latent_size = len(gos)).to('cuda')
        elif type == 'efficiency':
            sena_model = SENA(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict).to('cuda')

            sena_trainable = sena_model.encoder.mask.sum().__float__() + (len(gos)*adata.X.shape[1] + sena_model.decoder.bias.shape[0])
            latent_size = m.ceil(sena_trainable / (2*adata.X.shape[1]+2)) #  sena_W / (2*input+2)]
 
            model = Autoencoder(input_size = adata.X.shape[1], latent_size = latent_size).to('cuda')
            #params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

    elif mode == 'sena':
        model = SENA(input_size = adata.X.shape[1], latent_size = len(gos), relation_dict=rel_dict).to('cuda')

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    ##results
    results = []
    
    """ epoch 0 """

    if type == 'interpretability':
        ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
        summary_analysis_ep = st.compute_outlier_activation_test(ttest_df, adata, ptb_targets, mode = mode)
    elif type == 'efficiency':
        summary_analysis_ep = {}

    #knockout = 'combined'
    #st.plot_knockout_distribution(ttest_df, epoch = 0, mode = mode, seed = seed, adata=adata, ptb_targets=ptb_targets, knockout = knockout)
   
    ##compute mse
    with torch.no_grad():
        epoch_mse = []
        for batched_exp in train_loader:
            output = model(batched_exp.cuda())
            loss = criterion(output, batched_exp.cuda())
            epoch_mse.append(loss.item())

        summary_analysis_ep['epoch'] = 0
        summary_analysis_ep['mse'] = np.mean(epoch_mse)

    ##append result
    results.append(summary_analysis_ep)

    # Training Loop
    epochs = 100 # 40 
    for epoch in range(epochs):

        epoch_mse = []
        for batched_exp in train_loader:
            optimizer.zero_grad()
            output = model(batched_exp.cuda())
            loss = criterion(output, batched_exp.cuda())
            loss.backward()
            optimizer.step()
            epoch_mse.append(loss.item())

        print(f'Epoch {epoch+1}, Loss: {np.mean(epoch_mse)}')

        if not (epoch+1)%5: #2

            if type == 'interpretability':
                ttest_df = st.compute_activation_df(model, adata, gos, scoretype = 'mu_diff', mode = mode)
                #st.plot_knockout_distribution(ttest_df, epoch = 0, mode = mode, seed = seed, adata=adata, ptb_targets=ptb_targets, knockout = knockout)
                summary_analysis_ep = st.compute_outlier_activation_test(ttest_df, adata, ptb_targets, mode = mode)
            elif type == 'efficiency':
                summary_analysis_ep = {}

            summary_analysis_ep['epoch'] = epoch
            summary_analysis_ep['mse'] = np.mean(epoch_mse)

            results.append(summary_analysis_ep)

    results_df = pd.DataFrame(results)
    results_df['seed'] = seed

    return results_df

if __name__ == '__main__':

    ##define inputs
    modeltype = sys.argv[1]
    analysis = sys.argv[2]
    metric = sys.argv[3]
    results_dict = {}

    #seeds
    results = []
    for i in range(5): #5

        # Load data
        adata, ctrl_samples, ko_samples, ptb_targets, gos, rel_dict = st.load_norman_2019_dataset()
        train_loader = DataLoader(torch.tensor(adata.X.todense()).float(), batch_size=128, shuffle=True)

        #run the model
        results.append(run_model(mode = modeltype, seed = i, analysis = analysis, metric = metric))
        
    ##build the dataframe and save
    results_df = pd.concat(results).reset_index(drop=True)
    print(results_df)
    results_df.to_csv(os.path.join('./../../result/ablation_study',f'ae_{modeltype}',f'autoencoder_{modeltype}_ablation_{analysis}_1layer_{metric}.tsv'),sep='\t')
