import numpy as np
import pickle
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from warnings import filterwarnings
from tqdm import tqdm
import os
filterwarnings('ignore')
sys.path.append('./../src')
import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from inference import *
from utils import SCDATA_sampler, MMD_loss
from torch.utils.data import DataLoader
from model import *

def compute_metrics(model_name = 'full_go_regular', seed = 42, latdim = 70, save=True):
	
    # read different trained models here
    savedir = f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}' 

    model = torch.load(f'{savedir}/best_model.pt')
    mode = 'CMVAE'

    def evaluate_model(fold):

        if fold == 'test':
            MMD, MSE, KLD, L1 = evaluate_single_leftout(model, savedir, model.device, mode)
        elif fold == 'train':
            MMD, MSE, KLD, L1 = evaluate_single_train(model, savedir, model.device, mode)
        elif fold == 'double':
            MMD, MSE, KLD, L1 = evaluate_double(model, savedir, model.device, mode, temp=1)

        # Prepare data in the format mean ± std
        data = {
            'Metric': ['MMD', 'MSE', 'KLD', 'L1'],
            'Values': [MMD, MSE, KLD, L1]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        return df

    # ## train 
    # print("train metrics")
    # df_train = evaluate_model(fold='train')
    # df_train['mode'] = 'train'

    ## test
    # print("test metrics")
    # df_test = evaluate_model(fold='test')
    # df_test['mode'] = 'test'

    ## doubles
    print("double metrics")
    df_double = evaluate_model(fold="double")
    df_double['mode'] = 'double'

    ## concat
    df = pd.concat([df_double]).reset_index(drop=True)
    df['seed'] = seed
    df['latdim'] = latdim
    df['model_name'] = model_name

    #save
    if save:
        df.to_csv(os.path.join('./../../../','result', 'uhler', model_name, f'seed_{seed}_latdim_{latdim}', f'{model_name}_mmd_r2_rmse_metrics_summary.tsv'),sep='\t')
    return df
    
#only one seed
seeds = [42,13]
df_l = []
tuplas = [
        ('full_go_sena_delta_0', 105),
        ('full_go_sena_delta_0', 70),
        ('full_go_sena_delta_0', 35),
        ('full_go_sena_delta_0', 10),
        ('full_go_sena_delta_0', 5),
          ('full_go_sena_delta_2', 105),
          ('full_go_sena_delta_2', 70),
          ('full_go_sena_delta_2', 35),
          ('full_go_sena_delta_2', 10),
          ('full_go_sena_delta_2', 5),
        ('full_go_sena_delta_3', 105),
        ('full_go_sena_delta_3', 70),
        ('full_go_sena_delta_3', 35),
        ('full_go_sena_delta_3', 10),
        ('full_go_sena_delta_3', 5),
           ('full_go_sena_delta_1', 105),
           ('full_go_sena_delta_1', 70),
           ('full_go_sena_delta_1', 35),
           ('full_go_sena_delta_1', 10),
           ('full_go_sena_delta_1', 5),
           ('full_go_regular', 105),
           ('full_go_regular', 70),
           ('full_go_regular', 35),
           ('full_go_regular', 10),
           ('full_go_regular', 5)
          ]

## model name
for seed in seeds:
    for model_name, latdim in tuplas:
        df_l.append(compute_metrics(model_name = model_name, seed = seed, latdim = latdim, save=False))

#create dfs
df = pd.concat(df_l)
df.to_csv(os.path.join('./../../result/uhler/post_analysis/performance_metrics/performance_table.tsv'),sep='\t')
print(df)


