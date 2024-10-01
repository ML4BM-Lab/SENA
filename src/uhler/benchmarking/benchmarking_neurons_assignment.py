import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pickle
import torch
import pandas as pd 
import os
import seaborn as sns
import graphical_models as gm
from tqdm import tqdm
from utils import get_data
import scipy.stats as stats
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import ttest_ind
import numpy as np
from utils import get_data
from collections import Counter
from scipy.stats import gaussian_kde
import utils as ut



"""compute activation scores"""
def compute_activation_scores(layer_name, model, adata, idx_dict, gos, zs):

    netactivity_scores = []
    for knockout in idx_dict:
        
        idx = idx_dict[knockout]
        mat = torch.from_numpy(adata.X[idx,:].todense()).to('cuda').double()

        if layer_name == 'fc1':
            colnames = gos
            na_score = model.fc1(mat).detach().cpu().numpy()

        elif layer_name == 'fc_mean':
            na_score = model.fc_mean(model.fc1(mat)).detach().cpu().numpy()
            colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))

        elif layer_name == 'fc_var':
            na_score = model.fc_var(model.fc1(mat)).detach().cpu().numpy()
            colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))

        elif layer_name == 'z':

            mu, var = model.encode(mat)
            z = model.reparametrize(mu, var).detach().cpu().numpy()
            na_score = z
            colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))

        elif layer_name == 'u':

            bc, csz = model.c_encode(torch.from_numpy(c).to('cuda:0'), temp=1)
            mu, var = model.encode(mat)
            z = model.reparametrize(mu, var)
            u = model.dag(z, bc, csz, bc, csz, num_interv=1).detach().cpu().numpy()

        ##
        na_score_df = pd.DataFrame(na_score)
        na_score_df.columns = colnames
        na_score_df['type'] = knockout
        netactivity_scores.append(na_score_df)

    df_netactivity_scores = pd.concat(netactivity_scores)
    return df_netactivity_scores

"""plot layer weights"""
def plot_layer_weights(layer_name):

    # read different trained models here
    fpath = os.path.join('./../../figures','uhler',f'{mode_type}_{trainmode}')
    savedir = f'./../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    ## get non-zero gradients
    gradients = eval(f'model.{layer_name}.weight.detach().cpu().numpy()')
   
    ## Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(gradients.flatten(), alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Layer {layer_name} weights')
    plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

"""analyze latent factors"""
def generate_latent_factor_matrix(na_activity_score, model_name, layer_name, seed, norm=False, var = 'absdm', save=False):

    #load activity scores
    fpath = os.path.join('./../../result','uhler',model_name, f'seed_{seed}')

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score['type'] == 'ctrl']

    ## init df
    ttest_df = []

    for knockout in set(na_activity_score['type']):
    
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[na_activity_score['type']  == knockout]

            for geneset in na_activity_score.columns[:-1]:

                ## abs(logFC)
                knockout_mean = knockout_cells.loc[:,geneset].values.mean() 
                ctrl_mean = ctrl_cells.loc[:,geneset].values.mean()

                if (not ctrl_mean) or (not knockout_mean):
                    diffmean, absdm,= 0,0
                else:
                    diffmean = ctrl_mean - knockout_mean
                    absdm = abs(diffmean)
            
                #append info
                ttest_df.append([knockout, geneset, diffmean, absdm])
                
    ## build df
    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout','geneset', 'diffmean', 'absdm']
    ttest_df = ttest_df.sort_values(by=['knockout','geneset']).reset_index(drop=True)

    ##regroup
    heatmap_data = ttest_df.pivot(index="knockout", columns="geneset", values=var).dropna(axis=1)
    if norm:
        heatmap_data = (heatmap_data.T/heatmap_data.max(axis=1)).T
    if save:
        heatmap_data.to_csv(os.path.join(fpath, f'activation_scores_{var}_DEA_layer_{layer_name}_{"norm" if norm else ""}_matrix.tsv'), sep='\t')
    
    return heatmap_data

"""analyze activation independence"""
def compute_differential_activation_independence(heatmap_data):
    latent_assignment = np.argmax(heatmap_data,axis=0)
    return len(set(latent_assignment)) / len(latent_assignment)


## load our model
mode_type = 'full_go'
results_l = {}

for trainmode in ['regular','sena_delta_0', 'sena_delta_1','sena_delta_3']:

    model_name = f'{mode_type}_{trainmode}'
    seed = 42
    layer_name = 'z'

    #load data
    dai_l = []
    model, adata, idx_dict, gos, zs = load_data(model_name, seed)

    ## get AS
    for _ in tqdm(range(1000)):
        na_activity_score = compute_activation_scores(layer_name, model, adata, idx_dict, gos, zs)
        heatmap_data = generate_latent_factor_matrix(na_activity_score, model_name = model_name, layer_name = layer_name, seed=seed, var='absdm', norm=True, save=False)
        dai_l.append(compute_differential_activation_independence(heatmap_data))

    ##compute dai
    dai_mean, dai_st = np.mean(dai_l), np.std(dai_l)
    results_l[trainmode] = f'{dai_mean} +- {dai_st}'
    
#build dataframe
results_df = pd.DataFrame(results_l, index = [0]).T
results_df.to_csv(os.path.join('./../../result/uhler/post_analysis/diff_act_independence/summary.tsv'),sep='\t')