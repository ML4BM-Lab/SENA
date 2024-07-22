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
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import ttest_ind
import numpy as np
from utils import get_data
import utils as ut

## load our model
mode_type = 'raw_go'
trainmode = 'NA+deltas'
model_name = f'{mode_type}_{trainmode}'

"""
plot layer weights
"""

def plot_layer_weights(layer_name):

    # read different trained models here
    fpath = os.path.join('./../../figures','uhler_paper',f'{mode_type}_{trainmode}')
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

#plot_layer_weights(layer_name = 'fc_mean')
#plot_layer_weights(layer_name = 'fc_var')


"""
analyze the latent factor relationship to perturbation
"""

def analyze_latent_factor_relationship(layer_name):

    #load activity scores
    fpath = os.path.join('./../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv')
    na_activity_score = pd.read_csv(fpath,sep='\t',index_col=0)

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score['type'] == 'ctrl']

    ## init df
    ttest_df = []

    for knockout in tqdm(set(na_activity_score['type'])):
    
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[na_activity_score['type']  == knockout]

            for geneset in na_activity_score.columns[:-1]:

                #perform ttest
                _, p_value = ttest_ind(ctrl_cells.loc[:,geneset].values, knockout_cells.loc[:,geneset].values, equal_var=False)
                
                #append info
                ttest_df.append([knockout, geneset, p_value])

    ## build df
    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout','geneset','pval']
    ttest_df = ttest_df.sort_values(by=['knockout','geneset']).reset_index(drop=True)

    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = ttest_df.pivot(index="knockout", columns="geneset", values="pval")
    heatmap_data = heatmap_data.dropna(axis=1)
    log_heatmap_data = -np.log10(heatmap_data)
    log_heatmap_data = (log_heatmap_data.T/log_heatmap_data.max(axis=1)).T
    #log_heatmap_data = log_heatmap_data/log_heatmap_data.max()
    log_heatmap_data.to_csv(os.path.join('./../../result',f'{mode_type}_{trainmode}', f'activation_scores_DEA_layer_{layer_name}_matrix.tsv'), sep='\t')

    # # Perform hierarchical clustering on the columns
    # reordered_rows = dendrogram(linkage(log_heatmap_data, method='ward', optimal_ordering=True), labels=log_heatmap_data.index, no_plot=True)['ivl']
    # reordered_columns = dendrogram(linkage(log_heatmap_data.T, method='ward', optimal_ordering=True), labels=log_heatmap_data.columns, no_plot=True)['ivl']
    
    # # Generate the heatmap
    # fpath_plots = os.path.join('./../../figures','uhler_paper',f'{mode_type}_{trainmode}')
    # plt.figure(figsize=(12, 12))
    # #sns.clustermap(log_heatmap_data, method="complete"), #row_colors=[cluster_colormap[i] for i in clusters])
    # sns.heatmap(log_heatmap_data.loc[reordered_rows, :], annot=False, cmap="inferno", cbar_kws={'label': 'p-value'})
    # plt.title("Heatmap of p-values for Knockout Genes and Genesets")
    # plt.xlabel("Geneset")
    # plt.ylabel("Knockout Gene")
    # plt.savefig(os.path.join(fpath_plots, f'activation_scores_layer_{layer_name}_heatmap.png'))

##
#analyze_latent_factor_relationship(layer_name = 'fc1')
analyze_latent_factor_relationship(layer_name = 'fc_mean')
analyze_latent_factor_relationship(layer_name = 'fc_var')
analyze_latent_factor_relationship(layer_name = 'z')
