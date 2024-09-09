import sys
sys.path.append('./../')
import numpy as np
import importlib
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
import os
import pickle
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42

def visualize_gradients(model_name = 'full_go'):

    # read different trained models here
    savedir = f'./../../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    fpath = os.path.join('./../../../','figures','uhler_paper',model_name)
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    def plot_layer_weights(layer_name, model, fpath):

        try:

            ## get non-zero gradients
            non_masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) != 0].detach().cpu().numpy()')
            masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) == 0].detach().cpu().numpy()')

            ## Plotting the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(non_masked_gradients, bins=30, alpha=0.5, label='Non-masked values')
            plt.hist(masked_gradients, bins=30, alpha=0.5, label='Masked values')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Layer {layer_name} weights')
            plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

        except:

            ##
            gradients = eval(f'model.{layer_name}.weight.detach().cpu().numpy().flatten()')

            ## Plotting the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(gradients, bins=30, alpha=0.5, label = f'layer {layer_name}', color='blue')
            plt.yscale('log')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Layer {layer_name} weights')
            plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

    def plot_weight_heatmap(layer_name, model, fpath):

        ##get the output of NetActivity Layer
        #batch_size, mode = 128, 'train'
        #dataloader, _, _, _, ptb_targets = get_data(batch_size=batch_size, mode=mode)
        adata, _, gos, zs = load_data_raw_go(ptb_targets)

        weight_mat = eval(f'model.{layer_name}.weight.detach().cpu().numpy()').T
        weight_df = pd.DataFrame(weight_mat)

        if layer_name == 'fc1':

            weight_df.index = adata.var['gene_symbols']
            weight_df.columns = gos
           
        elif layer_name == 'fc_mean' or layer_name == 'fc_var':# or layer_name == 'z':

            weight_df.index = gos
            weight_df.columns = zs

        ## Plotting the histogram
        plt.figure(figsize=(25, 150))
        sns.heatmap(weight_df.abs(), cmap='coolwarm', center=0, annot=False, linewidths=.5)
        plt.title(f'Heatmap of Gene Weights - Layer {layer_name}')
        plt.xlabel('GO Terms')
        plt.ylabel('Gene Symbols')
        plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_heatmap.png'))
        
    ## hist
    ##plot_layer_weights(layer_name='fc1', model=model, fpath=fpath)
    plot_layer_weights(layer_name='fc_mean', model=model, fpath=fpath)
    plot_layer_weights(layer_name='fc_var', model=model, fpath=fpath)

    ## heatmap
    #plot_weight_heatmap(layer_name='fc_mean', model=model, fpath=fpath)
    #plot_weight_heatmap(layer_name='fc_var', model=model, fpath=fpath)

def plot_groupal_metric(df, dataset, mode, metric='recall_at_100', methods = []):

    #define fpath
    fpath = os.path.join('./../../../figures', 'uhler', 'all_models')
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    #define colors
    colors = sns.color_palette("Set2", len(methods))

    #group across seeds
    grouped = df.groupby(['epoch', 'mode']).agg(
            metric_mean=(metric, 'mean'),
            metric_std=(metric, 'std')
            ).reset_index()

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Loop over each method and plot the corresponding data
    for method, color in zip(methods, colors):
        # Filter data for the current method
        grouped_method = grouped[grouped['mode'] == method]
        
        # Plot for the current method
        plt.plot(grouped_method['epoch'], grouped_method['metric_mean'], '-o', label=method, color=color)
        plt.fill_between(
            grouped_method['epoch'], 
            grouped_method['metric_mean'] - grouped_method['metric_std'], 
            grouped_method['metric_mean'] + grouped_method['metric_std'], 
            color=color, 
            alpha=0.2
        )

    # Add labels and title
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Metric Mean', fontsize=14)
    plt.title('Metric vs. Epoch for Different Methods', fontsize=16)
    if ('recall' not in metric) and ('z_diff' not in metric):
        plt.yscale('log')

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()
    plt.savefig(os.path.join(fpath, f'{dataset}_{mode}_analysis_{metric}.pdf'))
    plt.cla()
    plt.clf()
    plt.close() 

dataset = 'full_go'
mode = 'encoder'
seeds = [42]

#load summary
summary_l = []
tuplas = [('regular','seed_42_latdim_70'),
           ('regular','seed_42_latdim_35'),
           ('regular','seed_42_latdim_10'),
           ('regular','seed_42_latdim_5'), 
           ('sena_delta_1','seed_42_latdim_70'), 
           ('sena_delta_1','seed_42_latdim_35'),
           ('sena_delta_1','seed_42_latdim_10'),
           ('sena_delta_1','seed_42_latdim_5')]

methods = [x[0] + '_latdim' + x[1].split('latdim')[-1] for x in tuplas]

for tupla in tuplas:
    method = tupla[0]
    seed, latdim = tupla[1].split('_')[1], tupla[1].split('_')[-1]
    df = pd.read_csv(os.path.join('./../../../', 'result', 'uhler', f'{dataset}_{method}/seed_{seed}_latdim_{latdim}', f'uhler_{method}_summary.tsv'),sep='\t',index_col=0)
    df['seed'] = seed
    df['mode'] = df['mode'] + f'_latdim_{latdim}'
    summary_l.append(df)

summary_df = pd.concat(summary_l)

#plot
plot_groupal_metric(summary_df, dataset, mode, metric='recall_at_100', methods = methods)
plot_groupal_metric(summary_df, dataset, mode, metric='z_diff', methods = methods)
plot_groupal_metric(summary_df, dataset, mode, metric='mmd_loss', methods = methods)
plot_groupal_metric(summary_df, dataset, mode, metric='recon_loss', methods = methods)
plot_groupal_metric(summary_df, dataset, mode, metric='kl_loss', methods = methods)
plot_groupal_metric(summary_df, dataset, mode, metric='l1_loss', methods = methods)