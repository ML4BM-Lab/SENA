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
seeds = [42, 13]

#load summary
summary_l = []
methods = ['sena_delta_0','regular'] #'sena_delta_1', 'sena_delta_3'
for method in methods:
    for seed in seeds:
        df = pd.read_csv(os.path.join('./../../../', 'result', 'uhler', f'{dataset}_{method}/seed_{seed}', f'uhler_{method}_summary.tsv'),sep='\t',index_col=0)
        df['seed'] = seed
        summary_l.append(df)
summary_df = pd.concat(summary_l)

#plot
plot_groupal_metric(summary_df, dataset, mode, metric='recall_at_100', methods = methods)
# plot_groupal_metric(summary_df, dataset, mode, metric='z_diff', methods = methods)
# plot_groupal_metric(summary_df, dataset, mode, metric='mmd_loss', methods = methods)
# plot_groupal_metric(summary_df, dataset, mode, metric='recon_loss', methods = methods)
# plot_groupal_metric(summary_df, dataset, mode, metric='kl_loss', methods = methods)
# plot_groupal_metric(summary_df, dataset, mode, metric='l1_loss', methods = methods)