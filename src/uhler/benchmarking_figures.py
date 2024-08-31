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

def plot_metric(df, dataset, mode, metric='recall_at_100', methods = []):

    #define fpath
    fpath = os.path.join('./../../figures', 'uhler_paper', f'{dataset}_{mode}','layer_analysis')
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

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()
    plt.savefig(os.path.join(fpath, f'analysis_{metric}.png'))
    plt.cla()
    plt.clf()
    plt.close() 

dataset = 'full_go'
mode = 'regular'
seed = 42
name = f'{dataset}_{mode}/seed_{seed}'

#load summary
summary_df = pd.read_csv(os.path.join('./../../', 'result', 'uhler', name, f'uhler_{mode}_summary.tsv'),sep='\t',index_col=0)

#plot
plot_metric(summary_df, dataset, mode, metric='mmd_loss', methods = ['regular'])