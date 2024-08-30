import numpy as np
import sena_tools as st
import importlib
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
importlib.reload(st)
import os
import pickle
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42


def plot_mse_analysis(mode = '1layer', subsample = 'topgo'):

    def build_dataset():

        #mode
        variables = ['mode','epoch','test_mse','seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'ae_{arch}',f'autoencoder_{arch}_ablation_efficiency_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_mse[variables])
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    methods = ['sena_0', 'sena_1', 'sena_3', 'regular', 'regular_orig', 'l1_3', 'l1_5', 'l1_7']
    colors = sns.color_palette("Set2", len(methods))
    df = build_dataset()

    # Group by epoch and mode, then calculate the median and IQR for MSE
    grouped = df.groupby(['epoch', 'mode']).agg(
        test_mse_median=('test_mse', 'median'),
        Q1=('test_mse', lambda x: np.percentile(x, 25)),
        Q3=('test_mse', lambda x: np.percentile(x, 75))
    ).reset_index()

    # Calculate the IQR
    grouped['IQR'] = grouped['Q3'] - grouped['Q1']

    # Calculate the lower and upper bounds for the shadow
    grouped['lower_bound'] = grouped['Q1'] - 1.5 * grouped['IQR']
    grouped['upper_bound'] = grouped['Q3'] + 1.5 * grouped['IQR']

    # Set up the figure and axes
    plt.figure(figsize=(12, 8))

    # Loop through each mode to plot
    for method, color in zip(methods, colors):
        method_group = grouped[grouped['mode'] == method]
        
        plt.plot(method_group['epoch'], method_group['test_mse_median'], '-o', label=method.capitalize(), color=color, markersize=4)
        plt.fill_between(
            method_group['epoch'], 
            method_group['lower_bound'], 
            method_group['upper_bound'], 
            color=color, 
            alpha=0.2
        )

    # Set y-axis to log scale
    plt.yscale('log')

    # Add a title and labels
    plt.title('Comparison of test_mse Across Epochs: SENA vs Regular (Median & IQR)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Median test_mse (log scale)', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join('./../../figures','ablation_study',f'ae_all_ablation_1layer_test_mse_{subsample}.png'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_sparsity_analysis(mode = '1layer', subsample = 'topgo'):

    def build_dataset():

        #mode
        variables = ['mode', 'epoch', 'sparsity', 'seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'ae_{arch}',f'autoencoder_{arch}_ablation_efficiency_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse[variables])
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    methods = ['sena_0', 'sena_1', 'sena_3', 'regular_orig', 'l1_3', 'l1_5', 'l1_7']
    colors = sns.color_palette("Set2", len(methods))
    df = build_dataset()

    # Group by epoch and mode, then calculate the median and IQR for test_mse
    grouped = df.groupby(['epoch', 'mode']).agg(
        sparsity_median=('sparsity', 'median'),
        Q1=('sparsity', lambda x: np.percentile(x, 25)),
        Q3=('sparsity', lambda x: np.percentile(x, 75))
    ).reset_index()

    # Calculate the IQR
    grouped['IQR'] = grouped['Q3'] - grouped['Q1']

    # Calculate the lower and upper bounds for the shadow
    grouped['lower_bound'] = grouped['Q1'] - 1.5 * grouped['IQR']
    grouped['upper_bound'] = grouped['Q3'] + 1.5 * grouped['IQR']

    # Set up the figure and axes
    plt.figure(figsize=(12, 8))

    # Loop through each mode to plot
    for method, color in zip(methods, colors):
        method_group = grouped[grouped['mode'] == method]
        
        plt.plot(method_group['epoch'], method_group['sparsity_median'], '-o', label=method.capitalize(), color=color, markersize=4)
        plt.fill_between(
            method_group['epoch'], 
            method_group['lower_bound'], 
            method_group['upper_bound'], 
            color=color, 
            alpha=0.2
        )


    # Add a title and labels
    plt.title('Comparison of sparsity Across Epochs: SENA vs Regular (Median & IQR)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Median sparsity', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join('./../../figures','ablation_study',f'ae_all_ablation_1layer_sparsity_{subsample}.png'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

    """keep only last epoch"""
    last_epoch_df = df[df['epoch'] == df['epoch'].max()]

    grouped_sparsity = last_epoch_df.groupby('mode').agg(
        sparsity_mean=('sparsity', 'mean'),
        sparsity_stderr=('sparsity', lambda x: np.std(x) / np.sqrt(len(x)))
    ).reset_index().sort_values(by='sparsity_mean', ascending=False)

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Create a barplot with error bars for the sparsity
    sns.barplot(x='mode', y='sparsity_mean', data=grouped_sparsity, palette='Set2', ci=None)
    plt.errorbar(x=np.arange(len(grouped_sparsity['mode'])), 
                y=grouped_sparsity['sparsity_mean'], 
                yerr=grouped_sparsity['sparsity_stderr'], 
                fmt='none', 
                c='black', 
                capsize=5)

    # Set labels and title
    plt.xlabel('Mode', fontsize=14)
    plt.ylabel('Mean Sparsity', fontsize=14)
    plt.title(f'Sparsity by Mode at Epoch {df["epoch"].max()+1}', fontsize=16)

    plt.savefig(os.path.join('./../../figures','ablation_study',f'ae_all_ablation_1layer_sparsity_{subsample}_last_epoch.png'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_outlier_analysis(mode = '1layer', subsample = 'topgo', metric = 'z_diff'):

    def build_dataset():

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'ae_{arch}',f'autoencoder_{arch}_ablation_interpretability_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse)
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    methods = ['sena_0', 'sena_1', 'sena_3', 'regular','regular_orig', 'l1_3', 'l1_5', 'l1_7']
    colors = sns.color_palette("Set2", len(methods))

    df = build_dataset()

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
    plt.savefig(os.path.join('./../../figures','ablation_study',f'ae_all_ablation_1layer_{metric}_{subsample}.png'))
    plt.cla()
    plt.clf()
    plt.close()

def plot_latent_correlation(mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', subsample = 'topgo', epoch = 5):

    ## load data
    with open(os.path.join('./../../result/ablation_study',f'ae_{modeltype}',f'autoencoder_{modeltype}_ablation_{analysis}_{mode}_{subsample}.pickle'), 'rb') as handle:
        results = pickle.load(handle)

    #subset
    seed_0_df = results[0]
    subset_epoch = seed_0_df[seed_0_df['epoch']==epoch]
    
    # Melt the DataFrame to plot both input_zdiff and latent_zdiff in a single plot
    df_melted = subset_epoch.melt(value_vars=['input_zdiff', 'latent_zdiff'], var_name='Type', value_name='z_diff')

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(data=df_melted, x='Type', y='z_diff', palette='Set3', width=0.5)

    # Set titles and labels
    plt.title(f'Comparison of Input and Latent z_diff - Epoch {epoch}', fontsize=16)
    plt.xlabel('Type', fontsize=14)
    plt.ylabel('z_diff', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.savefig(os.path.join('./../../figures','ablation_study',f'ae_{modeltype}_ablation_{mode}_{analysis}_{epoch}_{subsample}.png'))
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    
    #compare sena vs regular
    #plot_outlier_analysis(metric = 'z_diff', subsample = 'topgo')
    #plot_outlier_analysis(metric = 'recall_at_100', subsample = 'topgo')

    #plot_outlier_analysis(metric = 'z_diff', subsample = 'raw')
    #plot_outlier_analysis(metric = 'recall_at_100', subsample = 'raw')

    #analyze single architecture (e.g. sena) between "mean of affected expression DE" and "latent space DE" at a specific epochs
    plot_latent_correlation(epoch=45, mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', subsample = 'topgo')

    #plot mse analysis
    #plot_mse_analysis(mode = '1layer', subsample = 'topgo')
    #plot_sparsity_analysis(mode = '1layer', subsample = 'topgo')
    pass