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


def plot_mse_analysis(mode = '1layer', methods = [], subsample = 'topgo', structure='ae', metric='test_mse', plot_type = 'std'):

    def build_dataset():

        #mode
        variables = ['mode','epoch',f'{metric}','seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_efficiency_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_mse[variables])
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    colors = sns.color_palette("Set2", len(methods))
    df = build_dataset()

    # Group by epoch and mode, then calculate the median and IQR for MSE
    if plot_type == 'quantile':
        grouped = df.groupby(['epoch', 'mode']).agg(
            metric_median=(f'{metric}', 'median'),
            Q1=(f'{metric}', lambda x: np.percentile(x, 25)),
            Q3=(f'{metric}', lambda x: np.percentile(x, 75))
        ).reset_index()

        #Calculate the IQR
        grouped['IQR'] = grouped['Q3'] - grouped['Q1']

        # Calculate the lower and upper bounds for the shadow
        grouped['lower_bound'] = grouped['Q1'] - 1.5 * grouped['IQR']
        grouped['upper_bound'] = grouped['Q3'] + 1.5 * grouped['IQR']

    else:
        grouped = df.groupby(['epoch', 'mode']).agg(
                metric_mean=(metric, 'mean'),
                metric_std=(metric, 'std')
        ).reset_index()

    # Set up the figure and axes
    plt.figure(figsize=(12, 8))

    # Loop through each mode to plot
    for method, color in zip(methods, colors):

        method_group = grouped[grouped['mode'] == method]
        
        if plot_type == 'std':
            plt.plot(method_group['epoch'], method_group['metric_mean'], '-o', label=method.capitalize(), color=color, markersize=4)
            plt.fill_between(
                method_group['epoch'], 
                method_group['metric_mean'] - method_group['metric_std']/2, 
                method_group['metric_mean'] + method_group['metric_std']/2, 
                color=color, 
                alpha=0.2
            )
        else:
            plt.plot(method_group['epoch'], method_group['metric_median'], '-o', label=method.capitalize(), color=color, markersize=4)
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
    plt.title(f'Comparison of {metric} Across Epochs: SENA vs Regular (Median & IQR)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'Median {metric} (log scale)', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_{mode}_{metric}_{subsample}.pdf'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_sparsity_analysis(mode = '1layer', methods = [], subsample = 'topgo', structure='ae'):

    def build_dataset():

        #mode
        variables = ['mode', 'epoch', 'sparsity', 'seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_efficiency_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse[variables])
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    colors = sns.color_palette("Set2", len(methods))
    color_mapping = dict(zip(methods, colors))
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
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_{mode}_sparsity_{subsample}.pdf'))

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
    sns.barplot(x='mode', y='sparsity_mean', data=grouped_sparsity, palette=color_mapping, ci=None)
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

    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_1layer_sparsity_{subsample}_last_epoch.pdf'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_outlier_analysis(mode = '1layer', subsample = 'topgo', methods = [], name = '', metric = 'z_diff', structure='ae'):

    def build_dataset():

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_interpretability_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse)
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset

    if name == '':
        name = 'all'

    if not len(methods):
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
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_{name}_ablation_{mode}_{metric}_{subsample}.pdf'))
    plt.cla()
    plt.clf()
    plt.close()

def plot_latent_correlation(mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', subsample = 'topgo', epoch = 5, structure='ae'):

    ## load data
    with open(os.path.join('./../../result/ablation_study',f'{structure}_{modeltype}',f'{"autoencoder" if structure=="ae" else "vae"}_{modeltype}_ablation_{analysis}_{mode}_{subsample}.pickle'), 'rb') as handle:
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
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_{modeltype}_ablation_{mode}_{analysis}_{epoch}_{subsample}.pdf'))
    plt.cla()
    plt.clf()
    plt.close()

def compute_recall_metrics(mode = '1layer', methods = [], subsample = 'topgo', metric = 'recall_at_25', structure='ae'):

    def build_dataset():

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_interpretability_{mode}_{subsample}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse)
        
        df = pd.concat(arch_l)
        return df

    df = build_dataset()

    grouped = df.groupby(['epoch', 'mode']).agg(
            metric_mean=(metric, 'mean'),
            metric_std=(metric, 'std')
            ).reset_index()
    
    subset_lepoch = grouped[grouped['epoch'] == grouped['epoch'].max()].reset_index(drop=True)
    for i in range(subset_lepoch.shape[0]):
        print(f'{subset_lepoch["mode"].iloc[i]}, {metric}: {subset_lepoch["metric_mean"].iloc[i]} +- {subset_lepoch["metric_std"].iloc[i]}')

"""AE"""
def _call_ae():

    """1layer""" #compare sena vs regular
    
    methods = ['sena_0', 'sena_1', 'sena_3', 'regular_orig', 'l1_3', 'l1_5', 'l1_7']
    plot_mse_analysis(mode = '1layer', methods = methods, subsample = 'topgo')
    plot_sparsity_analysis(mode = '1layer', methods=methods, subsample = 'topgo')

    methods = ['sena_0', 'sena_1', 'sena_3']
    plot_outlier_analysis(mode='1layer', metric = 'recall_at_25', methods=methods, subsample = 'topgo')
    plot_outlier_analysis(mode='1layer', metric = 'recall_at_100', methods=methods, subsample = 'topgo')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_100', methods = methods, subsample = 'topgo')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_25', methods = methods, subsample = 'topgo')


    """2layer""" #sena-delta
    
    methods = ['sena_delta_0', 'sena_delta_1','sena_delta_3', 'regular_orig', 'l1_3', 'l1_5', 'l1_7']
    #plot_mse_analysis(mode = '1layer', methods = methods, subsample = 'topgo')
    #plot_sparsity_analysis(mode = '1layer', methods=methods, subsample = 'topgo')

    methods = ['sena_delta_0', 'sena_delta_1','sena_delta_3']
    plot_outlier_analysis(mode='2layer', metric = 'z_diff', methods = methods, name = 'all', subsample = 'topgo')
    plot_outlier_analysis(mode='2layer', metric = 'recall_at_100', methods = methods, name = 'all', subsample = 'topgo')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_100', methods = methods, subsample = 'topgo')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_25', methods = methods, subsample = 'topgo')

    #analyze single architecture (e.g. sena) between "mean of affected expression DE" and "latent space DE" at a specific epochs
    #plot_latent_correlation(epoch=45, mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', subsample = 'topgo')

def _call_vae():

    """single layer"""
    #compare sena vs regular
    methods = ['regular', 'sena_0','sena_1','sena_3', 'l1_3','l1_5','l1_7']
    plot_mse_analysis(mode = '1layer', methods = methods, subsample = 'topgo', structure='vae', metric='test_mse')
    plot_mse_analysis(mode = '1layer', methods = methods, subsample = 'topgo', structure='vae', metric='test_KL')

    methods = ['sena_0','sena_1','sena_3']
    plot_outlier_analysis(mode='1layer', metric = 'recall_at_25', methods=methods, subsample = 'topgo', structure='vae')
    plot_outlier_analysis(mode='1layer', metric = 'recall_at_100', methods=methods, subsample = 'topgo', structure='vae')

    """two layers layer"""
    methods = ['regular', 'sena_delta_0','sena_delta_1','sena_delta_3', 'l1_3', 'l1_5', 'l1_7']
    plot_mse_analysis(mode = '2layer', methods = methods, subsample = 'topgo', structure='vae', metric='test_mse')
    plot_mse_analysis(mode = '2layer', methods = methods, subsample = 'topgo', structure='vae', metric='test_KL')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_100', methods=methods, subsample = 'topgo', structure='vae')
    #compute_recall_metrics(mode='2layer', metric = 'recall_at_25', methods=methods, subsample = 'topgo', , structure='vae')

    methods = ['sena_delta_0','sena_delta_1','sena_delta_3']
    plot_outlier_analysis(mode='2layer', metric = 'recall_at_25', methods=methods, subsample = 'topgo', structure='vae')
    plot_outlier_analysis(mode='2layer', metric = 'recall_at_100', methods=methods, subsample = 'topgo', structure='vae')

    

if __name__ == '__main__':
    
    #_call_ae()
    _call_vae()