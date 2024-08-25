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


def plot_ranking_pvalue():

    grouped = sena_layer1.groupby('epoch').agg(
        aggregated_pval=('p-val', lambda pvals: combine_pvalues(pvals)[1])
    ).reset_index()

    # Plot the aggregated p-values against epochs on a logarithmic scale
    plt.plot(grouped['epoch'], grouped['aggregated_pval'], '-o', label='Aggregated p-value')

    # Set y-axis to log scale
    plt.yscale('log')

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Aggregated p-value (log scale)')
    plt.title('Aggregated p-value vs. Epoch (Fisher\'s method)')

    # Show legend
    plt.legend()

    # Show the plot
    plt.savefig(os.path.join('./../../figures','ablation_study','ae_SENA','sena_ablation_1layer_pvalue.png'))
    plt.cla()
    plt.clf()
    plt.close()

def plot_mean_recall():

    # Group by epoch and calculate necessary statistics
    grouped = sena_layer1.groupby('epoch').agg(
        mean_recall_mean=('mean_recall_at_100', 'mean'),
        mean_recall_std=('mean_recall_at_100', 'std'),
        mean_recall_direct_mean=('mean_recall_at_100_direct', 'mean'),
        mean_recall_direct_std=('mean_recall_at_100_direct', 'std'),
        mse_mean=('mse', 'mean'),
        mse_std = ('mse', 'std')
    ).reset_index()

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Plot mean_recall_at_100 with shaded standard deviation
    line1, = ax1.plot(grouped['epoch'], grouped['mean_recall_mean'], '-o', label='Mean Recall at 100', color='#98984d')
    ax1.fill_between(
        grouped['epoch'], 
        grouped['mean_recall_mean'] - grouped['mean_recall_std'], 
        grouped['mean_recall_mean'] + grouped['mean_recall_std'], 
        color='gray', 
        alpha=0.2, 
        label='Standard Deviation'
    )

    # Plot mean_recall_at_100_direct
    line2, = ax1.plot(grouped['epoch'], grouped['mean_recall_direct_mean'], '-o', label='Mean Recall at 100 Direct', color='#b3669e')
    ax1.fill_between(
        grouped['epoch'], 
        grouped['mean_recall_direct_mean'] - grouped['mean_recall_direct_std'], 
        grouped['mean_recall_direct_mean'] + grouped['mean_recall_direct_std'], 
        color='gray', 
        alpha=0.2, 
        label='Standard Deviation'
    )

    # Set labels for the first y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Recall at 100')
    ax1.set_title('Mean Recall and MSE vs. Epoch')
    ax1.set_ylim(0, 1)

    # Create a second y-axis for mse
    ax2 = ax1.twinx()
    line3, = ax2.plot(grouped['epoch'], grouped['mse_mean'], '--', label='MSE', color='red')
    ax2.fill_between(
        grouped['epoch'], 
        grouped['mse_mean'] - grouped['mse_std'], 
        grouped['mse_mean'] + grouped['mse_std'], 
        color='gray', 
        alpha=0.2, 
        label='Standard Deviation'
    )

    # Set label for the second y-axis
    ax2.set_ylabel('MSE')
    ax2.set_ylim(-0.05, 1)

    # Combine legends from both axes into one
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Save the plot
    plt.savefig(os.path.join('./../../figures','ablation_study','ae_SENA','sena_ablation_1layer_mean_recall_and_mse.png'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_outlier_analysis():

    def build_dataset():

        #mode
        mode = f'1layer'
        sena_outliers = pd.read_csv(os.path.join('./../../result','ablation_study','ae_sena',f'autoencoder_sena_ablation_outlier_{mode}.tsv'),sep='\t',index_col=0)
        regular_outliers = pd.read_csv(os.path.join('./../../result','ablation_study','ae_regular',f'autoencoder_regular_ablation_outlier_{mode}.tsv'),sep='\t',index_col=0)
        df = pd.concat([sena_outliers, regular_outliers])
        return df

    df = build_dataset()

    grouped = df.groupby(['epoch', 'mode']).agg(
            z_diff_mean=('z_diff', 'mean'),
            z_diff_std=('z_diff', 'std')
            ).reset_index()

    # Filter data for 'sena' and 'regular' modes
    grouped_sena = grouped[grouped['mode'] == 'sena']
    grouped_regular = grouped[grouped['mode'] == 'regular']

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot for 'sena' mode
    plt.plot(grouped_sena['epoch'], grouped_sena['z_diff_mean'], '-o', label='SENA', color='blue')
    plt.fill_between(
        grouped_sena['epoch'], 
        grouped_sena['z_diff_mean'] - grouped_sena['z_diff_std'], 
        grouped_sena['z_diff_mean'] + grouped_sena['z_diff_std'], 
        color='blue', 
        alpha=0.2
    )

    # Plot for 'regular' mode
    plt.plot(grouped_regular['epoch'], grouped_regular['z_diff_mean'], '-o', label='Regular', color='green')
    plt.fill_between(
        grouped_regular['epoch'], 
        grouped_regular['z_diff_mean'] - grouped_regular['z_diff_std'], 
        grouped_regular['z_diff_mean'] + grouped_regular['z_diff_std'], 
        color='green', 
        alpha=0.2
    )

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Mean z_diff')
    plt.title('Mean z_diff vs. Epoch with Standard Deviation for SENA and Regular')

    # Show legend
    plt.legend()
    plt.savefig(os.path.join('./../../figures','ablation_study','sena_regular_ablation_1layer_outliers.png'))
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plot_outlier_analysis()