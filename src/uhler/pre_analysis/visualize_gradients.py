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
import torch
import matplotlib.ticker as ticker
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42

def visualize_gradients(model_names = 'full_go_sena_delta_0', seed_and_lat = 'seed_13_latdim_105'):

    # Create a figure with subplots, one for each model_name, arranged in a single row
    fig, axs = plt.subplots(len(model_names), 2, figsize=(10, 10), gridspec_kw={'width_ratios': [4, 1]})  # Adjust width based on the number of models

    # Iterate over each model_name and plot its respective histogram in the corresponding subplot
    for i, model_name in enumerate(model_names):

        # Read different trained models here
        savedir = f'./../../result/uhler/{model_name}/{seed_and_lat}' 
        model = torch.load(f'{savedir}/best_model.pt')

        layer_name = 'fc1'
        fpath = os.path.join('./../../', 'figures', 'uhler', model_name)
        if not os.path.isdir(fpath):
            os.mkdir(fpath)

        # Get non-zero gradients
        non_masked_gradients =  eval(f'(model.{layer_name}.weight * model.{layer_name}.mask.T)[model.{layer_name}.mask.T == 1].detach().cpu().numpy()')
        masked_gradients =  eval(f'(model.{layer_name}.weight * model.{layer_name}.mask.T)[model.{layer_name}.mask.T != 1].detach().cpu().numpy()')

        #non_masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) != 0].detach().cpu().numpy()')
        #masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) == 0].detach().cpu().numpy()')

        ## Plotting the histogram
        if i == 0:
            axs[i][0].hist(non_masked_gradients, bins=30, label='Non-masked gradients', color='#ff7f0e')
            axs[i][0].hist(masked_gradients, bins=30, label='Masked gradients', color='#1f77b4')
        else:
            axs[i][0].hist(non_masked_gradients, bins=50, label='Non-masked gradients', color='#ff7f0e')
            axs[i][0].hist(masked_gradients, bins=50, label='Masked gradients', color='#1f77b4', alpha=0.8)

        axs[i][0].set_yscale('log')
        if i == 0:
            axs[i][0].legend(loc='upper left')
            axs[i][0].set_ylabel('Frequency', fontsize=20)
        axs[i][0].grid(True, linestyle='-', lw=2)
        axs[i][0].set_axisbelow(True)
        axs[i][0].tick_params(axis='x', labelsize=15)
        axs[i][0].tick_params(axis='y', labelsize=15)
        axs[i][0].tick_params(axis='both', width=2)

        # Get non-zero gradients
        non_masked_weights = eval(f'model.{layer_name}.mask.T[model.{layer_name}.mask.T != 1].detach().cpu().numpy()')
        masked_weights = eval(f'model.{layer_name}.mask.T[model.{layer_name}.mask.T == 1].detach().cpu().numpy()')

        # Count the occurrences of unique values
        non_masked_unique, non_masked_counts = np.unique(non_masked_weights, return_counts=True)
        masked_unique, masked_counts = np.unique(masked_weights, return_counts=True)

        # Plotting the histograms using bar plots to show only unique values
        axs[i][1].bar(non_masked_unique, non_masked_counts, label='Non-masked values')
        axs[i][1].bar(masked_unique, masked_counts, label='Masked values')
        axs[i][1].set_xticks(np.concatenate((non_masked_unique, masked_unique)))  # Ensure all unique values appear on the x-axis
        #axs[i][1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  # Set the format to 3 decimal places
        axs[i][1].set_yscale('log')
        axs[i][1].grid(True, linestyle='-', lw=2)
        axs[i][1].set_axisbelow(True)
        axs[i][1].tick_params(axis='x', labelsize=15)
        axs[i][1].tick_params(axis='y', labelsize=15)
        axs[i][1].tick_params(axis='both', width=2)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # Save the full figure with all histograms
    plt.savefig(os.path.join(f'./../../figures/uhler', 'final_figures', 'all_models_layer_fc1_histplot.pdf'))

    # Show the plot
    plt.show()
        

model_names = ['full_go_sena_delta_0', 'full_go_sena_delta_3', 'full_go_sena_delta_2', 'full_go_sena_delta_1']
visualize_gradients(model_names = model_names)