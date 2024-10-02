import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Create figure and subplots (2 rows x 5 columns)
seed = 13
column_widths = [105, 70, 35, 10, 5]  # The number of columns for each heatmap
column_ratios = [65, 55, 35, 25, 20]
fig, axes = plt.subplots(2, 5, figsize=(25, 20), gridspec_kw={'width_ratios': column_ratios})
name_mapping = {'full_go_sena_delta_0': r'SENA$_{\lambda=0}$', 'full_go_regular': 'MLP'}

for i, model_name in enumerate(['full_go_sena_delta_0', 'full_go_regular']):
    for j, latdim in enumerate(column_widths):

        print(f"{model_name}: {latdim}")

        """Load the data"""
        with open(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle', 'rb') as handle:
            results_dict = pickle.load(handle)

        # Generate heatmaps in each subplot
        data = results_dict['bc_temp100']
        aspect_ratio = latdim / 105  # Calculate aspect ratio based on number of columns (for width adjustment)

        heatmap = axes[i, j].imshow(data, cmap='cividis', aspect='auto')

        # Show y-axis labels (gene names) only for the first column
        if i == 0 and j == 0:
            axes[i, j].set_yticks(range(data.shape[0]))
            axes[i, j].set_yticklabels(data.index.values, fontsize=6)  # Increase font size for better visibility
        else:
            axes[i, j].set_yticks([])


        # Add colorbar for the first row and second column
        if i == 0 and j == len(column_widths)-1:
            fig.colorbar(heatmap, ax=axes[i, j])

        # Add x-axis labels, proportional to the number of columns
        if j > 2:
            axes[i, j].set_xticks(range(data.shape[1]))  # Add some ticks, not too many
        #axes[i, j].set_xticklabels(list(range(data.shape[1])), fontsize=8)

        # Add titles for each subplot
        axes[i, j].set_title(f'{name_mapping[model_name]} - Latdim {latdim}', fontsize=12)

# Adjust layout for better fit
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join('./../../', 'figures', 'uhler', 'final_figures', 'intervention_latent_factor_mapping.pdf'))
