import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def compute_activation_df(na_activity_score, scoretype, gos, mode, gene_go_dict, genename_ensemble_dict, ptb_targets):

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score.index == 'ctrl'].to_numpy()

    ## init df
    ttest_df = []

    for knockout in tqdm(ptb_targets):

        if knockout not in genename_ensemble_dict:
            continue
        
        #get knockout cells       
        knockout_cells = na_activity_score[na_activity_score.index == knockout].to_numpy()

        #compute affected genesets
        if mode[:4] == 'sena':
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[genename_ensemble_dict[knockout]]] 

        for i, geneset in enumerate(gos):
            
            if scoretype == 'mu_diff':
                score = abs(ctrl_cells[:,i].mean() - knockout_cells[:,i].mean())

            #append info
            if mode[:4] == 'sena':
                ttest_df.append([knockout, geneset, scoretype, score, geneset in belonging_genesets])
            elif mode[:7] == 'regular' or mode[:2] == 'l1':
                ttest_df.append([knockout, i, scoretype, score, i in belonging_genesets])

    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout', 'geneset', 'scoretype', 'score', 'affected']

    return ttest_df

df_list = []
dataset = 'full_go'
modes = ['sena_delta_0']#, 'sena_delta_2', 'sena_delta_3']
seed = 42
latdims = [105, 70, 35, 10, 5]

for mode in tqdm(modes):
    model_name = f'{dataset}_{mode}'
    for latdim in latdims:
        #load summary file
        with open(os.path.join(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle'), 'rb') as handle:
            model_summary = pickle.load(handle)

        #activation layer
        _, _, ptb_targets_all, ptb_targets_affected, gos, rel_dict, gene_go_dict, genename_ensemble_dict = utils.load_norman_2019_dataset()
        fc1 = model_summary['fc1']

        #compute DA by geneset at the output of the SENA layer
        DA_df_by_geneset = compute_activation_df(fc1, scoretype = 'mu_diff', gos=gos, mode=mode, 
                                                gene_go_dict=gene_go_dict, genename_ensemble_dict=genename_ensemble_dict,
                                                ptb_targets=ptb_targets_affected)

        DA_ratio_mean_df = DA_df_by_geneset.groupby(['knockout','affected'])['score'].mean().reset_index()
        pivot_df = DA_ratio_mean_df.pivot(index='knockout', columns='affected', values='score')
        DA_ratio_df = pd.DataFrame(pivot_df[True]/pivot_df[False])
        DA_ratio_df.columns = ['score']
        DA_ratio_df['latdim'] = latdim
        DA_ratio_df['mode'] = mode
        df_list.append(DA_ratio_df)

DAR_df = pd.concat(df_list)
DAR_df_sena_delta_0 = DAR_df[DAR_df['mode'] == 'sena_delta_0']


# Get the unique 'latdim' values and sort them in reverse order
latdim_order = sorted(DAR_df_sena_delta_0['latdim'].unique(), reverse=True)

# Convert 'latdim' to a categorical variable with the specified order
DAR_df_sena_delta_0['latdim'] = pd.Categorical(
    DAR_df_sena_delta_0['latdim'],
    categories=latdim_order,
    ordered=True
)

"""RATIO ANALYSIS"""
# Set the style and color palette
sns.set(style='whitegrid')  # Adds a grid to the plot

# Adjusting the data to plot with seaborn
plt.figure(figsize=(8, 10))
ax = plt.gca()

# Create a boxplot with customizations
sns.boxplot(
    x='latdim',
    y='score',
    data=DAR_df_sena_delta_0,
    fliersize=5,
    linewidth=2.5,
    boxprops=dict(edgecolor='#143b44', facecolor="#326881"),
    medianprops=dict(color='#143b44', linewidth=2),
    whiskerprops=dict(color='#143b44'),
    capprops=dict(color='#143b44')
)

# Add individual data points with a swarmplot
sns.swarmplot(
    x='latdim',
    y='score',
    data=DAR_df_sena_delta_0,
    color='black',
    size=5,
)


# Customize axes labels and title
plt.xlabel('Latent Dimension', fontsize=20)
plt.ylabel('DAR', fontsize=20)
plt.tick_params(axis='x', which='both', bottom=True, top=False, length=5, width=1, direction='out')
plt.tick_params(axis='y', which='both', bottom=True, top=False, length=5, width=1, direction='out')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Add grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join('./../../', 'figures', 'uhler', 'final_figures', 'DA_ratio_analysis.pdf'))

