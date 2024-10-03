import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import torch
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from statannotations.Annotator import Annotator
import numpy as np

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

#
dataset = 'full_go'
mode = 'sena_delta_0'
model_name = f'{dataset}_{mode}'
seed = 42
latdim = 105
layer1 = 'fc1'
layer2 = 'fc_mean'

#load summary file
with open(os.path.join(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle'), 'rb') as handle:
    model_summary = pickle.load(handle)

#load model
savedir = f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}' 
model = torch.load(f'{savedir}/best_model.pt')

#activation layer
_, _, ptb_targets_all, ptb_targets_affected, gos, rel_dict, gene_go_dict, genename_ensemble_dict = utils.load_norman_2019_dataset()
fc1 = model_summary[layer1]
mean = model_summary[layer2]

#compute DA by geneset at the output of the SENA layer
DA_df_by_geneset_fc1 = compute_activation_df(fc1, scoretype = 'mu_diff', 
                                                gos=gos, mode=mode, 
                                                gene_go_dict=gene_go_dict, genename_ensemble_dict=genename_ensemble_dict,
                                                ptb_targets=ptb_targets_affected)

""" compute genesets mean"""
def compute_geneset_mean_weight():

    ##get affected idx
    affected_genesets = DA_df_by_geneset_fc1.loc[DA_df_by_geneset_fc1['affected'] == True,'geneset'].unique()
    nonaffected_genesets = DA_df_by_geneset_fc1.loc[DA_df_by_geneset_fc1['affected'] == False,'geneset'].unique()

    affected_genesets_idx = [i for i,x in enumerate(gos) if x in affected_genesets]
    nonaffected_genesets_idx = [i for i,x in enumerate(gos) if x in nonaffected_genesets]

    absmean_affected_weight = np.median(np.vstack([np.abs(model.fc_mean.weight.T[i,:].detach().cpu()) for i in affected_genesets_idx]),axis=0)
    absmean_nonaffected_weight = np.median(np.vstack([np.abs(model.fc_mean.weight.T[i,:].detach().cpu()) for i in nonaffected_genesets_idx]),axis=0)

    absmean_df = pd.DataFrame([absmean_affected_weight, absmean_nonaffected_weight]).T
    absmean_df.columns = ['affected','nonaffected']
    absmean_df_melt = pd.melt(absmean_df)

    # Adjusting the data to plot with seaborn
    plt.figure(figsize=(8, 10))
    sns.boxplot(x='variable',y='value',data=absmean_df_melt)

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join('./../../', 'figures', 'uhler', 'final_figures', f'DA_analysis_mean_weights_latentspace.pdf'))

""" compute geneset contribution """
def compute_geneset_contribution_latent_space():

    def compute_geneset_contribution_ls(mode):

        if mode == 'fc_mean':
            model_mode = model.fc_mean
        else:
            model_mode = model.fc_var
        ctrl_cells = fc1[fc1.index == 'ctrl'].to_numpy().mean(axis=0)
        contribution_list = []
        for knockout in tqdm(ptb_targets_affected):

            ##
            DA_df_by_geneset_fc1_knockout = DA_df_by_geneset_fc1[DA_df_by_geneset_fc1['knockout'] == knockout]
            affected_genesets = DA_df_by_geneset_fc1_knockout.loc[DA_df_by_geneset_fc1_knockout['affected'] == True, 'geneset'].unique()
            nonaffected_genesets = DA_df_by_geneset_fc1_knockout.loc[DA_df_by_geneset_fc1_knockout['affected'] == False, 'geneset'].unique()

            affected_genesets_idx = [i for i,x in enumerate(gos) if x in affected_genesets]
            nonaffected_genesets_idx = [i for i,x in enumerate(gos) if x in nonaffected_genesets]

            ##knockout
            knockout_exp = fc1[fc1.index == knockout].to_numpy().mean(axis=0)

            absmean_affected_contr = np.mean(np.vstack([model_mode.weight.T[i,:].detach().cpu() * ctrl_cells[i] for i in affected_genesets_idx]),axis=0)
            absmean_affected_knockout = np.median(np.vstack([model_mode.weight.T[i,:].detach().cpu() * knockout_exp[i] for i in affected_genesets_idx]),axis=0)

            absmean_nonaffected_contr = np.median(np.vstack([model_mode.weight.T[i,:].detach().cpu() * ctrl_cells[i] for i in nonaffected_genesets_idx]),axis=0)
            absmean_nonaffected_knockout = np.median(np.vstack([model_mode.weight.T[i,:].detach().cpu() * knockout_exp[i] for i in nonaffected_genesets_idx]),axis=0)

            score_affected = np.abs(absmean_affected_knockout - absmean_affected_contr)
            score_non_affected = np.abs(absmean_nonaffected_knockout - absmean_nonaffected_contr)


            ##build dataframe
            score_df_knockout = pd.DataFrame([score_affected, score_non_affected]).T
            score_df_knockout.columns = ['affected','non_affected']
            score_df_knockout['knockout'] = knockout
            contribution_list.append(score_df_knockout)
            
        score_df = pd.concat(contribution_list)
        score_df_melt = pd.melt(score_df, id_vars = 'knockout')
        score_df_melt['value'] = score_df_melt['value'].values
        score_df_melt['mode'] = mode
        return score_df_melt

    # mean & var
    df_mean = compute_geneset_contribution_ls('fc_mean')
    df_var = compute_geneset_contribution_ls('fc_var')
    df_latent_space = pd.concat([df_mean,df_var]).sort_values(by='variable',ascending=False)

    # Create pairs for statistical comparison
    sns.set(style='whitegrid')
    custom_palette = sns.color_palette("Set2")[:2]  
    pairs = [((layer, 'non_affected'), (layer, 'affected')) for layer in ['fc_mean','fc_var']]

    # Adjusting the data to plot with seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='mode',y='value',hue='variable',
                     data=df_latent_space,
                     fliersize=3,
                     palette=custom_palette,
                     linewidth=1.5,
                     whis=1.5,  # Adjust whisker length
                     boxprops=dict(edgecolor='black'),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     flierprops=dict(
                        marker='o',                # Marker style
                        markerfacecolor='black',     # Fill color of outliers
                        markeredgecolor='black',     # Edge color of outliers
                        markersize=2,              # Size of outlier markers
                        linestyle='none'           # No connecting lines
                     ),
                      order=['fc_mean','fc_var'])

    # Initialize the Annotator
    annotator = Annotator(
        ax,
        pairs,
        data=df_latent_space,
        x='mode',
        y='value',
        hue='variable'
    )

    # Configure and apply the statistical test
    annotator.configure(
        test='Mann-Whitney',
        text_format='star',
        loc='outside',
        comparisons_correction='Benjamini-Hochberg',
        show_test_name=False
    )

    annotator.apply_and_annotate()

    plt.yscale('log')
    plt.ylim(1e-8, 1)
    plt.tight_layout()
    plt.savefig(os.path.join('./../../', 'figures', 'uhler', 'final_figures', f'DA_analysis_latentspace_contribution.pdf'))


# first check if mean weights are differentially activated
#compute_geneset_mean_weight()

#then work with the contributions
compute_geneset_contribution_latent_space()
