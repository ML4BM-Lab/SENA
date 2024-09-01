import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pickle
import torch
import pandas as pd 
import os
import seaborn as sns
import graphical_models as gm
from tqdm import tqdm
from utils import get_data
from collections import defaultdict
from scipy.stats import ttest_ind
import numpy as np
from utils import get_data
import utils as ut

## load our model
layer_name = 'fc1'
mode_type = 'full_go'
trainmode = 'NA_NA'
model_name = f'{mode_type}_{trainmode}'

def compute_knockout_ranking(var = 'pval'):

    #load activity scores
    na_activity_score = pd.read_csv(os.path.join('./../../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv'),sep='\t',index_col=0)

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score['type'] == 'ctrl']

    ## init df
    ttest_df = []

    for knockout in tqdm(set(na_activity_score['type'])):
    
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[na_activity_score['type']  == knockout]

            for geneset in na_activity_score.columns[:-1]:

                #perform ttest
                _, p_value = ttest_ind(ctrl_cells.loc[:,geneset].values, knockout_cells.loc[:,geneset].values, equal_var=False)
                
                try:
                    fc = knockout_cells.loc[:,geneset].values.mean() / ctrl_cells.loc[:,geneset].values.mean()
                except:
                    fc = 1

                abslogfc = np.abs(np.log(np.abs(fc)))

                #append info
                ttest_df.append([knockout, geneset, p_value, abslogfc])

    ## build df
    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout','geneset','pval', 'logfc']

    ## load genesets-genes mapping
    db_gene_go_map = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
    gene_go_dict = defaultdict(list)

    for go,ens in tqdm(db_gene_go_map.values):
        gene_go_dict[ens].append(go)

    ensemble_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_dict = dict(zip(ensemble_genename_mapping.iloc[:,1], ensemble_genename_mapping.iloc[:,0]))

    ## total genesets
    #total_genesets = [x for x in na_activity_score.columns[:-1].values if na_activity_score[x].sum() != 0]
    total_genesets = na_activity_score.columns[:-1].values

    ## belonging
    ranking_analysis_knockouts = []

    
    ## analyze
    for knockout in tqdm(set(ttest_df['knockout'])):
        
        #get the pvalues
        knockout_pvals = ttest_df[ttest_df['knockout'] == knockout][['geneset',var]].sort_values(by=var, ascending=(var == 'pval')).reset_index(drop=True)

        #get knockout ensemble name
        knockout_ensembl = ensembl_genename_dict[knockout]    

        #check how many are in our 1428 gene sets
        belonging_genesets = [geneset for geneset in total_genesets if geneset in gene_go_dict[knockout_ensembl]]

        #compute the median ranking
        ranking = [knockout_pvals[knockout_pvals['geneset'] == geneset].index[0]+1 for geneset in belonging_genesets]

        median_ranking = np.median(ranking)
        minimum_ranking = np.min(ranking)
        random_ranking = np.random.choice(list(range(len(total_genesets))), len(belonging_genesets))
        random_median_ranking = np.median(random_ranking)
        random_minimum_ranking = np.min(random_ranking)

        ranking_analysis_knockouts.append([knockout, median_ranking, minimum_ranking, random_median_ranking, random_minimum_ranking, len(belonging_genesets)])

    ## build dataframe
    ranking_analysis_knockouts = pd.DataFrame(ranking_analysis_knockouts)
    ranking_analysis_knockouts.columns = ['knockout', 'med_rank_NA', 'min_rank_NA', 'med_rank_rand', 'min_rank_rand', 'num_genesets']
    return ranking_analysis_knockouts

"""
evaluate ranking of architecture vs random for knockout/ctrl cells
"""

def analyze_ranking(ranking_analysis_knockouts, var='pval', hist=False):

    #do t test
    #_, p_value = ttest_ind(ranking_analysis_knockouts["med_rank_NetActivity"], ranking_analysis_knockouts["med_rank_random"], equal_var=False)

    # Plot boxplot
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=ranking_analysis_knockouts[['min_rank_NA', 'med_rank_NA', 'min_rank_rand', 'med_rank_rand']], palette="Set2")
    plt.title(f'Ranking Analysis for Biologically Driven weights - Mode {var}')
    plt.ylabel('Ranking')
    plt.grid(True)
    plt.savefig(os.path.join('..','..','figures','uhler_paper', f'{mode_type}_{trainmode}', 'activation_scores', 'general_analysis', f'layer_{layer_name}_boxplot_ranking_analysis_{var}.png'))

    if hist:
        for mode in ['med','min']:

            # Plot histograms
            plt.figure(figsize=(12, 8))
            sns.histplot(ranking_analysis_knockouts[f'{mode}_rank_NA'], color='blue', label=f'{mode} Ranking NetActivity', alpha=0.6, bins = 15)
            sns.histplot(ranking_analysis_knockouts[f'{mode}_rank_rand'], color='red', label=f'{mode} Random Ranking', alpha=0.6, bins = 15)
            plt.title(f'Histogram of {mode} Ranking and Random Ranking', fontsize=16)
            plt.xlabel('Ranking', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join('..','..','figures','uhler_paper', f'{mode_type}_{trainmode}', 'activation_scores', 'general_analysis', f'layer_{layer_name}_histplot_ranking_analysis_{mode}.png'))

def analyze_best_and_worse_performers(ranking_analysis_knockouts):

    for mode in ['min','med']:
        # Sort the DataFrame by 'med_rank_NetActivity'
        df_sorted = ranking_analysis_knockouts.sort_values(by=f'{mode}_rank_NA')

        # Select the top and bottom 10 knockouts
        top_10 = df_sorted.head(10)
        bottom_10 = df_sorted.tail(10)

        # Combine top and bottom 10 into a single DataFrame
        combined = pd.concat([top_10, bottom_10]).reset_index(drop=True)
        xrange = list(range(0, int(combined[f'{mode}_rank_NA'].max()) + 1, int(combined[f'{mode}_rank_NA'].max()//5)))
        xrange[0] = 1

        # Plot bar plot
        plt.figure(figsize=(9, 9))
        scatter = plt.scatter(combined[f'{mode}_rank_NA'], combined['knockout'], c = combined['num_genesets'], cmap='viridis', s=150, edgecolor='k', alpha=0.7)
        plt.title(f'Top and Bottom 10 Knockouts by {mode} Net Activity Ranking', fontsize=16)
        plt.xlabel(f'{mode} Net Activity Ranking', fontsize=14)
        plt.ylabel('Knockout', fontsize=14)
        plt.xticks(xrange)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(scatter, label='Number of Gene Sets')
        plt.savefig(os.path.join('..', '..', 'figures', 'uhler_paper', f'{mode_type}_{trainmode}', 'activation_scores', 'general_analysis', f'layer_{layer_name}_barplot_ranking_analysis_top_perform_{mode}.png'))

## compute ranking
ranking_analysis_knockouts = compute_knockout_ranking(var = 'logfc')
ranking_analysis_knockouts.to_csv(os.path.join('./../../../','result', f'{mode_type}_{trainmode}',f'ranking_analysis_knockout_{layer_name}.tsv'),sep='\t')

## plot analysis
analyze_ranking(ranking_analysis_knockouts, var = 'logfc')
analyze_best_and_worse_performers(ranking_analysis_knockouts)


"""
plot results together
"""
def analyze_ranking_models_combined():

    #load models
    ranking_SENA_2 = pd.read_csv(os.path.join('./../../../','result', f'{mode_type}_NA_NA',f'ranking_analysis_knockout_{layer_name}.tsv'),sep='\t', index_col=0)
    ranking_SENA_delta = pd.read_csv(os.path.join('./../../../','result', f'{mode_type}_NA+deltas',f'ranking_analysis_knockout_{layer_name}.tsv'),sep='\t', index_col=0)
    ranking_analysis = pd.concat([ranking_SENA_2[['med_rank_NA', 'min_rank_NA']], ranking_SENA_delta[['med_rank_NA', 'min_rank_NA']], ranking_SENA_2[['med_rank_rand','min_rank_rand']]],axis=1)
    ranking_analysis.columns = ['med_rank_SENA-2', 'min_rank_SENA-2', 'med_rank_SENA-delta', 'min_rank_SENA-delta','med_rank_rand','min_rank_rand']

    # Plot boxplot
    plt.figure(figsize=(12, 12))
    sns.boxplot(data=ranking_analysis[ranking_analysis.mean().sort_values().index], palette="Set2")
    plt.title(f'Ranking Analysis for Biologically Driven weights')
    plt.ylabel('Ranking')
    plt.grid(True)
    plt.savefig(os.path.join('..','..','figures','uhler_paper', 'both_models', f'layer_{layer_name}_boxplot_ranking_analysis.png'))

    # Plot violinplot
    plt.figure(figsize=(12, 12))
    sns.violinplot(data=ranking_analysis[ranking_analysis.mean().sort_values().index], palette="Set2")
    plt.title(f'Ranking Analysis for Biologically Driven weights')
    plt.ylabel('Ranking')
    plt.grid(True)
    plt.savefig(os.path.join('..','..','figures','uhler_paper', 'both_models', f'layer_{layer_name}_violinplot_ranking_analysis.png'))

    # Plot histograms
    plt.figure(figsize=(12, 8))

    # plot histplot
    modes = ['med', 'min']
    for i, mode in enumerate(modes):

        ax = plt.subplot(len(modes), 1, i + 1)
        sns.histplot(ranking_analysis[f'{mode}_rank_SENA-2'], color='blue', label=f'{mode} Ranking SENA-2', alpha=0.6, bins = 40, ax=ax)
        sns.histplot(ranking_analysis[f'{mode}_rank_SENA-delta'], color='red', label=f'{mode} Ranking SENA-delta', alpha=0.6, bins = 40, ax=ax)
        sns.histplot(ranking_analysis[f'{mode}_rank_rand'], color='green', label=f'{mode} Ranking random', alpha=0.6, bins = 40, ax=ax)
        
        ax.set_title(f'Histogram of {mode} Ranking and Random Ranking', fontsize=16)
        ax.set_xlabel('Ranking', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join('..','..','figures','uhler_paper', 'both_models', f'layer_{layer_name}_hist_ranking_analysis.png'))
