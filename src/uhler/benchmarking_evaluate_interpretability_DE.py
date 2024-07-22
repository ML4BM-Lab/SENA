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
mode_type = 'raw_go'
trainmode = 'NA+deltas'
model_name = f'{mode_type}_{trainmode}'

"""
analyze activity scores numerically
"""

#load activity scores
na_activity_score = pd.read_csv(os.path.join('./../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv'),sep='\t',index_col=0)

def analyze_activity_scores():

    ##analyze activation scores
    activation_mean = na_activity_score.iloc[:,:-1].mean(axis=0)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(activation_mean, bins=activation_mean.shape[0]//2, edgecolor='black')
    plt.title('Histogram of Mean Activation Scores')
    plt.yscale('log')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join('..','..','figures','uhler_paper','activation_scores', model_name, f'layer_{layer_name}_fc1_sparse_activation_scores_hist_mean.png'))

##plot mean of activation scores
analyze_activity_scores()

"""
build differential acitivation analysis dataframe
"""

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
            
            #append info
            ttest_df.append([knockout, geneset, p_value])

## build df
ttest_df = pd.DataFrame(ttest_df)
ttest_df.columns = ['knockout','geneset','pval']


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
median_ranking_knockouts = []

## analyze
for knockout in tqdm(set(ttest_df['knockout'])):
    
    #get the pvalues
    knockout_pvals = ttest_df[ttest_df['knockout'] == knockout][['geneset','pval']].sort_values(by='pval').reset_index(drop=True)
    #knockout_pvals_filtered = knockout_pvals.sort_values(by='pval').dropna().reset_index(drop=True)

    #get knockout ensemble name
    knockout_ensembl = ensembl_genename_dict[knockout]    

    #check how many are in our 1428 gene sets
    belonging_genesets = [geneset for geneset in total_genesets if geneset in gene_go_dict[knockout_ensembl]]

    #compute the median ranking
    ranking = [knockout_pvals[knockout_pvals['geneset'] == geneset].index[0]+1 for geneset in belonging_genesets]
    median_ranking = np.median(ranking)
    random_ranking = np.median(np.random.choice(list(range(len(total_genesets))), len(belonging_genesets)))

    median_ranking_knockouts.append([knockout, median_ranking, random_ranking, len(belonging_genesets)])

## build dataframe
median_ranking_knockouts = pd.DataFrame(median_ranking_knockouts)
median_ranking_knockouts.columns = ['knockout','med_rank_NetActivity', 'med_rank_random', 'num_genesets']

"""
evaluate ranking of architecture vs random for knockout/ctrl cells
"""

def analyze_ranking(median_ranking_knockouts):

    #do t test
    _, p_value = ttest_ind(median_ranking_knockouts["med_rank_NetActivity"], median_ranking_knockouts["med_rank_random"], equal_var=False)

    # Plot boxplot
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=median_ranking_knockouts[['med_rank_NetActivity', 'med_rank_random']], palette="Set2")
    plt.title(f'Median Ranking for Biologically Driven weights')
    plt.ylabel('Ranking')
    plt.grid(True)
    plt.savefig(os.path.join('..','..','figures','uhler_paper','activation_scores',model_name, f'layer_{layer_name}_boxplot_ranking_analysis.png'))

    # Plot histograms
    plt.figure(figsize=(12, 8))

    # Median ranking NetActivity histogram
    sns.histplot(median_ranking_knockouts['med_rank_NetActivity'], color='blue', label='Median Ranking NetActivity', alpha=0.6, bins = 15)

    # Random ranking histogram
    sns.histplot(median_ranking_knockouts['med_rank_random'], color='red', label='Random Ranking', alpha=0.6, bins = 15)

    plt.title('Histogram of Median Ranking and Random Ranking', fontsize=16)
    plt.xlabel('Ranking', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join('..','..','figures','uhler_paper','activation_scores', model_name, f'layer_{layer_name}_histplot_ranking_analysis.png'))

def analyze_best_and_worse_performers(median_ranking_knockouts):

    # Sort the DataFrame by 'med_rank_NetActivity'
    df_sorted = median_ranking_knockouts.sort_values(by='med_rank_NetActivity')

    # Select the top and bottom 10 knockouts
    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)

    # Combine top and bottom 10 into a single DataFrame
    combined = pd.concat([top_10, bottom_10]).reset_index(drop=True)
    xrange = list(range(0, int(combined['med_rank_NetActivity'].max()) + 1, int(combined['med_rank_NetActivity'].max()//5)))
    xrange[0] = 1

    # Plot bar plot
    plt.figure(figsize=(9, 9))
    scatter = plt.scatter(combined['med_rank_NetActivity'], combined['knockout'], c = combined['num_genesets'], cmap='viridis', s=150, edgecolor='k', alpha=0.7)
    plt.title('Top and Bottom 10 Knockouts by Median Net Activity Ranking', fontsize=16)
    plt.xlabel('Median Net Activity Ranking', fontsize=14)
    plt.ylabel('Knockout', fontsize=14)
    plt.xticks(xrange)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add colorbar legend
    plt.colorbar(scatter, label='Number of Gene Sets')
    plt.savefig(os.path.join('..','..','figures','uhler_paper','activation_scores', model_name, f'layer_{layer_name}_barplot_ranking_analysis_top_perform.png'))

## 
analyze_ranking(median_ranking_knockouts)
analyze_best_and_worse_performers(median_ranking_knockouts)