import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from multiprocessing import Pool, cpu_count
from scipy.stats import ttest_ind, ttest_rel
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
import importlib
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import combine_pvalues
import scipy.stats as stats
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample
import math as m

"""dataset"""
def load_norman_2019_dataset():

    def build_gene_go_relationships(adata):

        ## get genes
        genes = adata.var.index.values
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        GO_to_ensembl_id_assignment.columns = ['GO_id','ensembl_id']
        gos = sorted(set(pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')['PathwayID'].values.tolist()))

        go_dict, gen_dict = dict(zip(gos, range(len(gos)))), dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)

        gene_set = set(genes)
        go_set = set(gos)

        for go, gen in tqdm(zip(GO_to_ensembl_id_assignment['GO_id'], GO_to_ensembl_id_assignment['ensembl_id']), total = GO_to_ensembl_id_assignment.shape[0]):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return gos, rel_dict

    def load_gene_go_assignments():

        #filter genes that are not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        GO_to_ensembl_id_assignment.columns = ['GO_id','ensembl_id']
        gos = sorted(set(pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')['PathwayID'].values.tolist()))
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(gos)]

        return GO_to_ensembl_id_assignment

    def compute_affecting_perturbations(GO_to_ensembl_id_assignment):

        #filter interventions that not in any GO
        ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
        ensembl_genename_mapping = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
        intervention_genenames = map(lambda x: ensembl_genename_mapping.get(x,None), GO_to_ensembl_id_assignment['ensembl_id'])
        ptb_targets = list(set(intervention_genenames).intersection(set([x for x in adata.obs['guide_ids'] if x != '' and ',' not in x])))

        return ptb_targets

    #define fpath
    fpath = './../../data/Norman2019_raw.h5ad'

    #load adata, keep only ctrl + single interventions
    adata = sc.read_h5ad(fpath)
    adata = adata[(~adata.obs['guide_ids'].str.contains(','))]
    
    #drop genes that are not in any GO
    GO_to_ensembl_id_assignment = load_gene_go_assignments()
    adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]

    #build gene-go rel
    gos, rel_dict = build_gene_go_relationships(adata)

    #compute perturbations with at least 1 gene set
    ptb_targets = compute_affecting_perturbations(GO_to_ensembl_id_assignment) 
    
    #keep only perturbations affecting at least one gene set
    adata = adata[adata.obs['guide_ids'].isin(ptb_targets+[''])]

    #ctrl + ko
    ctrl_samples = adata[adata.obs['guide_ids']=='']
    ko_samples = adata[adata.obs['guide_ids']!='']

    return adata, ctrl_samples, ko_samples, ptb_targets, gos, rel_dict

"""analysis"""

def compute_layer_weight_contribution(model, adata, knockout, ens_knockout, num_affected, mode = 'sena'):

    #get knockout id
    genes = adata.var_names.values
    if ens_knockout in genes:
        idx = np.where(genes == ens_knockout)[0][0]
    else: 
        idx = 0
    
    #get layer matrix
    W = model.encoder.weight[:,idx].detach().cpu().numpy()

    if mode == 'sena':
        bias = 0
    elif mode == 'regular':
        bias = model.encoder.bias.detach().cpu().numpy()

    #get contribution to knockout gene from ctrl to each of the activation (gene input * weight)
    ctrl_exp_mean = adata[adata.obs['guide_ids'] == ''].X[:,idx].todense().mean()
    knockout_exp_mean = adata[adata.obs['guide_ids'] == knockout].X[:,idx].todense().mean()

    #compute contribution
    ctrl_contribution = W * ctrl_exp_mean + bias
    knockout_contribution = W * knockout_exp_mean + bias

    #now compute difference of means
    diff_vector = np.abs(ctrl_contribution - knockout_contribution)
    top_idxs = np.argsort(diff_vector)[::-1]

    return list(top_idxs[:num_affected])

def compute_activation_df(model, adata, gos, scoretype = 'ttest', mode = 'sena'):

    def build_activity_score_df():

        na_activity_score = {}
        intervention_types = list(adata.obs['guide_ids'].values.unique())
   
        for int_type in intervention_types:
            
            obs = adata[adata.obs['guide_ids'] == int_type].X.todense()
            int_df = pd.DataFrame(model.encoder(torch.tensor(obs).float().to('cuda')).detach().cpu().numpy())
            new_int_type = int_type if int_type != '' else 'ctrl'
            na_activity_score[new_int_type] = int_df.to_numpy()

        return na_activity_score

    #build ac
    na_activity_score = build_activity_score_df()

    ## load genesets-genes mapping
    db_gene_go_map = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
    gene_go_dict = defaultdict(list)

    for go,ens in db_gene_go_map.values:
        gene_go_dict[ens].append(go)

    ##build geneset mapping
    ensemble_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_dict = dict(zip(ensemble_genename_mapping.iloc[:,1], ensemble_genename_mapping.iloc[:,0]))

    ## define control cells
    ctrl_cells = na_activity_score['ctrl']

    ## init df
    ttest_df = []

    for knockout in na_activity_score.keys():
        
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[knockout]

            #compute affected genesets
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[ensembl_genename_dict[knockout]]]
            # if mode == 'regular':
            belonging_genesets = compute_layer_weight_contribution(model, adata, knockout, ensembl_genename_dict[knockout], len(belonging_genesets)) 

            for i, geneset in enumerate(gos):
                
                #perform ttest
                if scoretype == 'ttest':
                    _, p_value = ttest_ind(ctrl_cells[:,i], knockout_cells[:,i], equal_var=False)
                    score = -1 * m.log10(p_value)

                elif scoretype == 'mu_diff':
                    score = abs(ctrl_cells[:,i].mean() - knockout_cells[:,i].mean())
                    
                #append info
                if mode == 'sena':
                    ttest_df.append([knockout, geneset, scoretype, score, i in belonging_genesets])
                elif mode == 'regular':
                    ttest_df.append([knockout, i, scoretype, score, i in belonging_genesets])

    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout', 'geneset', 'scoretype', 'score', 'affected']

    #apply min-max norm
    if scoretype == 'mu_diff':
        for knockout in ttest_df['knockout'].unique():
            ttest_df.loc[ttest_df['knockout'] == knockout, 'score'] = MinMaxScaler().fit_transform(ttest_df.loc[ttest_df['knockout'] == knockout, 'score'].values.reshape(-1,1))

    return ttest_df

def compute_knockout_ranking(model, adata, gos, var = 'pval'):

    def build_activity_score_df():

        intervention_types = list(adata.obs['guide_ids'].values.unique())
        outs = []
        for int_type in intervention_types:
            
            obs = adata[adata.obs['guide_ids'] == int_type].X.todense()
            int_df = pd.DataFrame(model.encoder(torch.tensor(obs).float().to('cuda')).detach().cpu().numpy())
            int_df['type'] = int_type if int_type != '' else 'ctrl'
            outs.append(int_df)
        
        na_activity_score = pd.concat(outs)
        na_activity_score.columns = gos + ['type']
        return na_activity_score

    def build_knockout_statistics(ttest_df):

        ## belonging
        ranking_analysis_knockouts = []
        top_100_recall = []
        top_25_recall = []

        ## analyze
        for knockout in set(ttest_df['knockout']):
            
            #check how many are in our 1428 gene sets
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[ensembl_genename_dict[knockout]]]
            assert belonging_genesets != []

            #get the pvalues
            knockout_pvals = ttest_df[ttest_df['knockout'] == knockout][['geneset','pval']].sort_values(by='pval', ascending=True).reset_index(drop=True)
            knockout_pvals['affected'] = knockout_pvals['geneset'].isin(belonging_genesets).apply(lambda x: int(x))

            #compute the median ranking
            ranking = [knockout_pvals[knockout_pvals['geneset'] == geneset].index[0]+1 for geneset in belonging_genesets]

            median_ranking = np.median(ranking)
            minimum_ranking = np.min(ranking)
            random_ranking = np.random.choice(list(range(len(gos))), len(belonging_genesets))
            random_median_ranking = np.median(random_ranking)
            random_minimum_ranking = np.min(random_ranking)

            ranking_analysis_knockouts.append([knockout, median_ranking, minimum_ranking, random_median_ranking, random_minimum_ranking, len(belonging_genesets)])
            top_100_recall.append([knockout, knockout_pvals['affected'].iloc[:100].sum() / len(belonging_genesets)])
            top_25_recall.append([knockout, knockout_pvals['affected'].iloc[:25].sum() / len(belonging_genesets)])

        # build dataframe
        ranking_analysis_knockouts = pd.DataFrame(ranking_analysis_knockouts)
        ranking_analysis_knockouts.columns = ['knockout', 'med_rank_NA', 'min_rank_NA', 'med_rank_rand', 'min_rank_rand', 'num_genesets']
        _, p_value = ttest_ind(ranking_analysis_knockouts['med_rank_NA'], ranking_analysis_knockouts['med_rank_rand'], equal_var=False)

        #top recall
        top_100_recall_knockouts = pd.DataFrame(top_100_recall)
        top_100_recall_knockouts.columns = ['knockout','top_100_recall']
        top_25_recall_knockouts = pd.DataFrame(top_25_recall)
        top_25_recall_knockouts.columns = ['knockout','top_25_recall']

        return top_25_recall_knockouts, top_100_recall_knockouts, p_value

    #build ac
    na_activity_score = build_activity_score_df()

    ## load genesets-genes mapping
    db_gene_go_map = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
    gene_go_dict = defaultdict(list)

    for go,ens in tqdm(db_gene_go_map.values):
        gene_go_dict[ens].append(go)

    ##build geneset mapping
    ensemble_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_dict = dict(zip(ensemble_genename_mapping.iloc[:,1], ensemble_genename_mapping.iloc[:,0]))
    ensembl_genename_revdict = dict(zip(ensemble_genename_mapping.iloc[:,0], ensemble_genename_mapping.iloc[:,1]))

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score['type'] == 'ctrl']

    ## init df
    ttest_df = []

    for knockout in set(na_activity_score['type']):
        
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[na_activity_score['type']  == knockout]
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[ensembl_genename_dict[knockout]]]

            for geneset in gos:

                #perform ttest
                _, p_value = ttest_ind(ctrl_cells[geneset], knockout_cells[geneset], equal_var=False)

                #append info
                ttest_df.append([knockout, geneset, p_value, geneset in belonging_genesets])

    ## build df
    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout', 'geneset', 'pval', 'affected']

    ##
    top_25_recall_knockouts, top_100_recall_knockouts, p_value = build_knockout_statistics(ttest_df)

    ## get only knockouts that are included in the input gene sets
    knockouts = ttest_df['knockout'].unique()
    knockouts_ens = list(map(lambda x: ensembl_genename_dict[x], knockouts))
    knockouts_in = [ensembl_genename_revdict[x] for x in knockouts_ens if x in adata.var_names]
    ttest_df_direct = ttest_df[ttest_df['knockout'].isin(knockouts_in)].reset_index(drop=True).copy()
    top_25_recall_knockouts_direct, top_100_recall_knockouts_direct, p_value_direct = build_knockout_statistics(ttest_df_direct)

    #build summary
    summary_analysis = {
                        'pvalue': p_value, 
                        'pvalue_direct': p_value_direct,
                        'mean_recall_at_100': top_100_recall_knockouts.iloc[:,1].mean(), 
                        'mean_recall_at_100_direct': top_100_recall_knockouts_direct.iloc[:,1].mean(), 
                        'mean_recall_at_25': top_25_recall_knockouts.iloc[:,1].mean(),
                        'mean_recall_at_25_direct': top_25_recall_knockouts_direct.iloc[:,1].mean()
                        }

    return summary_analysis, ttest_df

def compute_outlier_activation_test(ttest_df, adata, ptb_targets, mode = 'sena'):

    ## correct pvalues
    if ttest_df['scoretype'].iloc[0] == 'ttest':
        ttest_df['score'] = ttest_df['score'].fillna(1)
        ttest_df['score'] = ttest_df['score'].replace({0: 1e-200})
        ttest_df['score'] = ttest_df['score'] * ttest_df.shape[0] #bonferroni correction

    ptb_targets_direct = retrieve_direct_genes(adata, ptb_targets)
    #['LHX1', 'COL2A1','BCL2L11','CLDN6','SGK1','TGFBR2','TMSB4X']

    ## perform t test between the top affected and the next non-affected gene sets
    outlier_activation = []
    for knockout in ttest_df['knockout'].unique():
        
        if knockout not in ptb_targets_direct:
            continue

        knockout_distr = ttest_df[ttest_df['knockout'] == knockout]
        #num_affected = knockout_distr['affected'].sum()

        score_affected = knockout_distr.loc[knockout_distr['affected'] == True, 'score'].values.mean()

        # if mode == 'sena':
        #     score_affected = knockout_distr.loc[knockout_distr['affected'] == True, 'score'].values.mean()
        # elif mode == 'regular':
        #     score_affected = np.mean(sorted(knockout_distr['score'].values, reverse=True)[:num_affected])

        #second_top_affected = sorted(knockout_distr, reverse=True)[num_affected:2*num_affected]
        #randomly_affected = sample(list(knockout_distr['score'].values), knockout_distr_affected.shape[0])
        median_affected = knockout_distr['score'].median()

        # for i in range(knockout_distr_affected.shape[0]):
        #     outlier_activation.append([knockout, knockout_distr_affected[i], randomly_affected[i]])
        outlier_activation.append([knockout, score_affected, median_affected])
        #print(f"knockout: {knockout}, affected: {score_affected}, mean: {mean_affected}")
            
    outlier_activation_df = pd.DataFrame(outlier_activation)
    outlier_activation_df.columns = ['knockout','top_score', 'median_score']

    ##compute metrics
    paired_pval = ttest_ind(outlier_activation_df.iloc[:,1].values, outlier_activation_df.iloc[:,2].values).pvalue
    z_diff = (outlier_activation_df.iloc[:,1].values - outlier_activation_df.iloc[:,2].values).mean()

    #append to dict
    outlier_activation_dict = {'mode': mode, '-log10(pval)': -1 * m.log10(paired_pval), 'z_diff': z_diff, 'scoretype': ttest_df['scoretype'].iloc[0]}

    return outlier_activation_dict

"""tools"""
def retrieve_direct_genes(adata, ptb_targets):
    ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_mapping_dict = dict(zip(ensembl_genename_mapping.iloc[:,1], ensembl_genename_mapping.iloc[:,0]))
    ensembl_genename_mapping_revdict = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
    ptb_targets_ens = list(map(lambda x: ensembl_genename_mapping_dict[x], ptb_targets))
    ptb_targets_direct = [ensembl_genename_mapping_revdict[x] for x in ptb_targets_ens if x in adata.var_names]
    return ptb_targets_direct

"""plot"""
def plot_score_distribution(ttest_df, epoch, mode, affected=True):

    #drop NAs
    df = ttest_df.copy()
    df['score'] = df['score'].fillna(1)

    # Set up the matplotlib figure
    sns.set(style="whitegrid")

    # Determine the layout of subplots
    cols = 9  # Number of columns in subplot grid
    rows = 8  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6 * rows), sharey=True)
    axes = axes.flatten()  # Flatten in case of single row
    
    # Plot for each gene set
    for idx, knockout in enumerate(sorted(df['knockout'].unique())):
        
        ax = axes[idx]
        subset = df[df['knockout'] == knockout]
        
        sns.scatterplot(
            data=subset,
            x=subset.index,
            y='pval',
            hue='affected' if affected else None,
            color='blue' if not affected else None,
            alpha=0.7,
            ax=ax
        )

        ax.set_title(f"Knockout: {knockout} - Number of affected {subset['affected'].sum()}")
        ax.set_xlabel('Gene Index')
        ax.set_ylabel('-log10(p-value)')
        if affected:
            ax.legend(title='Affected', loc='lower right')
        ax.set_yscale('log')
        
    plt.tight_layout()
    output_path = os.path.join('./../../figures/ablation_study',f'ae_{mode}',f'autoencoder_{mode}_ablation_1layer_hist_epoch_{epoch}.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_knockout_distribution(ttest_df, epoch, mode, seed, adata, ptb_targets, knockout = 'LHX1'):

    affected = mode == 'sena'
    df = ttest_df.copy()
    df['score'] = df['score'].fillna(1)

    # Set up the matplotlib figure
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharey=True)

    if knockout != 'combined':
        subset = df[df['knockout'] == knockout]
    else:
        ptb_targets_direct = retrieve_direct_genes(adata, ptb_targets)
        subset = df[df['knockout'].isin(ptb_targets_direct)]
    
    sns.scatterplot(
        data=subset,
        x=subset.index,
        y='score',
        hue='affected' if affected else None,
        color='blue' if not affected else None,
        alpha=0.7,
        ax=ax
    )

    ax.set_title(f"mode: {mode} - Knockout: {knockout} - Number of affected {subset['affected'].sum()}")
    ax.set_xlabel('Gene Index')
    ax.set_ylabel(ttest_df['scoretype'].iloc[0])
    if affected:
        ax.legend(title='Affected', loc='lower right')

    if ttest_df['scoretype'].iloc[0] == 'ttest':
        ax.set_yscale('log')
        
    plt.tight_layout()
    output_path = os.path.join('./../../figures/ablation_study',f'ae_{mode}',f'autoencoder_{mode}_ablation_1layer_hist_knockout_{knockout}_epoch_{epoch}_seed_{seed}.png')
    plt.savefig(output_path, dpi=300)
    plt.close(fig)



"""NA LAYER"""
class NetActivity_layer(torch.nn.Module):

    def __init__(self, input_genes, output_gs, relation_dict, bias = True, device = None, dtype = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_genes = input_genes
        self.output_gs = output_gs
        self.relation_dict = relation_dict

        ## create sparse weight matrix according to GO relationships
        mask = torch.empty((self.input_genes, self.output_gs), **factory_kwargs)

        ## set to 1 remaining values
        for i in range(self.input_genes):
            for latent_go in self.relation_dict[i]:
                mask[i,latent_go] = 1

        self.mask = mask
        self.weight = nn.Parameter(torch.empty((self.output_gs, self.input_genes), **factory_kwargs))
        self.bias = None

        # if bias:
        #     self.bias = nn.Parameter(torch.empty(middle_layer, **factory_kwargs))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        return (x @ ((self.weight * self.mask.T).T))
        #return (torch.sparse.mm(x, self.weight))
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)