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
def load_norman_2019_dataset(subsample):

    def load_gene_go_assignments(subsample):

        #filter genes that are not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        GO_to_ensembl_id_assignment.columns = ['GO_id', 'ensembl_id']
        
        #apply subsampling
        if subsample == 'raw':
            gos = sorted(set(GO_to_ensembl_id_assignment['GO_id'].values))

        elif 'topgo' in subsample:

            if 'topgo_' in subsample:

                subsample_ratio = int(subsample.split('_')[1])/100
                gos = sorted(set(pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')['PathwayID'].values.tolist()))

                #select those gos that are not affected by perturbations
                gos = sample(gos, len(gos) * subsample_ratio)

            else:
                gos = sorted(set(pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')['PathwayID'].values.tolist()))

            #now filter by go
            GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(gos)]

        return gos, GO_to_ensembl_id_assignment

    def compute_affecting_perturbations(GO_to_ensembl_id_assignment):

        #filter interventions that not in any GO
        ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
        ensembl_genename_mapping_dict = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
        ensembl_genename_mapping_rev = dict(zip(ensembl_genename_mapping.iloc[:,1], ensembl_genename_mapping.iloc[:,0]))

        ##
        intervention_genenames = map(lambda x: ensembl_genename_mapping_dict.get(x,None), GO_to_ensembl_id_assignment['ensembl_id'])
        ptb_targets = list(set(intervention_genenames).intersection(set([x for x in adata.obs['guide_ids'] if x != '' and ',' not in x])))
        ptb_targets_ens = list(map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets))

        return ptb_targets, ptb_targets_ens

    def build_gene_go_relationships(adata, gos, GO_to_ensembl_id_assignment):

        ## get genes
        genes = adata.var.index.values
        go_dict, gen_dict = dict(zip(gos, range(len(gos)))), dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)
        gene_set, go_set = set(genes), set(gos)

        for go, gen in tqdm(zip(GO_to_ensembl_id_assignment['GO_id'], GO_to_ensembl_id_assignment['ensembl_id']), total = GO_to_ensembl_id_assignment.shape[0]):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return rel_dict

    #define fpath
    fpath = './../../data/Norman2019_raw.h5ad'

    """keep only single interventions"""
    adata = sc.read_h5ad(fpath)
    adata = adata[(~adata.obs['guide_ids'].str.contains(','))]
    
    #drop genes that are not in any GO
    gos, GO_to_ensembl_id_assignment = load_gene_go_assignments(subsample)
    adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]

    #compute perturbations with at least 1 gene set
    ptb_targets, ptb_targets_ens = compute_affecting_perturbations(GO_to_ensembl_id_assignment) 

    #build gene-go rel
    rel_dict = build_gene_go_relationships(adata, gos, GO_to_ensembl_id_assignment)
    
    #keep only perturbations affecting at least one gene set
    adata = adata[adata.obs['guide_ids'].isin(ptb_targets+[''])]

    #(adata.var_names.isin(ptb_targets_ens)).sum()

    return adata, ptb_targets, ptb_targets_ens, gos, rel_dict

"""tools"""
def retrieve_direct_genes(adata, ptb_targets):
    ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_mapping_dict = dict(zip(ensembl_genename_mapping.iloc[:,1], ensembl_genename_mapping.iloc[:,0]))
    ensembl_genename_mapping_revdict = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
    ptb_targets_ens = list(map(lambda x: ensembl_genename_mapping_dict[x], ptb_targets))
    ptb_targets_direct = [ensembl_genename_mapping_revdict[x] for x in ptb_targets_ens if x in adata.var_names]
    return ptb_targets_direct

def build_activity_score_df(model, adata):

    na_activity_score = {}
    intervention_types = list(adata.obs['guide_ids'].values.unique())

    for int_type in intervention_types:
        
        obs = adata[adata.obs['guide_ids'] == int_type].X.todense()
        int_df = pd.DataFrame(model.encoder(torch.tensor(obs).float().to('cuda')).detach().cpu().numpy())
        new_int_type = int_type if int_type != '' else 'ctrl'
        na_activity_score[new_int_type] = int_df.to_numpy()

    return na_activity_score

"""analysis"""
def compute_layer_weight_contribution(model, adata, knockout, ens_knockout, num_affected, mode = 'sena', only_exp=False):

    #get knockout id
    genes = adata.var_names.values
    if ens_knockout in genes:
        idx = np.where(genes == ens_knockout)[0][0]
    else: 
        idx = 0
    
    #get layer matrix
    W = model.encoder.weight[:,idx].detach().cpu().numpy()

    if mode[:7] == 'regular' or mode[:2] == 'l1':
        bias = model.encoder.bias.detach().cpu().numpy()
    elif mode[:4] == 'sena':
        bias = 0

    #get contribution to knockout gene from ctrl to each of the activation (gene input * weight)
    ctrl_exp_mean = adata[adata.obs['guide_ids'] == ''].X[:,idx].todense().mean()
    knockout_exp_mean = adata[adata.obs['guide_ids'] == knockout].X[:,idx].todense().mean()

    #compute contribution
    ctrl_contribution = W * ctrl_exp_mean + bias
    knockout_contribution = W * knockout_exp_mean + bias

    #now compute difference of means
    diff_vector = np.abs(ctrl_contribution - knockout_contribution)

    if only_exp:
        return diff_vector

    top_idxs = np.argsort(diff_vector)[::-1]

    return list(top_idxs[:num_affected])

def compute_activation_df(model, adata, gos, scoretype = 'ttest', mode = 'sena'):

    #build ac
    na_activity_score = build_activity_score_df(model, adata)

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
            if mode[:7] == 'regular' or mode[:2] == 'l1':
                belonging_genesets = compute_layer_weight_contribution(model, adata, knockout, ensembl_genename_dict[knockout], len(belonging_genesets)) 

            #generate diff vector
            if scoretype == 'knockout_contr':
                diff_vector = compute_layer_weight_contribution(model, adata, knockout, ensembl_genename_dict[knockout], len(belonging_genesets), only_exp=True)    

            for i, geneset in enumerate(gos):
                
                #perform ttest
                if scoretype == 'ttest':
                    _, p_value = ttest_ind(ctrl_cells[:,i], knockout_cells[:,i], equal_var=False)
                    score = -1 * m.log10(p_value)

                elif scoretype == 'mu_diff':
                    score = abs(ctrl_cells[:,i].mean() - knockout_cells[:,i].mean())

                elif scoretype == 'knockout_contr':
                    score = diff_vector[i]
                    
                #append info
                if mode[:4] == 'sena':
                    ttest_df.append([knockout, geneset, scoretype, score, geneset in belonging_genesets])
                elif mode[:7] == 'regular' or mode[:2] == 'l1':
                    ttest_df.append([knockout, i, scoretype, score, i in belonging_genesets])

    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout', 'geneset', 'scoretype', 'score', 'affected']

    #apply min-max norm
    if scoretype == 'mu_diff':
        for knockout in ttest_df['knockout'].unique():
            ttest_df.loc[ttest_df['knockout'] == knockout, 'score'] = MinMaxScaler().fit_transform(ttest_df.loc[ttest_df['knockout'] == knockout, 'score'].values.reshape(-1,1))

    return ttest_df

def compute_outlier_activation_analysis(ttest_df, adata, ptb_targets, mode = 'sena'):

    ## correct pvalues
    if ttest_df['scoretype'].iloc[0] == 'ttest':
        ttest_df['score'] = ttest_df['score'].fillna(1)
        ttest_df['score'] = ttest_df['score'].replace({0: 1e-200})
        ttest_df['score'] = ttest_df['score'] * ttest_df.shape[0] #bonferroni correction

    ptb_targets_direct = retrieve_direct_genes(adata, ptb_targets)
    
    ## compute metric for each knockout
    outlier_activation = []
    for knockout in ttest_df['knockout'].unique():
        
        if knockout not in ptb_targets_direct:
            continue

        knockout_distr = ttest_df[ttest_df['knockout'] == knockout]

        """first test - zdiff distribution differences"""
        score_affected = knockout_distr.loc[knockout_distr['affected'] == True, 'score'].values.mean()
        median_affected = knockout_distr['score'].median()
        
        """second test - recall at 25/100"""
        recall_at_100 = knockout_distr.sort_values(by='score', ascending=False)['affected'].iloc[:100].sum() / knockout_distr['affected'].sum()
        recall_at_25 = knockout_distr.sort_values(by='score', ascending=False)['affected'].iloc[:25].sum() / knockout_distr['affected'].sum()

        ##append
        outlier_activation.append([knockout, score_affected, median_affected, recall_at_100, recall_at_25])

    """append results into dataframe"""        
    outlier_activation_df = pd.DataFrame(outlier_activation)
    outlier_activation_df.columns = ['knockout','top_score', 'median_score', 'recall_at_100', 'recall_at_25']

    """ compute metrics for first test """
    paired_pval = ttest_ind(outlier_activation_df.iloc[:,1].values, outlier_activation_df.iloc[:,2].values).pvalue
    z_diff = (outlier_activation_df.iloc[:,1].values - outlier_activation_df.iloc[:,2].values).mean()

    #append to dict
    outlier_activation_dict = {'mode': mode, '-log10(pval)': -1 * m.log10(paired_pval), 
                               'z_diff': z_diff, 'scoretype': ttest_df['scoretype'].iloc[0],
                               'recall_at_100': outlier_activation_df['recall_at_100'].mean(),
                               'recall_at_25': outlier_activation_df['recall_at_25'].mean()}

    return pd.DataFrame(outlier_activation_dict, index = [0])

def compute_latent_correlation_analysis(model, adata, ptb_targets, gos, ttest_df):

    #build ac
    #na_activity_score = build_activity_score_df(model, adata)
    ptb_targets_direct = retrieve_direct_genes(adata, ptb_targets)
    
    #ctrl 
    ctrl_samples = adata[adata.obs['guide_ids']=='']

    # ##build geneset mapping
    # ensemble_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    # ensembl_genename_dict = dict(zip(ensemble_genename_mapping.iloc[:,1], ensemble_genename_mapping.iloc[:,0]))

    latent_corr = []

    for knockout in ptb_targets_direct:
        
        #get scaled zdiff for input expression of knocked-out gene and for latent space activation of affected GOs
        ko_samples = adata[adata.obs['guide_ids'] == knockout]
        
        ##input
        zdiff_input_norm = MinMaxScaler().fit_transform(np.array(np.abs(ctrl_samples.X.mean(axis=0) - ko_samples.X.mean(axis=0))).reshape((-1,1)))

        #get subset of activations
        subset = ttest_df[ttest_df['knockout'] == knockout].copy()
        belonging_genesets = set(subset.loc[subset['affected'] == True,'geneset'].values)

        for i,geneset in enumerate(gos):

            if geneset in belonging_genesets:
                
                #compute affected genes
                affected_genes_idxs = np.where(model.encoder.mask[:,i].detach().cpu().numpy() != 0)[0]
                zdiff_input_norm_knockout = (zdiff_input_norm[affected_genes_idxs,0]).mean()

                ##latent
                zdiff_latent_norm_knockout = subset[subset['geneset'] == geneset]['score'].values[0]

                ##append
                latent_corr.append([knockout, geneset, zdiff_input_norm_knockout, zdiff_latent_norm_knockout])

    """append results into dataframe"""        
    lcorr_df = pd.DataFrame(latent_corr)
    lcorr_df.columns = ['knockout', 'geneset', 'input_zdiff', 'latent_zdiff']

    return lcorr_df

def compute_sparsity_contribution(model, dataset, mode, sparsity_th = 1e-3):

    contributions = []

    with torch.no_grad():
        
        #compute layer weights
        if mode[:4] == 'sena':
            layer_weights = ((model.encoder.weight * model.encoder.mask.T).T)
        else:
            layer_weights = model.encoder.weight.T

        #compute score contributions
        dmean = dataset.mean(axis=0)
        for i in range(dmean.shape[0]):
            contributions.append((dmean[i] * layer_weights[i,:]).detach().cpu().numpy())

    contr_mat = np.vstack(contributions)
    sparsity = (np.abs(contr_mat) <= sparsity_th).sum() / (contr_mat.shape[0]*contr_mat.shape[1])

    return sparsity

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

def plot_weight_distribution(model, epoch, mode):

    w = model.encoder.weight.detach().cpu().numpy().flatten()
    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(w, bins=1000, color='blue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Data', fontsize=16)

    # Show gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_xlim([1e-100,10])
    plt.tight_layout()
    plt.savefig(os.path.join('./../../','figures','ablation_study','weight_matrices',f'ae_{mode}_encoder_1layer_hist_epoch_{epoch}.png'))
    plt.close()
    plt.clf()
    plt.cla()

"""NA LAYER"""
class NetActivity_layer(torch.nn.Module):

    def __init__(self, input_genes, output_gs, relation_dict, bias = True, device = None, dtype = None, sp = 0):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_genes = input_genes
        self.output_gs = output_gs
        self.relation_dict = relation_dict

        ## create sparse weight matrix according to GO relationships
        mask = torch.zeros((self.input_genes, self.output_gs), **factory_kwargs)

        ## set to 1 remaining values
        for i in range(self.input_genes):
            for latent_go in self.relation_dict[i]:
                mask[i,latent_go] = 1

        self.mask = mask
        self.mask[self.mask == 0] = sp

        #apply sp

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