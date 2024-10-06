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
from collections import Counter

class Norman2019DataLoader:
    def __init__(
        self, num_gene_th=5, dataname="Norman2019_raw"
    ):
        self.num_gene_th = num_gene_th
        self.datafile = os.path.join('data',f"{dataname}.h5ad")

        # Initialize variables
        self.adata = None
        self.double_adata = None
        self.ptb_targets = None
        self.ptb_targets_affected = None
        self.gos = None
        self.rel_dict = None
        self.gene_go_dict = None
        self.ensembl_genename_mapping_rev = None

        # Load the dataset
        self.load_norman_2019_dataset()

    def load_norman_2019_dataset(self):
        # Define file path
        fpath = self.datafile

        # Keep only single interventions
        adata = sc.read_h5ad(fpath)
        adata = adata[(~adata.obs["guide_ids"].str.contains(","))]

        # Build gene sets
        gos, GO_to_ensembl_id_assignment, gene_go_dict = self.load_gene_go_assignments(
            adata
        )

        # Compute perturbations with at least 1 gene set
        ptb_targets_affected, _, ensembl_genename_mapping_rev = (
            self.compute_affecting_perturbations(adata, GO_to_ensembl_id_assignment)
        )

        # Build gene-GO relationships
        rel_dict = self.build_gene_go_relationships(
            adata, gos, GO_to_ensembl_id_assignment
        )

        # Load double perturbation data
        ptb_targets = sorted(adata.obs["guide_ids"].unique().tolist())[1:]
        double_adata = sc.read_h5ad(fpath).copy()
        double_adata = double_adata[
            (double_adata.obs["guide_ids"].str.contains(","))
            & (
                double_adata.obs["guide_ids"].map(
                    lambda x: all([y in ptb_targets for y in x.split(",")])
                )
            )
        ]

        # Assign instance variables
        self.adata = adata
        self.double_adata = double_adata
        self.ptb_targets = ptb_targets
        self.ptb_targets_affected = ptb_targets_affected
        self.gos = gos
        self.rel_dict = rel_dict
        self.gene_go_dict = gene_go_dict
        self.ensembl_genename_mapping_rev = ensembl_genename_mapping_rev

    def load_gene_go_assignments(self, adata):
        # Filter genes not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(
            os.path.join("data", "go_kegg_gene_map.tsv"), sep="\t"
        )
        GO_to_ensembl_id_assignment.columns = ["GO_id", "ensembl_id"]

        # Reduce GOs to the genes we have in adata
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["ensembl_id"].isin(adata.var_names)
        ]

        # Define GOs and filter
        gos = sorted(
            set(
                pd.read_csv(os.path.join("data", "topGO_uhler.tsv"), sep="\t")[
                    "PathwayID"
                ].values.tolist()
            )
        )
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(gos)
        ]

        # Keep only gene sets containing more than num_gene_th genes
        counter_genesets_df = pd.DataFrame(
            Counter(GO_to_ensembl_id_assignment["GO_id"]), index=[0]
        ).T
        genesets_in = counter_genesets_df[
            counter_genesets_df.values >= self.num_gene_th
        ].index
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[
            GO_to_ensembl_id_assignment["GO_id"].isin(genesets_in)
        ]

        # Redefine GOs
        gos = sorted(GO_to_ensembl_id_assignment["GO_id"].unique())

        # Generate gene-GO dictionary
        gene_go_dict = defaultdict(list)
        for go, ens in GO_to_ensembl_id_assignment.values:
            gene_go_dict[ens].append(go)

        return gos, GO_to_ensembl_id_assignment, gene_go_dict

    def compute_affecting_perturbations(self, adata, GO_to_ensembl_id_assignment):
        # Filter interventions not in any GO
        ensembl_genename_mapping = pd.read_csv(
            os.path.join("data", "ensembl_genename_mapping.tsv"), sep="\t"
        )
        ensembl_genename_mapping_dict = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 0], ensembl_genename_mapping.iloc[:, 1]
            )
        )
        ensembl_genename_mapping_rev = dict(
            zip(
                ensembl_genename_mapping.iloc[:, 1], ensembl_genename_mapping.iloc[:, 0]
            )
        )

        # Get intervention targets
        intervention_genenames = map(
            lambda x: ensembl_genename_mapping_dict.get(x, None),
            GO_to_ensembl_id_assignment["ensembl_id"],
        )
        ptb_targets = list(
            set(intervention_genenames).intersection(
                set([x for x in adata.obs["guide_ids"] if x != "" and "," not in x])
            )
        )
        ptb_targets_ens = list(
            map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets)
        )

        return ptb_targets, ptb_targets_ens, ensembl_genename_mapping_rev

    def build_gene_go_relationships(self, adata, gos, GO_to_ensembl_id_assignment):
        # Get genes
        genes = adata.var.index.values
        go_dict = dict(zip(gos, range(len(gos))))
        gen_dict = dict(zip(genes, range(len(genes))))
        rel_dict = defaultdict(list)
        gene_set, go_set = set(genes), set(gos)

        for go, gen in zip(
            GO_to_ensembl_id_assignment["GO_id"],
            GO_to_ensembl_id_assignment["ensembl_id"],
        ):
            if (gen in gene_set) and (go in go_set):
                rel_dict[gen_dict[gen]].append(go_dict[go])

        return rel_dict

"""tools"""
def build_activity_score_df(model, adata, ptb_targets):

    na_activity_score = {}
    for int_type in ptb_targets+['']: #+control
        
        obs = adata[adata.obs['guide_ids'] == int_type].X.todense()
        int_df = pd.DataFrame(model.encoder(torch.tensor(obs).float().to('cuda')).detach().cpu().numpy())
        new_int_type = int_type if int_type != '' else 'ctrl'
        na_activity_score[new_int_type] = int_df.to_numpy()

    return na_activity_score

"""analysis"""
def compute_layer_weight_contribution(model, adata, knockout, ens_knockout, num_affected, mode = 'sena', only_exp=False):

    #get knockout id
    genes = adata.var_names.values
    idx = np.where(genes == ens_knockout)[0][0]

    #get layer matrix
    if mode[:4] == 'sena':
        W = (model.encoder.weight * model.encoder.mask.T)[:,idx].detach().cpu().numpy()
    else:
        W = model.encoder.weight[:,idx].detach().cpu().numpy()
    # bias = model.encoder.bias.detach().cpu().numpy()

    #get contribution to knockout gene from ctrl to each of the activation (gene input * weight)
    ctrl_exp_mean = adata[adata.obs['guide_ids'] == ''].X[:,idx].todense().mean()
    knockout_exp_mean = adata[adata.obs['guide_ids'] == knockout].X[:,idx].todense().mean()

    #compute contribution
    ctrl_contribution = W * ctrl_exp_mean # + bias
    knockout_contribution = W * knockout_exp_mean # + bias

    #now compute difference of means
    diff_vector = np.abs(ctrl_contribution - knockout_contribution)

    if only_exp:
        return diff_vector

    top_idxs = np.argsort(diff_vector)[::-1]

    return list(top_idxs[:num_affected])

def compute_activation_df(model, adata, gos, scoretype, mode, gene_go_dict, ensembl_genename_dict, ptb_targets):

    #build ac
    na_activity_score = build_activity_score_df(model, adata, ptb_targets)

    ## define control cells
    ctrl_cells = na_activity_score['ctrl']

    ## init df
    ttest_df = []

    for knockout in na_activity_score.keys():
        
        if knockout == 'ctrl':
            continue
        
        #get knockout cells       
        ens_knockout = ensembl_genename_dict[knockout]
        knockout_cells = na_activity_score[knockout]

        #compute affected genesets
        if mode[:4] == 'sena':
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[ensembl_genename_dict[knockout]]]
        else:
            belonging_genesets = compute_layer_weight_contribution(model, adata, knockout, ens_knockout, 100, mode=mode) 

        for i, geneset in enumerate(gos):
            
            #perform ttest
            if scoretype == 'ttest':
                _, p_value = ttest_ind(ctrl_cells[:,i], knockout_cells[:,i], equal_var=False)
                score = -1 * m.log10(p_value)

            elif scoretype == 'mu_diff':
                score = abs(ctrl_cells[:,i].mean() - knockout_cells[:,i].mean())

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

def compute_outlier_activation_analysis(ttest_df, mode = 'sena'):

    ## correct pvalues
    if ttest_df['scoretype'].iloc[0] == 'ttest':
        ttest_df['score'] = ttest_df['score'].fillna(1)
        ttest_df['score'] = ttest_df['score'].replace({0: 1e-200})
        ttest_df['score'] = ttest_df['score'] * ttest_df.shape[0] #bonferroni correction

    ## compute metric for each knockout
    outlier_activation = []
    for knockout in ttest_df['knockout'].unique():
  
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

"""sparsity"""
def compute_sparsity_contribution(model, dataset, mode, sparsity_th = [1e-6]):

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
    sp_dict = {}
    for sth in sparsity_th:
        sparsity = (np.abs(contr_mat) <= sth).sum() / (contr_mat.shape[0]*contr_mat.shape[1])
        sp_dict[f'sparsity_{sth}'] = sparsity

    return sp_dict


    def __init__(self, input_gs, output_genes, relation_dict, bias = True, device = None, dtype = None, sp = 0):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.input_gs = input_gs
        self.output_genes = output_genes

        #rev
        self.relation_dict_rev = defaultdict(list)
        for key, values in relation_dict.items():
            for value in values:
                self.relation_dict_rev[value].append(key)

        ## create sparse weight matrix according to GO relationships
        mask = torch.zeros((self.input_gs, self.output_genes), **factory_kwargs)

        ## set to 1 remaining values
        for i in range(self.input_gs):
            for latent_go in self.relation_dict_rev[i]:
                mask[i,latent_go] = 1

        self.mask = mask
        self.mask[self.mask == 0] = sp

        #apply sp
        self.weight = nn.Parameter(torch.empty((self.output_genes, self.input_gs), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_genes, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x):
        output = (x @ ((self.weight * self.mask.T).T))
        if self.bias is not None:
            return output + self.bias
        return output
        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)