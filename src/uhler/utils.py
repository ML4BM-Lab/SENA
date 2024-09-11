import random
import numpy as np
import torch
import pandas as pd
import os
import scanpy as sc
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.stats import ttest_ind
import math as m
from random import sample
import pickle
from torch.utils.data import Dataset
from collections import Counter


"""activation score"""
def build_activity_score_df(model, adata, ptb_targets):

    na_activity_score = {}
    for int_type in ptb_targets+['']: #+control
        
        obs = adata[adata.obs['guide_ids'] == int_type].X.todense()
        int_df = pd.DataFrame(model.encoder(torch.tensor(obs).float().to('cuda')).detach().cpu().numpy())
        new_int_type = int_type if int_type != '' else 'ctrl'
        na_activity_score[new_int_type] = int_df.to_numpy()

    return na_activity_score

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
    outlier_activation_df.columns = ['knockout', 'top_score', 'median_score', 'recall_at_100', 'recall_at_25']

    """ compute metrics for first test """
    z_diff = (outlier_activation_df.iloc[:,1].values - outlier_activation_df.iloc[:,2].values).mean()

    #append to dict
    outlier_activation_dict = {'mode': mode, 'z_diff': z_diff, 'scoretype': ttest_df['scoretype'].iloc[0],
                               'recall_at_100': outlier_activation_df['recall_at_100'].mean(),
                               'recall_at_25': outlier_activation_df['recall_at_25'].mean()}

    return pd.DataFrame(outlier_activation_dict, index = [0])

"""Data related """

def load_norman_2019_dataset(num_gene_th=5):

    def load_gene_go_assignments():

        #filter genes that are not in any GO
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        GO_to_ensembl_id_assignment.columns = ['GO_id', 'ensembl_id']

        #reduce the gos to the genes we have in adata
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['ensembl_id'].isin(adata.var_names)]

        #define gos and filter
        gos = sorted(set(pd.read_csv(os.path.join('..','..','data','topGO_Jesus_uhler.tsv'),sep='\t')['PathwayID'].values.tolist()))
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(gos)]

        #keep only genesets when containing more than 5 genes
        counter_genesets_df = pd.DataFrame(Counter(GO_to_ensembl_id_assignment['GO_id']),index=[0]).T
        genesets_in = counter_genesets_df[counter_genesets_df.values >= num_gene_th].index
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(genesets_in)]

        #redefine gos
        gos = sorted(GO_to_ensembl_id_assignment['GO_id'].unique())

        #generate dict
        gene_go_dict = defaultdict(list)

        for go,ens in GO_to_ensembl_id_assignment.values:
            gene_go_dict[ens].append(go)

        return gos, GO_to_ensembl_id_assignment, gene_go_dict

    def compute_affecting_perturbations(GO_to_ensembl_id_assignment):

        #filter interventions that not in any GO
        ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
        ensembl_genename_mapping_dict = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
        ensembl_genename_mapping_rev = dict(zip(ensembl_genename_mapping.iloc[:,1], ensembl_genename_mapping.iloc[:,0]))

        ##get intervention targets
        intervention_genenames = map(lambda x: ensembl_genename_mapping_dict.get(x,None), GO_to_ensembl_id_assignment['ensembl_id'])
        ptb_targets = list(set(intervention_genenames).intersection(set([x for x in adata.obs['guide_ids'] if x != '' and ',' not in x])))
        ptb_targets_ens = list(map(lambda x: ensembl_genename_mapping_rev[x], ptb_targets))

        return ptb_targets, ptb_targets_ens, ensembl_genename_mapping_rev

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
    
    #build genesets
    ptb_targets = sorted(adata.obs['guide_ids'].unique().tolist())[1:]
    gos, GO_to_ensembl_id_assignment, gene_go_dict = load_gene_go_assignments()

    #compute perturbations with at least 1 gene set for interpretability metrics
    ptb_targets_affected, ptb_targets_ens_affected, ensembl_genename_mapping_rev = compute_affecting_perturbations(GO_to_ensembl_id_assignment) 

    #build gene-go rel
    rel_dict = build_gene_go_relationships(adata, gos, GO_to_ensembl_id_assignment)

    """double data"""
    double_adata = sc.read_h5ad(fpath).copy()
    double_adata = double_adata[:, double_adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
    double_adata = double_adata[(double_adata.obs['guide_ids'].str.contains(',')) & (double_adata.obs['guide_ids'].map(lambda x: all([y in ptb_targets for y in x.split(',')])))]

    return adata, double_adata, ptb_targets, ptb_targets_affected, gos, rel_dict, gene_go_dict, ensembl_genename_mapping_rev

def get_data(batch_size=32, mode='train', perturb_targets=None):
    assert mode in ['train', 'test'], 'mode not supported!'

    if mode == 'train':
        dataset = SCDataset(perturb_type='single', perturb_targets=perturb_targets)
        train_idx, test_idx = split_scdata(
            dataset, 
            split_ptbs = ['ETS2', 'SGK1', 'POU3F2', 'TBX2', 'CBL', 'MAPK1', 'CDKN1C', 'S1PR2', 'PTPN1', 'MAP2K6', 'COL1A1'],
            batch_size=batch_size
        ) # leave out some cells from the top 12 single target-gene interventions
    elif mode == 'test':
        assert perturb_targets is not None, 'perturb_targets has to be specified during testing, otherwise the index might be mismatched!'
        dataset = SCDataset(perturb_type='double', perturb_targets=perturb_targets)

    ptb_genes = dataset.ptb_targets
    
    if mode == 'train':
        dataset1 = Subset(dataset, train_idx)
        ptb_name = dataset.ptb_names[train_idx]
        dataloader = DataLoader(
            dataset1,
            batch_sampler=SCDATA_sampler(dataset1, batch_size, ptb_name),
            num_workers=0,
            #shuffle=True,
            #batch_size=batch_size
        )
        
        dim = dataset[0][0].shape[0]
        cdim = dataset[0][2].shape[0]

        dataset2 = Subset(dataset, test_idx)
        ptb_name = dataset.ptb_names[test_idx]
        dataloader2 = DataLoader(
            dataset2,
            batch_sampler=SCDATA_sampler(dataset2, 8, ptb_name), #using 8 coz 20% of cells somtimes is lower than 128 and it drops that gene
            num_workers=0,
            #shuffle=True,
            #batch_size=batch_size
        )

        return dataloader, dataloader2, dim, cdim, ptb_genes
    
    else:

        dataloader = DataLoader(
            dataset,
            batch_sampler=SCDATA_sampler(dataset, batch_size),
            num_workers=0,
            #shuffle=True,
            #batch_size=batch_size
        )
        
        dim = dataset[0][0].shape[0]
        cdim = dataset[0][2].shape[0]

        return dataloader, dim, cdim, ptb_genes

""" data sampler"""

class SCDataset(Dataset):
    def __init__(self, datafile='./../../data/Norman2019_raw.h5ad', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double'], 'perturb_type not supported!'

        adata, double_adata, ptb_targets, _, _, _, _, _ = load_norman_2019_dataset()

        #get genes and ptb targets
        self.genes = adata.var.index.tolist()
        self.ptb_targets = ptb_targets
        
        if perturb_type == 'single':
            
            ptb_adata = adata[(~adata.obs['guide_ids'].str.contains(',')) & (adata.obs['guide_ids']!='')].copy()

            #keep only cells containing our perturbed genes
            ptb_adata = ptb_adata[ptb_adata.obs['guide_ids'].isin(ptb_targets), :]

            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(self.ptb_targets, ptb_adata.obs['guide_ids'].values)
            
        elif perturb_type == 'double':

            ptb_adata = double_adata[(double_adata.obs['guide_ids'].str.contains(',')) & (double_adata.obs['guide_ids']!='')].copy()

            #keep only cells containing our perturbed genes
            ptb_adata = ptb_adata[ptb_adata.obs['guide_ids'].apply(lambda x: all([y in ptb_targets for y in x.split(',')])), :]

            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(self.ptb_targets, ptb_adata.obs['guide_ids'].values)            

        self.ctrl_samples = adata[adata.obs['guide_ids']==''].X.copy()
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
            ]
        

    def __getitem__(self, item):
        x = torch.from_numpy(self.rand_ctrl_samples[item].toarray().flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].toarray().flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c
    
    def __len__(self):
        return self.ptb_samples.shape[0]

def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[[all_ptb_targets.index(i) for i in id.split(',')]] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)

class SCDATA_sampler(Sampler):
    def __init__(self, scdataset, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = scdataset.ptb_names
        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0]
            self.intervindices.append(idx)
            self.len += len(idx) // batchsize
        self.batchsize = batchsize
    
    def __iter__(self):
        comb = []
        for i in range(len(self.intervindices)):
            random.shuffle(self.intervindices[i])
        
            interv_batches = chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return self.len

def split_scdata(scdataset, split_ptbs, batch_size=32):

    # num_batch = 96 // batch_size
    # num_sample = num_batch * batch_size
    pct = 0.2

    test_idx = []
    for ptb in split_ptbs:
        idx = list(np.where(scdataset.ptb_names == ptb)[0])
        #print(f"{ptb} - {idx}")
        test_idx.append(np.random.choice(idx, int(len(idx)*pct), replace=False))
        #test_idx.append(idx[0:num_sample])

    test_idx = list(np.hstack(test_idx))
    train_idx = [l for l in range(len(scdataset)) if l not in test_idx]
    
    return train_idx, test_idx

def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size)
    
    if len(indices) % chunk_size == 0:
        return split
    elif len(split) > 0:
        return split[:-1]
    else:
        return None

"""MMD LOSS"""
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
