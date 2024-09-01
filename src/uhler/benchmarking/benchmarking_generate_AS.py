import networkx as nx
import numpy as np
import sys
sys.path.append('./../')

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
import numpy as np

def load_data_full_go(ptb_targets):

    #define url
    datafile='./../../../data/Norman2019_raw.h5ad'
    adata = sc.read_h5ad(datafile)

    # load gos from NA paper
    GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
    GO_to_ensembl_id_assignment.columns = ['GO_id','ensembl_id']

    #load GOs
    go_2_z_post_heuristic = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_2_z_post_heuristic.csv'))['GO'].values
    GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(go_2_z_post_heuristic)]

    ## load interventions
    intervention_to_GO_assignment_genes = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways', 'z_2_interventions.csv')).columns[1:].tolist()
    assert all(np.array(sorted(ptb_targets)) == np.array(sorted(intervention_to_GO_assignment_genes)))
    
    # ## keep only GO_genes
    adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
    adata.obs = adata.obs.reset_index(drop=True)

    ## get control
    perturbations_idx_dict = {}
    for gene in ['ctrl'] + ptb_targets:

        if gene == 'ctrl':
            perturbations_idx_dict[gene] = (adata.obs[adata.obs['guide_ids'] == '']).index.values
        else:
            perturbations_idx_dict[gene] = (adata.obs[adata.obs['guide_ids'] == gene]).index.values

    return adata, perturbations_idx_dict, go_2_z_post_heuristic

def load_data_raw_go(ptb_targets):

    #define url
    datafile='./../../../data/Norman2019_raw.h5ad'
    adata = sc.read_h5ad(datafile)

    # load gos from NA paper
    GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
    GO_to_ensembl_id_assignment.columns = ['GO_id','ensembl_id']

    #load GOs
    go_2_z_raw = pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')
    GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(go_2_z_raw['PathwayID'].values)]

    ## load interventions
    ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_mapping = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
    intervention_genenames = map(lambda x: ensembl_genename_mapping.get(x,None), GO_to_ensembl_id_assignment['ensembl_id'])
    intervention_to_GO_assignment_genes = list(set(intervention_genenames).intersection(set([x for x in adata.obs['guide_ids'] if x != '' and ',' not in x])))
        
    assert all(np.array(sorted(ptb_targets)) == np.array(sorted(intervention_to_GO_assignment_genes)))
    
    # ## keep only GO_genes
    adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
    adata.obs = adata.obs.reset_index(drop=True)

    ## get control
    perturbations_idx_dict = {}
    for gene in ['ctrl'] + ptb_targets:

        if gene == 'ctrl':
            perturbations_idx_dict[gene] = (adata.obs[adata.obs['guide_ids'] == '']).index.values
        else:
            perturbations_idx_dict[gene] = (adata.obs[adata.obs['guide_ids'] == gene]).index.values

    return adata, perturbations_idx_dict, sorted(set(go_2_z_raw['PathwayID'].values)), sorted(set(go_2_z_raw['topGO'].values)) 

##load our model
seed = 42
mode_type = 'full_go'
trainmode = 'sena_delta_0'
layertype = 'z'
model_name = f'{mode_type}_{trainmode}'
savedir = f'./../../../result/uhler/{model_name}/seed_{seed}' 
model = torch.load(f'{savedir}/best_model.pt')

##get the output of NetActivity Layer
batch_size, mode = 128, 'train'
_, _, _, dim, cdim, ptb_targets = get_data(batch_size=batch_size, mode=mode)
adata, idx_dict, gos, zs = load_data_raw_go(ptb_targets)

netactivity_scores = []
for gene in tqdm(idx_dict, desc = 'generating activity score for perturbations'):
    
    idx = idx_dict[gene]
    mat = torch.from_numpy(adata.X[idx,:].todense()).to('cuda').double()

    if layertype == 'fc1':
        colnames = gos
        na_score = model.fc1(mat).detach().cpu().numpy()

    elif layertype == 'fc_mean':
        na_score = model.fc_mean(model.fc1(mat)).detach().cpu().numpy()
        colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))

    elif layertype == 'fc_var':
        na_score = model.fc_var(model.fc1(mat)).detach().cpu().numpy()
        colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))

    elif layertype == 'z':

        mu, var = model.encode(mat)
        z = model.reparametrize(mu, var).detach().cpu().numpy()
        na_score = z
        colnames = zs if na_score.shape[1] == len(zs) else list(range(na_score.shape[1]))


    ##
    na_score_df = pd.DataFrame(na_score)
    na_score_df.columns = colnames
    na_score_df['type'] = gene
    netactivity_scores.append(na_score_df)

##
df_netactivity_scores = pd.concat(netactivity_scores)
df_netactivity_scores.to_csv(os.path.join('./../../../result','uhler',f'{mode_type}_{trainmode}',f'seed_{seed}',f'na_activity_scores_layer_{layertype}.tsv'),sep='\t')
