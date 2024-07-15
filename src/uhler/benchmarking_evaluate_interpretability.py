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
import numpy as np

def load_data(ptb_targets):

    #define url
    datafile='./../../data/Norman2019_raw.h5ad'
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

##load our model
model_name = 'full_go_NA+deltas'
savedir = f'./../../result/{model_name}' 
model = torch.load(f'{savedir}/best_model.pt')

##get the output of NetActivity Layer
batch_size, mode = 128, 'train'
_, _, _, _, ptb_targets = get_data(batch_size=batch_size, mode=mode)
adata, idx_dict, go_2_z_post_heuristic = load_data(ptb_targets)

netactivity_scores = []
for gene in tqdm(idx_dict, desc = 'generating activity score for perturbations'):
    
    idx = idx_dict[gene]
    mat = torch.from_numpy(adata.X[idx,:].todense()).to('cuda').double()

    ##forward
    na_score = model.fc1(mat).detach().cpu().numpy()
    na_score_df = pd.DataFrame(na_score)
    na_score_df.columns = go_2_z_post_heuristic
    na_score_df['type'] = gene
    netactivity_scores.append(na_score_df)

##
df_netactivity_scores = pd.concat(netactivity_scores)
df_netactivity_scores.to_csv(os.path.join('./../../result','full_go_NA+deltas','na_activity_scores.tsv'),sep='\t')