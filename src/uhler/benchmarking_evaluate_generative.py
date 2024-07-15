import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import scanpy as sc
import pickle
import torch
from inference import evaluate_single_leftout, evaluate_double
import pandas as pd 
import os
import seaborn as sns

def visualize_data_generation(model_name = 'gosize5_orig_Adam'):

    #load data
    adata = sc.read_h5ad('./../../data/Norman2019_raw.h5ad')
    savedir = f'./../../result/{model_name}' 

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    ## load gosize=5 files
    GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..', '..', 'data', 'GO_to_ensembl_id_assignment_gosize5.csv'))

    # keep only GO_genes
    adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
    ctrl_X = adata.X.toarray()

    #load model
    model = torch.load(f'{savedir}/best_model.pt') 
    mode = 'CMVAE'

    ## generate test results
    rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_single_leftout(model, savedir, model.device, mode)
    C_y = [','.join([str(l) for l in np.where(c_y[i]!=0)[0]]) for i in range(c_y.shape[0])]

    ## generate plots
    all_data = np.vstack([ctrl_X, gt_y, pred_y])
    adata_new = sc.AnnData(all_data)

    #plot them
    sc.tl.pca(adata_new, svd_solver='arpack')
    sc.pp.neighbors(adata_new, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata_new, min_dist=0.3) 
    
    #plot for each target
    for c in set(C_y):
        label = ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' if C_y[i]==c else 'NA' for i in range(len(C_y))] + ['Generated Cells' if C_y[i]==c else 'NA' for i in range(len(C_y))]
        adata_new.obs['label'] = label
        sc.pl.umap(adata_new, size=50, color=['label'], 
                legend_fontsize=14, groups=['Actual Cells','Generated Cells'], 
                title=ptb_targets[int(c)], 
                palette={'Actual Cells': 'blue', 
                            'Generated Cells': 'orange',
                            'NA': 'grey'
                        },
                legend_loc=None,  
                save=f'_test_{ptb_targets[int(c)]}_CMVAE-obs_{model_name}.png'
                )

def visualize_data_generation_comparison(model_names = ['gosize5_orig_Adam']):

    def retrieve_single_run(model_name):

        #load data
        adata = sc.read_h5ad('./../../data/Norman2019_raw.h5ad')
        savedir = f'./../../result/{model_name}' 

        with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
            ptb_targets = pickle.load(f)

        ## load gosize=5 files
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..', '..', 'data', 'GO_to_ensembl_id_assignment_gosize5.csv'))

        # keep only GO_genes
        adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
        ctrl_X = adata.X.toarray()

        #load model
        model = torch.load(f'{savedir}/best_model.pt') 
        mode = 'CMVAE'

        ## generate test results
        rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_single_leftout(model, savedir, model.device, mode)
        C_y = [','.join([str(l) for l in np.where(c_y[i]!=0)[0]]) for i in range(c_y.shape[0])]

        return ctrl_X, gt_y, pred_y, ptb_targets, C_y
    
    preds = []
    for model_name in model_names:
        ctrl_X, gt_y, pred_y, ptb_targets, C_y = retrieve_single_run(model_name)
        preds.append(pred_y)

    ## generate plots
    all_data = np.vstack([ctrl_X, gt_y, np.vstack(preds)])
    adata_new = sc.AnnData(all_data)

    #plot them
    sc.tl.pca(adata_new, svd_solver='arpack')
    sc.pp.neighbors(adata_new, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata_new, min_dist=0.3) 

    ## create palettes
    palette_groups = sns.color_palette("husl", len(model_names))
    palette_colors = ['#%02x%02x%02x'%tuple((np.array(color)*255).astype(int)) for color in palette_groups]
    palette_dict = {'Actual Cells': 'blue'}
    palette_dict.update({f'Generated Cells_{mn}':pcolor for mn,pcolor in zip(model_names, palette_colors)})
    palette_dict.update({'NA': '#C0C0C08F'})

    #plot for each target
    for c in set(C_y):
        
        legend_groups = []
        for mn in model_names:
            legend_groups += [f'Generated Cells_{mn}' if C_y[i]==c else 'NA' for i in range(len(C_y))]

        label = ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' if C_y[i]==c else 'NA' for i in range(len(C_y))] + legend_groups
        adata_new.obs['label'] = label
        sizes = [30 if l != 'NA' else 15 for l in label]

        sc.pl.umap(adata_new, size=sizes, color=['label'], 
                legend_fontsize=14, groups=['Actual Cells'] + legend_groups, 
                title=ptb_targets[int(c)], 
                palette=palette_dict,
                #legend_loc=None,  
                save=f'_test_{ptb_targets[int(c)]}_CMVAE-obs_comparison.png'
                )

def visualize_data_generation_double_comparison(model_names = ['gosize5_orig_Adam'], single_val_names = []):
    
    def retrieve_single_run(model_name):

        #load data
        adata = sc.read_h5ad('./../../data/Norman2019_raw.h5ad')
        savedir = f'./../../result/{model_name}' 

        with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
            ptb_targets = pickle.load(f)

        ## load gosize=5 files
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..', '..', 'data', 'GO_to_ensembl_id_assignment_gosize5.csv'))

        # keep only GO_genes
        adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
        ctrl_X = adata.X.toarray()

        #load model
        model = torch.load(f'{savedir}/best_model.pt') 
        mode = 'CMVAE'

        ## generate test results
        rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_double(model, savedir, model.device, mode, temp=1)
        C_y = [','.join([str(l) for l in np.where(c_y[i]!=0)[0]]) for i in range(c_y.shape[0])]

        return ctrl_X, gt_y, pred_y, ptb_targets, C_y
    
    preds = []
    for model_name in model_names:
        ctrl_X, gt_y, pred_y, ptb_targets, C_y = retrieve_single_run(model_name)
        preds.append(pred_y)

    ## generate plots
    all_data = np.vstack([ctrl_X, gt_y, np.vstack(preds)])
    adata_new = sc.AnnData(all_data)

    #plot them
    sc.tl.pca(adata_new, svd_solver='arpack')
    sc.pp.neighbors(adata_new, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata_new, min_dist=0.3) 

    ## create palettes
    palette_groups = sns.color_palette("husl", len(model_names))
    palette_colors = ['#%02x%02x%02x'%tuple((np.array(color)*255).astype(int)) for color in palette_groups]
    palette_dict = {'Actual Cells': 'blue'}
    palette_dict.update({f'Generated Cells_{mn}':pcolor for mn,pcolor in zip(model_names, palette_colors)})
    palette_dict.update({'NA': '#C0C0C02F'})

    #plot for each target
    for c in set(C_y):

        # g1, g2 = c.split(',')
        # if ptb_targets[int(g1)] not in single_val_names and ptb_targets[int(g2)] not in single_val_names:
        #     continue
        
        legend_groups = []
        for mn in model_names:
            legend_groups += [f'Generated Cells_{mn}' if C_y[i]==c else 'NA' for i in range(len(C_y))]

        idx = c.split(',') 
        label = ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' if C_y[i]==c else 'NA' for i in range(len(C_y))] + legend_groups
        adata_new.obs['label'] = label
        sizes = [45 if l != 'NA' else 15 for l in label]

        sc.pl.umap(adata_new, size=sizes, color=['label'], 
                legend_fontsize=14, groups=['Actual Cells'] + legend_groups, 
                title=f'{ptb_targets[int(idx[0])]}+{ptb_targets[int(idx[1])]}', 
                palette=palette_dict,
                #legend_loc=None,  
                save=f'_test_{ptb_targets[int(idx[0])]}+{ptb_targets[int(idx[1])]}_CMVAE-obs_double_comparison.png'
                )


## single perturbations
single_val_names = ['FOXF1', 'PRDM1', 'HK2', 'RHOXF2', 'ZNF318', 'CEBPA', 'JUN', 'LHX1', 'CSRNP1', 'MAP7D1', 'CDKN1C', 'NIT1']
model_names = ['gosize5_orig_Adam', 'gosize5_sparse_unfreezed_NA_Adam', 'gosize5_sparse_unfreezed_vincenzo_Adam']

## single
#visualize_data_generation_comparison(model_names)

## double
visualize_data_generation_double_comparison(model_names)