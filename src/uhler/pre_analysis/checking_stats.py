import pandas as pd
import os
import utils as ut


## load data
adata, double_adata, ptb_targets, ptb_targets_affected, gos, rel_dict, gene_go_dict, ensembl_genename_mapping_rev = ut.load_norman_2019_dataset(num_gene_th=5)


#check number of genes
print(f"Number of genes used: {len(adata.var_names)}")

#check number of single perturbations (remove control)
print(f"Number single perturbations used: {adata.obs['guide_ids'].unique().shape[0]-1} ") 

#check number of double perturbations
print(f"Number single perturbations used: {double_adata.obs['guide_ids'].unique().shape[0]} ") 

#check number of knockout genes that have at least one affected geneset
print(f"Number of knockout genes with at least one affected geneset: {len(ptb_targets_affected)}")

#number of ctrl cells
print(f"Number of control cells: {adata[adata.obs['guide_ids'] == ''].shape[0]}")

#number of perturbed cells
print(f"Number of single-gene perturbed cells: {adata[adata.obs['guide_ids'] != ''].shape[0]}")
print(f"Number of double-gene perturbed cells: {double_adata.shape[0]}")