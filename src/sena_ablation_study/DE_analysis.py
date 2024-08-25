import sena_tools as st
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

#load norman dataset
adata, ctrl_samples, ko_samples, ptb_targets, gos, rel_dict = st.load_norman_2019_dataset()

#genes
def retrieve_direct_genes(adata):
    genes = adata.var_names
    ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
    ensembl_genename_mapping_dict = dict(zip(ensembl_genename_mapping.iloc[:,1], ensembl_genename_mapping.iloc[:,0]))
    ensembl_genename_mapping_revdict = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
    ptb_targets_ens = list(map(lambda x: ensembl_genename_mapping_dict[x], ptb_targets))
    ptb_targets_direct = [ensembl_genename_mapping_revdict[x] for x in ptb_targets_ens if x in genes]
    return ptb_targets_direct

#perform gene set enrichment analysis
intervention_types = list(adata.obs['guide_ids'].values.unique())

knockout = 'COL1A1'
knockout_ens = 'ENSG00000184486'
knockout_ens in adata.var_names
idx = np.where(ctrl_samples.var_names == knockout_ens)[0][0]

ctrl_samples.X[:,idx].todense().mean()
ko_samples[ko_samples.obs['guide_ids'] ==knockout].X[:,idx].todense().mean()

for knockout in tqdm(intervention_types):
    
    if knockout != '':

        #get knockout cells
        cells_exp = adata[adata.obs['guide_ids']  == knockout].X.todense()

        for geneset in gos:

            #perform ttest
            _, p_value = ttest_ind(ctrl_cells[geneset], knockout_cells[geneset], equal_var=False)
            
            #append info
            ttest_df.append([knockout, geneset, p_value])