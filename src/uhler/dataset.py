import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import os
import pandas as pd


# read the norman dataset.
# map the target genes of each cell to a binary vector, using a target gene list "perturb_targets".
# "perturb_type" specifies whether the returned object contains single trarget-gene samples, double target-gene samples, or both.
class SCDataset(Dataset):
    def __init__(self, datafile='./../../data/Norman2019_raw.h5ad', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double', 'both'], 'perturb_type not supported!'

        #load adata
        adata = sc.read_h5ad(datafile)
               
        # load gos from NA paper
        ensembl_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
        ensembl_genename_mapping = dict(zip(ensembl_genename_mapping.iloc[:,0], ensembl_genename_mapping.iloc[:,1]))
        GO_to_ensembl_id_assignment = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        GO_to_ensembl_id_assignment.columns = ['GO_id','ensembl_id']

        #load GOs
        go_2_z_raw = pd.read_csv(os.path.join('..','..','data','topGO_Jesus.tsv'),sep='\t')
        GO_to_ensembl_id_assignment = GO_to_ensembl_id_assignment[GO_to_ensembl_id_assignment['GO_id'].isin(go_2_z_raw['PathwayID'].values)]

        ## load interventions
        print(f"zs: {set(go_2_z_raw['topGO'])}, number of zs : {len(set(go_2_z_raw['topGO']))}")
        intervention_genenames = map(lambda x: ensembl_genename_mapping.get(x,None), GO_to_ensembl_id_assignment['ensembl_id'])
        ptb_targets = list(set(intervention_genenames).intersection(set([x for x in adata.obs['guide_ids'] if x != '' and ',' not in x])))
        
        # ## keep only GO_genes
        print(f"number of intervention genes: {ptb_targets}")
        adata = adata[:, adata.var_names.isin(GO_to_ensembl_id_assignment['ensembl_id'])]
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

            ptb_adata = adata[(adata.obs['guide_ids'].str.contains(',')) & (adata.obs['guide_ids']!='')].copy()

            #keep only cells containing our perturbed genes
            ptb_adata = ptb_adata[ptb_adata.obs['guide_ids'].apply(lambda x: all([y in ptb_targets for y in x.split(',')])), :]

            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(self.ptb_targets, ptb_adata.obs['guide_ids'].values)
            

        else:

            ptb_adata = adata[adata.obs['guide_ids']!=''].copy() 
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(self.ptb_targets, ptb_adata.obs['guide_ids'].values)
            

        self.ctrl_samples = adata[adata.obs['guide_ids']==''].X.copy()
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
            ]
        self.adata = adata

    def __getitem__(self, item):
        x = torch.from_numpy(self.rand_ctrl_samples[item].toarray().flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].toarray().flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c
    
    def __len__(self):
        return self.ptb_samples.shape[0]


# read simulation dataset
class SimuDataset(Dataset):
    def __init__(self, datafile='/home/jzhang/discrepancy_vae/identifiable_causal_vae/data/simulation/data_1.pkl', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double'], 'perturb_type not supported!'

        with open(datafile, 'rb') as f:
            dataset = pickle.load(f)

        if perturb_targets is None:
            ptb_targets = dataset['ptb_targets']
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets

        
        ptb_data = dataset[perturb_type]
        self.ctrl_samples = ptb_data['X']
        self.ptb_samples = ptb_data['Xc']
        self.ptb_names = np.array(ptb_data['ptbs'])
        self.ptb_ids = map_ptb_features(ptb_targets, ptb_data['ptbs'])
        del ptb_data 

        self.nonlinear = dataset['nonlinear']
        del dataset

    def __getitem__(self, item):
        x = torch.from_numpy(self.ctrl_samples[item].flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].flatten()).double()
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



