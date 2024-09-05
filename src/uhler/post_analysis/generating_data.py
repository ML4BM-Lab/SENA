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
import scipy.stats as stats
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import ttest_ind
import numpy as np
from utils import get_data
from collections import Counter
from scipy.stats import gaussian_kde
import utils as ut


"""load data"""
def load_data(model_name, seed):

    def load_data_raw_go(ptb_targets):

        #define url
        datafile='./../../data/Norman2019_raw.h5ad'
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
        for knockout in ['ctrl'] + ptb_targets:

            if knockout == 'ctrl':
                perturbations_idx_dict[knockout] = (adata.obs[adata.obs['guide_ids'] == '']).index.values
            else:
                perturbations_idx_dict[knockout] = (adata.obs[adata.obs['guide_ids'] == knockout]).index.values

        return adata, perturbations_idx_dict, sorted(set(go_2_z_raw['PathwayID'].values)), sorted(set(go_2_z_raw['topGO'].values)) 

    #load model
    savedir = f'./../../result/uhler/{model_name}/seed_{seed}' 
    model = torch.load(f'{savedir}/best_model.pt')

    ##get the output of NetActivity Layer
    batch_size, mode = 128, 'train'
    _, _, _, _, _, ptb_targets = get_data(batch_size=batch_size, mode=mode)

    adata, idx_dict, gos, zs = load_data_raw_go(ptb_targets)
    return model, adata, idx_dict, gos, zs, ptb_targets


##
model_name = 'full_go_regular'
seed=7
model, adata, idx_dict, gos, zs, ptb_targets = load_data(model_name, seed=seed)
device = "cuda:0"

##
n_pertb = len(ptb_targets)
pert_dict = {}
info_dict = defaultdict(lambda: defaultdict(list))
results_dict = {}


"""compute"""
with torch.no_grad():

    for gene in tqdm(idx_dict, desc = 'generating activity score for perturbations'):
        
        idx = idx_dict[gene]
        mat = torch.from_numpy(adata.X[idx,:].todense()).to('cuda').double()

        """first layer"""

        na_score_fc1 = model.fc1(mat)
        info_dict['fc1'][gene].append(na_score_fc1.detach().cpu().numpy())

        """mean + var"""

        na_score_fc_mean = model.fc_mean(na_score_fc1)
        info_dict['fc_mean'][gene].append(na_score_fc_mean.detach().cpu().numpy())

        na_score_fc_var = model.fc_var(na_score_fc1)
        info_dict['fc_var'][gene].append(na_score_fc_var.detach().cpu().numpy())

        """reparametrization trick"""

        na_score_mu, na_score_var = model.encode(mat)
        na_score_z = model.reparametrize(na_score_mu, na_score_var)
        info_dict['z'][gene].append(na_score_z.detach().cpu().numpy())

        """causal graph"""

        if gene != 'ctrl':

            #define ptb idx
            ptb_idx = np.where(np.array(ptb_targets)==gene)[0][0]

            #generate one-hot-encoder
            c = torch.zeros(size=(1,n_pertb))
            c[:,ptb_idx] = 1
            c = c.to('cuda').double()

            # decode an interventional sample from an observational sample    
            bc, csz = model.c_encode(c, temp=1)
            bc2, csz2 = bc, csz
            info_dict['bc_temp1'][gene].append(bc.detach().cpu().numpy())

            # decode an interventional sample from an observational sample    
            bc, csz = model.c_encode(c, temp=100)
            bc2, csz2 = bc, csz
            info_dict['bc_temp100'][gene].append(bc.detach().cpu().numpy())

            # decode an interventional sample from an observational sample    
            bc, csz = model.c_encode(c, temp=1000)
            bc2, csz2 = bc, csz
            info_dict['bc_temp1000'][gene].append(bc.detach().cpu().numpy())

            # compute assignation
            if ptb_idx not in pert_dict:
                pert_dict[ptb_idx] = bc[0].argmax().__int__()

            #interventional U
            na_score_u = model.dag(na_score_z, bc, csz, bc2, csz2, num_interv=1)
            info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

        else:

            #observational U
            na_score_u = model.dag(na_score_z, 0, 0, 0, 0, num_interv=0)
            info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

"""build dataframes within each category"""
for layer in info_dict:

    temp_df = []
    for gene in info_dict[layer]:
        info_dict[layer][gene] = pd.DataFrame(np.vstack(info_dict[layer][gene]))
        info_dict[layer][gene].index = [gene] * info_dict[layer][gene].shape[0]
        temp_df.append(info_dict[layer][gene])
    
    #substitute
    results_dict[layer] = pd.concat(temp_df)

#add pertb_dict
results_dict['pert_map'] = pd.DataFrame(pert_dict, index = [0]).T
results_dict['pert_map'].columns = ['c_enc_mapping']

"""save info"""

with open(f'./../../result/uhler/{model_name}/seed_{seed}/post_analysis_mlp_seed_7.pickle' , 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)