import numpy as np
import pickle
import torch
import pandas as pd 
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import sys
import logging
import json
from model import CMVAE, NetworkActivity_layer
from utils import Norman2019DataLoader, Wessel2023HEK293DataLoader
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generating_data(config_file, fpath,  batch_size = 32):

    ## detect device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # define parameters
    dataset_name = config_file["dataset_name"]

    logging.info(f"Loading {dataset_name} dataset")
    if 'Norman2019' in dataset_name:
        data_handler = Norman2019DataLoader(batch_size=batch_size)
        data_handler.gene_var = "guide_ids"
    elif dataset_name == 'wessel_hefk293':
        data_handler = Wessel2023HEK293DataLoader(batch_size=batch_size)
        data_handler.gene_var = 'TargetGenes'

    #load data and reset index
    adata = data_handler.adata
    adata.obs = adata.obs.reset_index(drop=True)
    ptb_targets = data_handler.ptb_targets
    gos = data_handler.gos

    """build pert idx dict"""
    idx_dict = {}
    for knockout in ['ctrl'] + ptb_targets:
        if knockout == 'ctrl':
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == '']).index.values
        else:
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == knockout]).index.values

    """load best model"""
    #load weights
    model = torch.load(f'{fpath}/best_model.pt')

    ##
    n_pertb = len(ptb_targets)
    pert_dict = {}
    info_dict = defaultdict(lambda: defaultdict(list))
    results_dict = {}


    """compute"""
    with torch.no_grad():

        for gene in tqdm(idx_dict, desc = 'generating activity score for perturbations'):
            
            idx = idx_dict[gene]
            mat = torch.from_numpy(adata.X[idx,:].todense()).to(device).double()

            """first layer"""

            na_score_fc1 = model.fc1(mat)
            info_dict['fc1'][gene].append(na_score_fc1.detach().cpu().numpy())

            """mean + var"""

            na_score_fc_mean = model.fc_mean(na_score_fc1)
            info_dict['fc_mean'][gene].append(na_score_fc_mean.detach().cpu().numpy())

            na_score_fc_var = torch.nn.Softplus()(model.fc_var(na_score_fc1))
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
                c = c.to(device).double()

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
        if layer == 'fc1':
            results_dict[layer].columns = gos

    #add pertb_dict
    results_dict['pert_map'] = pd.DataFrame(pert_dict, index = [0]).T
    results_dict['pert_map'].columns = ['c_enc_mapping']
    results_dict['causal_graph'] = model.G.detach().cpu().numpy()

    """add weights layers (delta) for """
    results_dict['mean_delta_matrix'] = pd.DataFrame(model.fc_mean.weight.detach().cpu().numpy().T, index = gos) 
    results_dict['std_delta_matrix'] = pd.DataFrame(model.fc_var.weight.detach().cpu().numpy().T, index = gos) 

    """save info"""
    with open(os.path.join(fpath, 'activation_scores.pickle'), 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process configuration and generate data.")
    parser.add_argument("folder_name", type=str, nargs='?', default="example", help="The name of the folder in the results directory")
    args = parser.parse_args()

    logging.info(f"Folder name {args.folder_name} selected")
    
    # Define fpath
    fpath = os.path.join(os.getcwd(), 'results', args.folder_name)

    #get dataset_name
    with open(os.path.join(fpath,'config.json'), 'r') as file:
        config_file = json.load(file)
    
    #generate pickle
    generating_data(config_file, fpath)

