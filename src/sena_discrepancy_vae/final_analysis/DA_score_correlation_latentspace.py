import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import utils as ut
import scipy

def compute_activation_df(na_activity_score, scoretype, gos, mode, gene_go_dict, genename_ensemble_dict, ptb_targets):

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score.index == 'ctrl'].to_numpy()

    ## init df
    ttest_df = []

    for knockout in tqdm(ptb_targets):

        if knockout not in genename_ensemble_dict:
            continue
        
        #get knockout cells       
        knockout_cells = na_activity_score[na_activity_score.index == knockout].to_numpy()

        #compute affected genesets
        if mode[:4] == 'sena':
            belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[genename_ensemble_dict[knockout]]] 

        for i, geneset in enumerate(gos):
            
            if scoretype == 'mu_diff':
                score = abs(ctrl_cells[:,i].mean() - knockout_cells[:,i].mean())

            #append info
            if mode[:4] == 'sena':
                ttest_df.append([knockout, geneset, scoretype, score, geneset in belonging_genesets])
            elif mode[:7] == 'regular' or mode[:2] == 'l1':
                ttest_df.append([knockout, i, scoretype, score, i in belonging_genesets])

    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout', 'geneset', 'scoretype', 'score', 'affected']

    return ttest_df

""" compute geneset correlation """
def compute_geneset_correlation_latent_space_sena():

    ##
    dataset = 'full_go'
    mode = 'sena_delta_0'
    model_name = f'{dataset}_{mode}'
    seed = 42
    latdim = 105

    #load summary file
    with open(os.path.join(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle'), 'rb') as handle:
        model_summary = pickle.load(handle)

    #load model
    savedir = f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}' 
    model = torch.load(f'{savedir}/best_model.pt')

    #activation layer
    _, _, ptb_targets_all, ptb_targets_affected, gos, rel_dict, gene_go_dict, genename_ensemble_dict = utils.load_norman_2019_dataset()
    fc1 = model_summary['fc1']
    mean = model_summary['fc_mean']
    var = model_summary['fc_var']

    #compute DA by geneset at the output of the SENA layer
    DA_df_by_geneset_fc1 = compute_activation_df(fc1, scoretype = 'mu_diff', 
                                                    gos=gos, mode=mode, 
                                                    gene_go_dict=gene_go_dict, genename_ensemble_dict=genename_ensemble_dict,
                                                    ptb_targets=ptb_targets_affected)

    adata, _, _, _, _, _, _, ensembl_genename_mapping_rev = ut.load_norman_2019_dataset()

    def compute_geneset_correlation_ls():

        ctrl_cells_fc1 = fc1[fc1.index == 'ctrl'].to_numpy().mean(axis=0)

        contribution_list = []
        for knockout in tqdm(ptb_targets_affected):
            
            #get original mean expression
            ctrl_mean_exp = pd.DataFrame(adata[adata.obs['guide_ids'] == ''].X.todense(), columns = adata.var_names).mean(axis=0)
            knockout_mean_exp = pd.DataFrame(adata[adata.obs['guide_ids'] == knockout].X.todense(), columns = adata.var_names).mean(axis=0)

            """get the specific expression value"""
            ctrl_input_gene_exp = ctrl_mean_exp.loc[ensembl_genename_mapping_rev[knockout]]
            knockout_input_gene_exp = knockout_mean_exp.loc[ensembl_genename_mapping_rev[knockout]]
            DAR_input = np.abs(knockout_input_gene_exp/ctrl_input_gene_exp)

            """compute contribution to fc1"""
            affected_genesets = np.array([i for i,geneset in enumerate(gos) if geneset in gene_go_dict[genename_ensemble_dict[knockout]]])

            ## get knockout
            DA_df_by_geneset_fc1_knockout = DA_df_by_geneset_fc1[DA_df_by_geneset_fc1['knockout'] == knockout]

            ctrl_affected_fc1_mean = ctrl_cells_fc1[affected_genesets].mean()
            knockout_affected_fc1 = pd.DataFrame(DA_df_by_geneset_fc1_knockout['score'].values, index = DA_df_by_geneset_fc1_knockout['geneset']).T[gos].T.values.flatten()
            knockout_affected_fc1_mean = knockout_affected_fc1[affected_genesets].mean()
            DAR_fc1 = np.abs(knockout_affected_fc1_mean/ctrl_affected_fc1_mean)
            
            # """same for fc_mean"""
            # ctrl_fc_mean_contribution = ctrl_cells_fc1[affected_genesets] @ model.fc_mean.weight.T.detach().cpu().numpy()[affected_genesets]
            # knockout_fc_mean_contribution = knockout_affected_fc1[affected_genesets] @ model.fc_mean.weight.T.detach().cpu().numpy()[affected_genesets]

            # DAR_fc_mean = np.abs(knockout_fc_mean_contribution.mean()/ctrl_fc_mean_contribution.mean())

            # """same for fc_var"""
            # ctrl_fc_var_contribution = ctrl_fc1_contribution @ model.fc_var.weight.T.detach().cpu().numpy()
            # knockout_fc_var_contribution = knockout_fc1_contribution @ model.fc_var.weight.T.detach().cpu().numpy()


            ##build dataframe
            contribution_list.append([DAR_input, DAR_fc1])
            
        contributions_df = pd.DataFrame(contribution_list)
        contributions_df.columns = ['input_space','fc1']
       
        return contributions_df

    # mean & var
    df_correlation = compute_geneset_correlation_ls()
    _, corr = scipy.stats.pearsonr(df_correlation['input_space'], df_correlation['fc1'])

    return corr

def compute_geneset_correlation_latent_space_original(ptb_targets_affected):

    ##
    dataset = 'full_go'
    mode = 'regular'
    model_name = f'{dataset}_{mode}'
    seed = 42
    latdim = 105

    #load summary file
    with open(os.path.join(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle'), 'rb') as handle:
        model_summary = pickle.load(handle)

    fc1 = model_summary['fc1']
    fc_mean = model_summary['fc_mean']
    fc_var = model_summary['fc_var']

    adata, _, _, _, _, _, _, ensembl_genename_mapping_rev = ut.load_norman_2019_dataset()
    contribution_list = []
    for knockout in tqdm(ptb_targets_affected):

        #get original mean expression
        ctrl_mean_exp = pd.DataFrame(adata[adata.obs['guide_ids'] == ''].X.todense(), columns = adata.var_names).mean(axis=0)
        knockout_mean_exp = pd.DataFrame(adata[adata.obs['guide_ids'] == knockout].X.todense(), columns = adata.var_names).mean(axis=0)

        """get the specific expression value"""
        ctrl_input_gene_exp = ctrl_mean_exp.loc[ensembl_genename_mapping_rev[knockout]]
        knockout_input_gene_exp = knockout_mean_exp.loc[ensembl_genename_mapping_rev[knockout]]
        DAR_input = np.abs(knockout_input_gene_exp/ctrl_input_gene_exp)
        
        DAR_fc1 = np.abs(fc1[fc1.index == knockout].mean(axis=0).mean()/fc1[fc1.index == 'ctrl'].mean(axis=0).mean())
        DAR_fc_mean = np.abs(fc_mean[fc_mean.index == knockout].mean(axis=0).mean()/fc_mean[fc_mean.index == 'ctrl'].mean(axis=0).mean())
        DAR_fc_var = np.abs(fc_var[fc_var.index == knockout].mean(axis=0).mean()/fc_var[fc_var.index == 'ctrl'].mean(axis=0).mean())

        ##build dataframe
        contribution_list.append([DAR_input, DAR_fc1, DAR_fc_mean, DAR_fc_var])
            
    contributions_df = pd.DataFrame(contribution_list)
    contributions_df.columns = ['input_space','fc1', 'fc_mean','fc_var']

    # mean & var
    _, corr_fc1 = scipy.stats.pearsonr(contributions_df['input_space'], contributions_df['fc1'])
    _, corr_fc_mean = scipy.stats.pearsonr(contributions_df['input_space'], contributions_df['fc_mean'])
    _, corr_fc_var = scipy.stats.pearsonr(contributions_df['input_space'], contributions_df['fc_var'])

    return corr_fc1, corr_fc_mean, corr_fc_var