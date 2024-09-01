import numpy as np
import pickle
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from warnings import filterwarnings
from tqdm import tqdm
import os
filterwarnings('ignore')
sys.path.append('./../src')
import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from inference import *
from utils import SCDATA_sampler, MMD_loss
from dataset import SCDataset
from torch.utils.data import DataLoader

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

def compute_metrics(model_name = 'full_go', seed = 42):
	
    # read different trained models here
    savedir = f'./../../../result/uhler/{model_name}/seed_{seed}' 

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    model = torch.load(f'{savedir}/best_model.pt')

    # set mode to the corresponding model/inference type
    mode = 'CMVAE'

    def evaluate_model(fold):

        if fold == 'test':
            rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_single_leftout(model, savedir, model.device, mode)
        elif fold == 'train':
            rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_single_train(model, savedir, model.device, mode)
        elif fold == 'double':
            rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var = evaluate_double(model, savedir, model.device, mode, temp=1)

        C_y = [','.join([str(l) for l in np.where(c_y[i]!=0)[0]]) for i in range(c_y.shape[0])]

        # compute statistics
        mmd_loss = {}
        rmse_loss = {}
        r2 = {}

        loss = MMD_loss(fix_sigma=1000, kernel_num=10)
        for i in range(len(C_y)//32):
            y = torch.from_numpy(gt_y[i*32:(i+1)*32])
            y_hat = torch.from_numpy(pred_y[i*32:(i+1)*32])
            c = C_y[i*32]
            if c in mmd_loss.keys():
                mmd_loss[c].append(loss(y_hat, y).item())
                rmse_loss[c].append(np.sqrt(np.mean(((pred_y[i*32:(i+1)*32] - gt_y[i*32:(i+1)*32])**2)) / np.mean(((gt_y[i*32:(i+1)*32])**2))))
                r2[c].append(max(r2_score(np.mean(pred_y[i*32:(i+1)*32] , axis=0),np.mean(gt_y[i*32:(i+1)*32], axis=0)),0))
            else:
                mmd_loss[c] = [loss(y_hat, y).item()] 
                rmse_loss[c] = [np.sqrt(np.mean(((pred_y[i*32:(i+1)*32] - gt_y[i*32:(i+1)*32])**2)) / np.mean(((gt_y[i*32:(i+1)*32])**2)))]
                r2[c] = [max(r2_score(np.mean(pred_y[i*32:(i+1)*32] , axis=0),np.mean(gt_y[i*32:(i+1)*32], axis=0)),0)]
        
        # summarize statistics
        mmd_loss_summary = {}
        rmse_loss_summary = {}
        r2_summary = {}
        for k in tqdm(mmd_loss.keys()):
            #print(ptb_targets[int(k)], np.average(mmd_loss[k]), np.std(mmd_loss[k]))
            mmd_loss_summary[k] = (np.average(mmd_loss[k]), np.std(mmd_loss[k]))
            rmse_loss_summary[k] = (np.average(rmse_loss[k]), np.std(mmd_loss[k]))
            r2_summary[k] = (np.average(r2[k]), np.std(r2[k]))
            
        mmd_mean = np.mean([i[0] for i in mmd_loss_summary.values()])
        mmd_std = np.std([i[0] for i in mmd_loss_summary.values()])/np.sqrt(len(mmd_loss_summary.keys()))
        print(f"{fold} mmd {round(mmd_mean,4)} +- {round(mmd_std,4)}")

        r2_mean = np.mean([i[0] for i in r2_summary.values()])
        r2_std = np.std([i[0] for i in r2_summary.values()])/np.sqrt(len(rmse_loss_summary.keys()))
        print(f"{fold} r2 {round(r2_mean,4)} +- {round(r2_std,4)}")

        rmse_mean = np.mean([i[0] for i in rmse_loss_summary.values()])
        rmse_std = np.std([i[0] for i in rmse_loss_summary.values()])/np.sqrt(len(rmse_loss_summary.keys()))
        print(f"{fold} rmse {round(rmse_mean,4)} +- {round(rmse_std,4)}")

        # Prepare data in the format mean ± std
        data = {
            'Metric': ['mmd', 'r2', 'rmse'],
            'Values': [
                f"{round(mmd_mean, 4)} ± {round(mmd_std, 4)}",
                f"{round(r2_mean, 4)} ± {round(r2_std, 4)}",
                f"{round(rmse_mean, 4)} ± {round(rmse_std, 4)}"
            ]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        return df

    ## train 
    print("train metrics")
    df_train = evaluate_model(fold='train')
    df_train['mode'] = 'train'

    ## test
    print("test metrics")
    df_test = evaluate_model(fold='test')
    df_test['mode'] = 'test'

    ## doubles
    print("double metrics")
    df_double = evaluate_model(fold="double")
    df_double['mode'] = 'double'

    ## concat
    df = pd.concat([df_train, df_test, df_double]).reset_index(drop=True)
    print(df)
    df.to_csv(os.path.join('./../../../','result', 'uhler', model_name, f'seed_{seed}', f'{model_name}_mmd_r2_rmse_metrics_summary.tsv'),sep='\t')

def visualize_gradients(model_name = 'full_go'):

    # read different trained models here
    savedir = f'./../../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    fpath = os.path.join('./../../../','figures','uhler_paper',model_name)
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    def plot_layer_weights(layer_name, model, fpath):

        try:

            ## get non-zero gradients
            non_masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) != 0].detach().cpu().numpy()')
            masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) == 0].detach().cpu().numpy()')

            ## Plotting the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(non_masked_gradients, bins=30, alpha=0.5, label='Non-masked values')
            plt.hist(masked_gradients, bins=30, alpha=0.5, label='Masked values')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Layer {layer_name} weights')
            plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

        except:

            ##
            gradients = eval(f'model.{layer_name}.weight.detach().cpu().numpy().flatten()')

            ## Plotting the histogram
            plt.figure(figsize=(10, 6))
            plt.hist(gradients, bins=30, alpha=0.5, label = f'layer {layer_name}', color='blue')
            plt.yscale('log')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Layer {layer_name} weights')
            plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

    def plot_weight_heatmap(layer_name, model, fpath):

        ##get the output of NetActivity Layer
        #batch_size, mode = 128, 'train'
        #dataloader, _, _, _, ptb_targets = get_data(batch_size=batch_size, mode=mode)
        adata, _, gos, zs = load_data_raw_go(ptb_targets)

        weight_mat = eval(f'model.{layer_name}.weight.detach().cpu().numpy()').T
        weight_df = pd.DataFrame(weight_mat)

        if layer_name == 'fc1':

            weight_df.index = adata.var['gene_symbols']
            weight_df.columns = gos
           
        elif layer_name == 'fc_mean' or layer_name == 'fc_var':# or layer_name == 'z':

            weight_df.index = gos
            weight_df.columns = zs

        ## Plotting the histogram
        plt.figure(figsize=(25, 150))
        sns.heatmap(weight_df.abs(), cmap='coolwarm', center=0, annot=False, linewidths=.5)
        plt.title(f'Heatmap of Gene Weights - Layer {layer_name}')
        plt.xlabel('GO Terms')
        plt.ylabel('Gene Symbols')
        plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_heatmap.png'))
        
    ## hist
    ##plot_layer_weights(layer_name='fc1', model=model, fpath=fpath)
    plot_layer_weights(layer_name='fc_mean', model=model, fpath=fpath)
    plot_layer_weights(layer_name='fc_var', model=model, fpath=fpath)

    ## heatmap
    #plot_weight_heatmap(layer_name='fc_mean', model=model, fpath=fpath)
    #plot_weight_heatmap(layer_name='fc_var', model=model, fpath=fpath)

## model name
model_name = 'full_go_sena_delta_0'
seed = 13

## compute metrics
compute_metrics(model_name = model_name, seed = seed)

## visualize gradients
##visualize_gradients(model_name = model_name) 