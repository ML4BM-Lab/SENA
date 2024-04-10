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
sys.path.append('../src')

import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from inference import *
from utils import SCDATA_sampler, MMD_loss
from dataset import SCDataset
from torch.utils.data import DataLoader


def compute_metrics(model_name = 'gosize5'):
	
    # read different trained models here
    savedir = f'./../../result/{model_name}' 

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

    ## train 
    print("train metrics")
    evaluate_model(fold='train')

    ## test
    print("test metrics")
    evaluate_model(fold='test')

def visualize_gradients(model_name = 'gosize5'):

    # read different trained models here
    savedir = f'./../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    # with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
    #     ptb_targets = pickle.load(f)

    def plot_layer_weights(layer_name):

        ## get non-zero gradients
        try:
            non_masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) != 0].detach().cpu().numpy()')
            masked_gradients = eval(f'model.{layer_name}.weight[(model.{layer_name}.weight * model.{layer_name}.mask.T) == 0].detach().cpu().numpy()')
        except:
            non_masked_gradients = eval(f'model.{layer_name}.weight[model.{layer_name}.weight != 0].detach().cpu().numpy()')
            masked_gradients = eval(f'model.{layer_name}.weight[model.{layer_name}.weight == 0].detach().cpu().numpy()')


        ## Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(non_masked_gradients, bins=30, alpha=0.5, label='Non-masked values')
        plt.hist(masked_gradients, bins=30, alpha=0.5, label='Masked values')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Layer {layer_name} weights')
        plt.show()
        plt.savefig(os.path.join('./../..','figures',f'{model_name}_layer_{layer_name}_histplot.png'))

    plot_layer_weights(layer_name='fc1')
    plot_layer_weights(layer_name='fc_mean')
    plot_layer_weights(layer_name='fc_var')

## model name
model_name = 'gosize5_sparse_unfreezed_vincenzo_Adam'

## compute metrics
compute_metrics(model_name = model_name)

## visualize gradients
visualize_gradients(model_name = model_name) 