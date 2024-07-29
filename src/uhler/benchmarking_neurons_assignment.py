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

## load our model
mode_type = 'full_go'
trainmode = 'NA_NA'
model_name = f'{mode_type}_{trainmode}'

"""
plot layer weights
"""

def plot_layer_weights(layer_name):

    # read different trained models here
    fpath = os.path.join('./../../figures','uhler_paper',f'{mode_type}_{trainmode}')
    savedir = f'./../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    ## get non-zero gradients
    gradients = eval(f'model.{layer_name}.weight.detach().cpu().numpy()')
   
    ## Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(gradients.flatten(), alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Layer {layer_name} weights')
    plt.savefig(os.path.join(fpath, f'{model_name}_layer_{layer_name}_histplot.png'))

#plot_layer_weights(layer_name = 'fc_mean')
#plot_layer_weights(layer_name = 'fc_var')

"""
analyze the latent factor relationship to perturbation
"""

def analyze_latent_factor_relationship(layer_name, only_analysis=False):

    def compute_IAS(heatmap_logfc_data, metric = 'max'):   

        outliers_distr = []
        for gene in heatmap_logfc_data.index:
            
            #get distr scores
            distr = heatmap_logfc_data.loc[gene].values

            #compute z-score
            zscore = distr - distr.mean() / distr.std()
            
            #check if its normal
            shapiro_test = stats.shapiro(zscore).pvalue
            
            ##kde
            kde = gaussian_kde(zscore)
            
            for outlier in sorted(zscore, reverse=True):
                
                #compute the pvalue of getting this value or more extreme
                outlier_pval_gauss = 1 - stats.norm.cdf(outlier, loc=0, scale=1)
                outlier_pval_kde = kde.evaluate(outlier)[0]

                if shapiro_test >= 0.001:
                    outlier_pval = outlier_pval_gauss
                else:
                    outlier_pval = outlier_pval_kde
                
                ##
                if outlier_pval <= 0.05:
                    outliers_distr.append([gene,outlier_pval])
                else:
                    break

        try:

            df = pd.DataFrame(outliers_distr)
            df.columns = ['gene','outlier_pval']

            """
            compute the outlier_activation metric
            """
            num_tot_interventions = heatmap_logfc_data.shape[0]
            freq_interventions = sum([(heatmap_logfc_data.shape[1] - x) / (heatmap_logfc_data.shape[1]-1) for x in df.groupby('gene').apply(lambda x: x.shape[0]).values])
            outlier_activation_metric = (1/num_tot_interventions) * freq_interventions
        
            return outlier_activation_metric

        except:
            return 0

    #load activity scores
    fpath = os.path.join('./../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv')
    na_activity_score = pd.read_csv(fpath,sep='\t',index_col=0)

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score['type'] == 'ctrl']

    ## init df
    ttest_df = []

    for knockout in tqdm(set(na_activity_score['type'])):
    
        if knockout != 'ctrl':

            #get knockout cells
            knockout_cells = na_activity_score[na_activity_score['type']  == knockout]

            for geneset in na_activity_score.columns[:-1]:

                #perform ttest
                _, p_value = ttest_ind(ctrl_cells.loc[:,geneset].values, knockout_cells.loc[:,geneset].values, equal_var=False)
                
                ## abs(logFC)
                knockout_mean = knockout_cells.loc[:,geneset].values.mean() 
                ctrl_mean = ctrl_cells.loc[:,geneset].values.mean()

                if (not ctrl_mean) or (not knockout_mean):
                    abslogfc = 0
                else:
                    abslogfc = np.abs(np.log(np.abs(knockout_mean/ctrl_mean)))
            
                #append info
                ttest_df.append([knockout, geneset, p_value, abslogfc])
                
    ## build df
    ttest_df = pd.DataFrame(ttest_df)
    ttest_df.columns = ['knockout','geneset','pval', 'abslogfc']
    ttest_df = ttest_df.sort_values(by=['knockout','geneset']).reset_index(drop=True)

    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = ttest_df.pivot(index="knockout", columns="geneset", values="pval")
    heatmap_data = heatmap_data.dropna(axis=1)
    heatmap_logfc_data = ttest_df.pivot(index="knockout", columns="geneset", values="abslogfc")
    heatmap_logfc_data = heatmap_logfc_data.dropna(axis=1)

    ## compute outlier df
    outlier_metric = compute_IAS(heatmap_logfc_data)

    ## check if only analysis is required
    if only_analysis:
        return outlier_metric
    
    heatmap_logfc_data = (heatmap_logfc_data.T/heatmap_logfc_data.max(axis=1)).T

    """
    save log-scaled heatmap
    """
    log_heatmap_data = -np.log10(heatmap_data)
    log_heatmap_data = (log_heatmap_data.T/log_heatmap_data.max(axis=1)).T
    fpath = os.path.join('./../../result',f'{mode_type}_{trainmode}')
    log_heatmap_data.to_csv(os.path.join(fpath, f'activation_scores_DEA_layer_{layer_name}_matrix.tsv'), sep='\t')
    heatmap_logfc_data.to_csv(os.path.join(fpath, f'activation_scores_logFC_DEA_layer_{layer_name}_matrix.tsv'), sep='\t')
    

##
analyze_latent_factor_relationship(layer_name = 'fc_mean', only_analysis=True)
analyze_latent_factor_relationship(layer_name = 'fc_var', only_analysis=True)
    
analyze_latent_factor_relationship(layer_name = 'fc1', only_analysis=True)
analyze_latent_factor_relationship(layer_name = 'z', only_analysis=True)
