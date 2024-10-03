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
import utils as ut
from collections import defaultdict
import numpy as np
from collections import Counter
from scipy.stats import ttest_ind

def compute_scores_knockout_analysis(target_knockout = "CDKN1A"):

    def load_model(mode_type = 'raw_go', trainmode = 'NA+deltas', layertype = 'genesets', layer_name = 'fc1'):

        ##load our model
        model_name = f'{mode_type}_{trainmode}'
        savedir = f'./../../../result/{model_name}' 
        model = torch.load(f'{savedir}/best_model.pt')
        
        return model

    def get_affected_zs_and_genesets():

        rel_latent_dict = ut.build_gene_go_relationships_latent_deltas(gos)

        ## load genesets-genes mapping
        db_gene_go_map = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','go_kegg_gene_map.tsv'),sep='\t')
        gene_go_dict = defaultdict(list)
        for go,ens in tqdm(db_gene_go_map.values):
            gene_go_dict[ens].append(go)

        ##get the genesets of the target_knockout
        ensemble_genename_mapping = pd.read_csv(os.path.join('..','..','data','delta_selected_pathways','ensembl_genename_mapping.tsv'),sep='\t')
        ensembl_genename_dict = dict(zip(ensemble_genename_mapping.iloc[:,1], ensemble_genename_mapping.iloc[:,0]))
        knockout_ensembl = ensembl_genename_dict[target_knockout]
        belonging_genesets = [geneset for geneset in gos if geneset in gene_go_dict[knockout_ensembl]]
        affected_zs = sorted(set(sum([rel_latent_dict[np.where(np.array(gos) == belonging_go)[0][0]] for belonging_go in belonging_genesets], [])))
        affected_zs = [zs[i] for i in affected_zs]

        return affected_zs, belonging_genesets

    def modify_model(model):

        affected_zs, belonging_genesets = get_affected_zs_and_genesets()

        """
        modify the fc_mean matrix to check where the significancy is lost
        """    
        for go in tqdm(gos):
            if go not in belonging_genesets:
                go_idx = np.where(np.array(gos) == go)[0][0]
                for j in [x for x in rel_latent_dict[go_idx]]:
                    with torch.no_grad():
                        model.fc_mean.weight[j,go_idx] = 0

        return model, affected_zs, belonging_genesets

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

    def compute_activity_scores(model, layertype):

        """
        compute activity score
        """

        netactivity_scores = []
        for gene in [target_knockout]:
            
            idx = idx_dict[gene]
            mat = torch.from_numpy(adata.X[idx,:].todense()).to('cuda').double()

            if layertype == 'genesets':
                colnames = gos
                na_score = model.fc1(mat).detach().cpu().numpy()

            elif layertype == 'zs':
                colnames = zs
                na_score = model.fc_mean(model.fc1(mat)).detach().cpu().numpy()

            ##
            na_score_df = pd.DataFrame(na_score)
            na_score_df.columns = colnames
            na_score_df['type'] = gene
            netactivity_scores.append(na_score_df)

        ##
        df_netactivity_scores = pd.concat(netactivity_scores)
        return df_netactivity_scores

    def compute_pvalues_and_median_rank(ctrl_cells, knockout_cells_raw, knockout_cells_mod, affected_genesets):

        ##compute pvalue
        genesets = ctrl_cells.columns[:-1]
        _, p_value_mod = ttest_ind(ctrl_cells.loc[:,genesets].values, knockout_cells_mod.loc[:,genesets].values, equal_var=False)
    
        #build dataframe
        pvalues_df = pd.DataFrame(p_value_mod).T
        pvalues_df.columns = genesets
        pvalues_df = pvalues_df.T
        pvalues_df.columns = ['pval_mod']
        pvalues_df = pvalues_df.sort_values(by='pval_mod')

        ##
        _, p_value_raw = ttest_ind(ctrl_cells.loc[:,pvalues_df.index].values, knockout_cells_raw.loc[:,pvalues_df.index].values, equal_var=False)
        pvalues_df['pval_raw'] = p_value_raw
        pvalues_df['affected'] = [1 if x in affected_genesets else 0 for x in pvalues_df.index]

        #ranking
        ranking_df = pd.DataFrame([i+1 for i,go in enumerate(pvalues_df.sort_values(by='pval_raw').index) if go in affected_genesets])
        ranking_df.index = [go for go in pvalues_df.sort_values(by='pval_raw').index if go in affected_genesets]
        ranking_df.columns = ['raw']
        ranking_df['mod'] = [i+1 for i,go in enumerate(pvalues_df.sort_values(by='pval_mod').index) if go in affected_genesets]

        return pvalues_df.fillna(1), ranking_df

    def plot_activation_scores(ctrl_cells, knockout_cells_raw, knockout_cells_mod, pvalues_df, name = "CDKN1A"):

        #do mean
        mean_ctrl = ctrl_cells.iloc[:,:-1].mean()
        mean_knockout_raw = knockout_cells_raw.iloc[:,:-1].mean()
        mean_knockout_mod = knockout_cells_mod.iloc[:,:-1].mean()

        assert all(mean_ctrl.index == mean_knockout_raw.index)
        assert all(mean_knockout_raw.index == mean_knockout_mod.index)

        mean_scores = pd.concat([mean_ctrl, mean_knockout_raw, mean_knockout_mod], axis=1)
        mean_scores.columns = ['ctrl_activation_score', 'knockout_raw_activation_score', 'knockout_mod_activation_score']
        mean_scores_plus_pvals = pd.concat([mean_scores, pvalues_df], axis=1)
        mean_scores_plus_pvals['GO_term'] = mean_scores_plus_pvals.index

        # Apply log scale to p-values
        mean_scores_plus_pvals['log_pval_mod'] = -np.log10(mean_scores_plus_pvals['pval_mod'])
        mean_scores_plus_pvals['log_pval_raw'] = -np.log10(mean_scores_plus_pvals['pval_raw'] + 1e-100)  # Add small value to avoid log(0)

        # Generate heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(14, 8))

        # Define a custom color map for the affected column
        cmap = sns.color_palette(['blue', 'red'])

        # Left subplot: Activation scores heatmap
        sns.heatmap(mean_scores_plus_pvals.set_index('GO_term')[['ctrl_activation_score', 'knockout_raw_activation_score', 'knockout_mod_activation_score']], cmap='coolwarm', ax=axs[0], cbar_kws={'label': 'Activation Scores'}, annot=True)
        for idx, label in enumerate(axs[0].get_yticklabels()):
            label.set_color(cmap[mean_scores_plus_pvals['affected'].iloc[idx]])

        axs[0].set_title('Activation Scores Heatmap')

        # Right subplot: p-values heatmap in log scale
        sns.heatmap(mean_scores_plus_pvals.set_index('GO_term')[['log_pval_mod', 'log_pval_raw']], cmap='coolwarm', ax=axs[1], cbar_kws={'label': '-log10(p-values)'}, annot=True)
        for idx, label in enumerate(axs[1].get_yticklabels()):
            label.set_color(cmap[mean_scores_plus_pvals['affected'].iloc[idx]])

        axs[1].set_title('p-values Heatmap (log scale)')

        plt.tight_layout()
        plt.savefig(os.path.join(fpath, f'layer_{layer_name}_dynamic_vs_raw_heatmap_{name}.png'))

    def plot_activation_scores_bigger(ctrl_cells, knockout_cells, pvalues_df, name = "CDKN1A"):

        #do mean
        mean_ctrl = ctrl_cells.iloc[:,:-1].mean()
        mean_knockout = knockout_cells.iloc[:,:-1].mean()
        assert all(mean_ctrl.index == mean_knockout.index)

        mean_scores = pd.concat([mean_ctrl, mean_knockout], axis=1)
        mean_scores.columns = ['ctrl_activation_score', 'knockout_activation_score']
        mean_scores_plus_pvals = pd.concat([mean_scores, pvalues_df[['pval_mod','affected']]], axis=1)
        mean_scores_plus_pvals.columns = ['ctrl_activation_score', 'knockout_activation_score', 'pval', 'affected']
        mean_scores_plus_pvals['GO_term'] = mean_scores_plus_pvals.index

        # Generate the subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Boxplot for activation scores
        act_aff_knockout = pd.melt(knockout_cells[affected_genesets])
        act_aff_knockout['cell'] = 'knockout'
        acf_aff_ctrl = pd.melt(ctrl_cells[affected_genesets])
        acf_aff_ctrl['cell'] = 'ctrl'             
        activation_affected = pd.concat([act_aff_knockout, acf_aff_ctrl])
        sns.boxplot(data=activation_affected, x = 'variable', y = 'value', hue = 'cell', ax=axs[0])
        axs[0].set_title(f'Boxplot of Activation Scores - Knockout {name}')
        
        # Scatter plot for p-values
        sns.scatterplot(x=mean_scores_plus_pvals.index, y='pval', hue='affected', data=mean_scores_plus_pvals, ax=axs[1], palette=['blue', 'red'])
        axs[1].set_yscale('log')
        axs[1].set_title(f'Scatter Plot of p-values - Knockout {name}')
        axs[1].set_xlabel('GO Term Index')
        axs[1].set_ylabel('p-value')
        axs[1].set_xticks([])

        # Annotate red (affected) points with GO terms
        j = 0
        for i, row in mean_scores_plus_pvals.iterrows():
            if row['affected'] == 1:
                dx = -10 if j % 2 else 10
                #dx = 10
                plt.annotate(row.name, (i, row['pval']), textcoords="offset points", xytext=(0,dx), ha='center', fontsize=9, color='red')
                j+=1
        plt.tight_layout()
        plt.savefig(os.path.join(fpath, f'layer_{layer_name}_boxplot_plus_scatterplot_{name}.png'))

    ## load info
    mode_type = 'full_go'
    trainmode = 'NA+deltas'
    
    fpath = os.path.join('./../../../','figures','uhler_paper',f'{mode_type}_{trainmode}','activation_scores', target_knockout)
    if not os.path.isdir(fpath):
        os.mkdir(fpath)

    ##get the output of NetActivity Layer
    batch_size, mode = 128, 'train'
    dataloader, _, _, _, ptb_targets = get_data(batch_size=batch_size, mode=mode)
    adata, idx_dict, gos, zs = load_data_raw_go(ptb_targets)

    if trainmode == 'NA_NA':

        ## fc_mean / zs
        layer_name = 'fc_mean'
        model, affected_zs, affected_genesets = modify_model(load_model(layertype = 'zs', layer_name = 'fc_mean'))
        na_activity_score_fc_mean = pd.read_csv(os.path.join('./../../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv'),sep='\t',index_col=0)
        ctrl_cells_fc_mean_raw = na_activity_score_fc_mean[na_activity_score_fc_mean['type'] == 'ctrl']
        knockout_cells_fc_mean_raw = na_activity_score_fc_mean[na_activity_score_fc_mean['type'] == target_knockout]
        knockout_cells_fc_mean_mod = compute_activity_scores(model, layertype = 'zs')
        pvalues_df_fc_mean, ranking_df_fc_mean = compute_pvalues_and_median_rank(ctrl_cells_fc_mean_raw, knockout_cells_fc_mean_raw, knockout_cells_fc_mean_mod, affected_zs)
        plot_activation_scores(ctrl_cells_fc_mean_raw, knockout_cells_fc_mean_raw, knockout_cells_fc_mean_mod, pvalues_df_fc_mean, name = target_knockout)

    ## fc1 / genesets
    layer_name = 'fc1'
    affected_zs, affected_genesets = get_affected_zs_and_genesets()
    na_activity_score_fc1 = pd.read_csv(os.path.join('./../../../result',f'{mode_type}_{trainmode}',f'na_activity_scores_layer_{layer_name}.tsv'),sep='\t',index_col=0)
    ctrl_cells_fc1 = na_activity_score_fc1[na_activity_score_fc1['type'] == 'ctrl']
    knockout_cells_fc1 = na_activity_score_fc1[na_activity_score_fc1['type'] == target_knockout]
    pvalues_df_fc1, ranking_df_fc1 = compute_pvalues_and_median_rank(ctrl_cells_fc1, knockout_cells_fc1, knockout_cells_fc1, affected_genesets)
    plot_activation_scores_bigger(ctrl_cells_fc1, knockout_cells_fc1, pvalues_df_fc1, name = target_knockout)

## analysis on target knockout "CDKN1A"
compute_scores_knockout_analysis(target_knockout = "CDKN1A")