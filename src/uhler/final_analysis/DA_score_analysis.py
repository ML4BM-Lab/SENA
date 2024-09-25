import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns

def compute_activation_df(na_activity_score, scoretype, gos, mode, gene_go_dict, genename_ensemble_dict, ptb_targets):

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score.index == 'ctrl'].to_numpy()

    ## init df
    ttest_df = []

    for knockout in tqdm(ptb_targets):
        
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

    # #apply min-max norm
    # if scoretype == 'mu_diff':
    #     for knockout in ttest_df['knockout'].unique():
    #         ttest_df.loc[ttest_df['knockout'] == knockout, 'score'] = MinMaxScaler().fit_transform(ttest_df.loc[ttest_df['knockout'] == knockout, 'score'].values.reshape(-1,1))

    return ttest_df

dataset = 'full_go'
mode = 'sena_delta_2'
model_name = f'{dataset}_{mode}'
seed = 42
latdim = 105

#load summary file
with open(os.path.join(f'./../../result/uhler/{model_name}/seed_{seed}_latdim_{latdim}/post_analysis_{model_name}_seed_{seed}_latdim_{latdim}.pickle'), 'rb') as handle:
    model_summary = pickle.load(handle)

#activation layer
_, _, _, ptb_targets_affected, gos, rel_dict, gene_go_dict, genename_ensemble_dict = utils.load_norman_2019_dataset()
fc1 = model_summary['fc1']

#compute DA by geneset at the output of the SENA layer
DA_df_by_geneset = compute_activation_df(fc1, scoretype = 'mu_diff', gos=gos, mode=mode, 
                                         gene_go_dict=gene_go_dict, genename_ensemble_dict=genename_ensemble_dict,
                                         ptb_targets=ptb_targets_affected)


"""p-value analysis"""

#only 
affected_counter = DA_df_by_geneset.groupby(['knockout'])['affected'].sum()
relevant_knockouts = sorted(affected_counter[affected_counter.values>=7].index)

#filter
DA_df_by_geneset_filtered = DA_df_by_geneset[DA_df_by_geneset['knockout'].isin(relevant_knockouts)]

# Adjusting the data to plot with seaborn
plt.figure(figsize=(10, 6))

# 

# Boxplot with knockout on x-axis, hue by affected, and score on y-axis
sns.boxplot(x='knockout', y='score', hue='affected', data=DA_df_by_geneset_filtered, fliersize=3)

plt.title('Score by Knockout with Affected=True/False')
plt.xlabel('Knockout')
plt.ylabel('Score')
plt.yscale('log')
plt.show()
plt.savefig(os.path.join('./../../','figures','uhler','final_figures','DA_analysis.png'))