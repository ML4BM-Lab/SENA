import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def compute_activation_df(na_activity_score, scoretype, gos, mode, gene_go_dict, genename_ensemble_dict, ptb_targets):

    ## define control cells
    ctrl_cells = na_activity_score[na_activity_score.index == 'ctrl'].to_numpy()

    ## init df
    ttest_df = []

    for knockout in ptb_targets:
        
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

def get_DA_df_by_geneset(mode, latdim, seed=42, dataset='full_go'):

    model_name = f'{dataset}_{mode}'

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

    return DA_df_by_geneset

##
modes = ['sena_delta_0']
latdims = [105, 70, 35, 10, 5]
th_genes = [1,3,5,7]
th_gene = 1
results = []
seeds = [42, 13]

for seed in seeds:
    for mode in tqdm(modes):
        for latdim in latdims:
            DA_df_by_geneset = get_DA_df_by_geneset(mode=mode, latdim=latdim, seed=seed)
            affected_counter = DA_df_by_geneset.groupby(['knockout'])['affected'].sum()
            relevant_knockouts = sorted(affected_counter[affected_counter.values>=th_gene].index)
            DA_df_by_geneset_filtered = DA_df_by_geneset[DA_df_by_geneset['knockout'].isin(relevant_knockouts)]

            ##
            affected_score = DA_df_by_geneset_filtered.loc[DA_df_by_geneset_filtered['affected'] == True,'score']
            non_affected_score = DA_df_by_geneset_filtered.loc[DA_df_by_geneset_filtered['affected'] == False,'score']
            _, p_value = ttest_ind(non_affected_score, affected_score, equal_var=False)

            results.append([mode, latdim, th_gene, p_value, seed])

results_df = pd.DataFrame(results)
results_df.columns = ['mode','latdim','th_gene','pvalue', 'seed']
results_df['latdim'] = results_df['latdim'].apply(lambda x: str(x))

# First, compute the mean, min, and max p-values for each group
grouped_df = results_df.groupby(['latdim', 'mode', 'th_gene']).agg(
    mean_pvalue=('pvalue', 'mean'),
    min_pvalue=('pvalue', 'min'),
    max_pvalue=('pvalue', 'max')
).reset_index()


"""p-value analysis"""

# Adjusting the data to plot with seaborn
plt.figure(figsize=(10, 6))

# Create the lineplot with error bands representing the variance over seeds
sns.lineplot(
    x='latdim',
    y='pvalue',
    hue='mode',
    style='th_gene',
    data=results_df,
    markers=True,
    dashes=False,
    ci=None  # Use standard deviation to represent variance as a shadow
)

# Get the current axes
ax = plt.gca()

# Loop over each combination of 'mode' and 'th_gene' to plot the shaded area
for (mode, th_gene), group in grouped_df.groupby(['mode', 'th_gene']):
    # Sort the group by 'latdim' to ensure proper plotting
    group = group.sort_values('latdim')
    
    # Extract the color for the current hue and style combination
    line = ax.lines[-1]  # Get the last line plotted
    color = line.get_color()
    
    # Plot the fill_between for the min and max p-values
    ax.fill_between(
        group['latdim'],
        group['min_pvalue'],
        group['max_pvalue'],
        alpha=0.2,
        color=color
    )

plt.yscale('log')  # Set y-axis to log scale for better visualization of p-values
plt.title('P-value vs Latent Dimension (Discrete X-axis)')
plt.xlabel('Latent Dimension (Discrete)')
plt.ylim(1e-15, 1)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join('./../../', 'figures', 'uhler', 'final_figures', f'DA_analysis_pvalues_thgene_{th_gene}.pdf'))

# Show the plot
plt.show()
