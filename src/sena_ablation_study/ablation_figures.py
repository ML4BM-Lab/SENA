import numpy as np
import sena_tools as st
import importlib
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
importlib.reload(st)
import os
import pickle
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42

"""efficient"""
def plot_mse_analysis(mode = '1layer', methods = [], dataset = 'norman', structure='ae', metric='test_mse', plot_type = 'std'):

    def build_dataset(structure, beta):

        #mode
        variables = ['mode', 'epoch', f'{metric}', 'seed']

        ##
        arch_l = []
        for arch in methods:
            if structure == 'vae':
                arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'vae_{arch}_ablation_efficiency_{mode}_{dataset}_beta_{beta}.tsv'), sep='\t', index_col=0)
            else:
                arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'autoencoder_{arch}_ablation_efficiency_{mode}_{dataset}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_mse[variables])
        
        df = pd.concat(arch_l)
        df['beta'] = beta
        return df

    #retrieve dataset
    colors = sorted(sns.color_palette("Set2", len(methods)))
    #color_mapping = dict(zip(methods, colors))

    #
    beta = float(structure.split('_')[-1]) if 'vae' in structure else 0
    structure = structure.split('_')[0]
    df = build_dataset(structure, beta)

    # Group by epoch and mode, then calculate the median and IQR for MSE
    if plot_type == 'quantile':
        grouped = df.groupby(['epoch', 'mode']).agg(
            metric_median=(f'{metric}', 'median'),
            Q1=(f'{metric}', lambda x: np.percentile(x, 25)),
            Q3=(f'{metric}', lambda x: np.percentile(x, 75))
        ).reset_index()

        #Calculate the IQR
        grouped['IQR'] = grouped['Q3'] - grouped['Q1']

        # Calculate the lower and upper bounds for the shadow
        grouped['lower_bound'] = grouped['Q1'] - 1.5 * grouped['IQR']
        grouped['upper_bound'] = grouped['Q3'] + 1.5 * grouped['IQR']

    else:
        grouped = df.groupby(['epoch', 'mode']).agg(
                metric_mean=(metric, 'mean'),
                metric_std=(metric, 'std')
        ).reset_index()

    # Set up the figure and axes
    plt.figure(figsize=(12, 8))

    # Loop through each mode to plot
    for method, color in zip(methods, colors):

        method_group = grouped[grouped['mode'] == method]
        
        if plot_type == 'std':
            plt.plot(method_group['epoch'], method_group['metric_mean'], '-o', label=method.capitalize(), color=color, markersize=4)
            plt.fill_between(
                method_group['epoch'], 
                method_group['metric_mean'] - method_group['metric_std']/2, 
                method_group['metric_mean'] + method_group['metric_std']/2, 
                color=color, 
                alpha=0.2
            )
        else:
            plt.plot(method_group['epoch'], method_group['metric_median'], '-o', label=method.capitalize(), color=color, markersize=4)
            plt.fill_between(
                method_group['epoch'], 
                method_group['lower_bound'], 
                method_group['upper_bound'],  
                color=color, 
                alpha=0.2
            )


    # Set y-axis to log scale
    plt.yscale('log')


    # Add a title and labels
    plt.title(f'Comparison of {metric} Across Epochs: SENA vs Regular (Median & IQR)', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'Median {metric} (log scale)', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()

    # Save the plot
    if 'vae' in structure:
        plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_{mode}_{metric}_{dataset}_beta_{beta}.pdf'))
    else:
        plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_{mode}_{metric}_{dataset}.pdf'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_sparsity_analysis(mode = '1layer', methods = [], dataset = 'norman', structure='ae'):

    def build_dataset():

        #mode
        variables = ['mode', 'epoch', 'seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_efficiency_{mode}_{dataset}.tsv'), sep='\t', index_col=0)
            sparse_vars = [x for x in arch_test_mse.columns if 'sparsity' in x]
            arch_l.append(arch_test_mse[variables+sparse_vars])
        
        df = pd.concat(arch_l)
        return df

    #retrieve dataset
    colors = sorted(sns.color_palette("Set2", len(methods)))
    color_mapping = dict(zip(methods, colors))
    df = build_dataset()
    

    #keep last threshold
    sparsity_thresholds = [x for x in df.columns if 'sparsity' in x][0]


    # Convert the data from wide to long format for better handling by seaborn
    last_epoch = df[df['epoch'] == df['epoch'].max()].reset_index(drop=True)

    # Melt the DataFrame to long format for easier plotting with seaborn
    long_df = last_epoch.melt(id_vars=['mode', 'epoch'], value_vars=sparsity_thresholds, 
                            var_name='Sparsity Threshold', value_name='Sparsity')

    # Sort by 'sparsity_1e-08_mean'
    grouped_sorted = long_df.sort_values(by='Sparsity', ascending=False).reset_index(drop=True)

    # Set up the figure
    plt.figure(figsize=(5, 8))

    # Create the barplot (Seaborn will automatically calculate the mean and std for error bars)
    sns.barplot(x='mode', y='Sparsity', hue='mode', data=grouped_sorted, errorbar='sd', palette=color_mapping, capsize=0.1)

    # Set up the labels and title
    plt.title('Boxplot of Sparsity Thresholds Across Modes', fontsize=16)
    plt.xlabel('Sparsity Threshold', fontsize=14)
    plt.ylabel('Sparsity', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    #save figure
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_all_ablation_1layer_sparsity_{dataset}.pdf'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def compute_time_consumption(mode = '1layer', methods = [], dataset = 'norman', structure='ae'):

    def build_dataset():

        #mode
        variables = ['mode', 'epoch', 'time', 'seed']

        ##
        arch_l = []
        for arch in methods:
            
            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'{"autoencoder" if structure=="ae" else "vae"}_{arch}_ablation_efficiency_{mode}_{dataset}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse[variables])
        
        df = pd.concat(arch_l)
        return df

    #load df
    df = build_dataset()

    #build t
    time_stats = df.groupby('mode').agg(
                mean_time=('time', 'mean'),
                std_time=('time', 'std')
                ).reset_index()
    
    print(time_stats)

"""interpretable"""
def plot_outlier_analysis_combined(mode = '1layer', dataset = ['norman'], metric = 'z_diff', methods = [], structure=['ae']):

    def build_dataset():
        ##
        arch_l = []
        for mod in mode:
            for arch in methods:
                for struct in structure:
                    beta = float(struct.split('_')[-1]) if 'vae' in struct else 0
                    struct = struct.split('_')[0]
                    if mod == '1layer':
                        arch = arch.replace('_delta','')
                    for dt in dataset:

                        if struct == 'ae':
                            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{struct}_{arch}',f'{"autoencoder" if struct=="ae" else "vae"}_{arch}_ablation_interpretability_{mod}_{dt}.tsv'), sep='\t', index_col=0)
                        elif struct == 'vae':
                            arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{struct}_{arch}',f'{"autoencoder" if struct=="ae" else "vae"}_{arch}_ablation_interpretability_{mod}_{dt}_beta_{beta}.tsv'), sep='\t', index_col=0)
                        
                            
                        arch_test_mse['mode'] = arch_test_mse['mode']+'_'+struct
                        arch_test_mse['dataset'] = dt
                        arch_test_mse['th'] = arch.split('_')[-1]
                        arch_test_mse['layer'] = mod
                        arch_l.append(arch_test_mse)
        
        df = pd.concat(arch_l)
        return df

    #get data
    df = build_dataset()

    # Filter the DataFrame for the specified epochs
    epochs_to_plot = [200]
    th_order = ['0', '3', '2', '1', '0.3'][::-1]
    colors = sns.color_palette("Set2", 2*2*1)
    #colors = ["#c57c3c", "#b3669e"]
    
    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Define markers for 1layer and 2layer
    markers = {'1layer': '^', '2layer': 's'}
    linestyles = {'1layer': '--', '2layer': '-'}

    # Loop through the epochs and colors
    i = 0
    for struct in structure:
        struct = struct.split('_')[0]
        for layer in ['1layer', '2layer']:
            color = colors[i]
            i+=1
            for epoch in epochs_to_plot:

                # Filter the DataFrame for the current epoch
                sub_df = df[(df['epoch'] == epoch) & (df['mode'].apply(lambda x: x.split('_')[-1]==struct in x)) & (df['layer'] == layer)]
                
                # Group by 'th' to calculate the mean and std across seeds
                aggregated_df = sub_df.groupby('th').agg(
                    recall_mean=(metric, 'mean'),
                    recall_std=(metric, 'std')
                ).reindex(th_order).reset_index()

                # Plot the mean recall_at_100 as a function of th
                plt.plot(
                    aggregated_df['th'], aggregated_df['recall_mean'],
                    marker=markers[layer], linestyle=linestyles[layer], color=color,
                    label=f'{struct} - {layer} - epoch {epoch}', markersize=8
                )
                
                # Fill the area between (mean - std) and (mean + std)
                plt.fill_between(
                    aggregated_df['th'], 
                    aggregated_df['recall_mean'] - aggregated_df['recall_std'], 
                    aggregated_df['recall_mean'] + aggregated_df['recall_std'], 
                    color=color, alpha=0.2
                )

    # Add labels and title
    plt.title(f'{metric} vs Threshold (th) for 1 Layer and 2 Layer Models', fontsize=16)
    plt.xlabel('Threshold (th)', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.xticks(th_order)

    # Add gridlines for readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Model Layer')
    plt.show()

    # Save the plot (if needed)
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{"_".join(structure)}_groupal_ablation_{mode}_{metric}_{dataset[-1]}.pdf'))

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()

def plot_outlier_analysis(mode = '1layer', dataset = ['norman'], methods = [], name = 'all', metric = 'z_diff', structure='ae'):

    def build_dataset(structure, beta):

        #mode
        variables = ['mode', 'epoch', f'{metric}', 'seed']

        ##
        arch_l = []
        for arch in methods:
            for dt in dataset:
                if structure == 'vae':
                    arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'vae_{arch}_ablation_interpretability_{mode}_{dt}_beta_{beta}.tsv'), sep='\t', index_col=0)
                else:
                    arch_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'"autoencoder"_{arch}_ablation_interpretability_{mode}_{dt}.tsv'), sep='\t', index_col=0)
                arch_l.append(arch_mse[variables])
        
        df = pd.concat(arch_l)
        df['beta'] = beta
        return df

    colors = sns.color_palette("Set2", len(methods))
    beta = float(structure.split('_')[-1]) if 'vae' in structure else 0
    structure = structure.split('_')[0]
    df = build_dataset(structure, beta)

    # Create a figure
    plt.figure(figsize=(12, 8))

    if len(dataset) == 1:

        grouped = df.groupby(['epoch', 'mode']).agg(
        metric_mean=(metric, 'mean'),
        metric_std=(metric, 'std')
        ).reset_index()

        # Loop over each method and plot the corresponding data
        for method, color in zip(methods, colors):
            # Filter data for the current method
            grouped_method = grouped[grouped['mode'] == method]
            
            # Plot for the current method
            plt.plot(grouped_method['epoch'], grouped_method['metric_mean'], '-o', label=method, color=color)
            plt.fill_between(
                grouped_method['epoch'], 
                grouped_method['metric_mean'] - grouped_method['metric_std'], 
                grouped_method['metric_mean'] + grouped_method['metric_std'], 
                color=color, 
                alpha=0.2
            )

            plt.xlabel('Epoch', fontsize=14)
            plt.title('Metric vs. Epoch for Different Methods', fontsize=16)

    else:

        df = df[df['epoch'] == df['epoch'].max()].reset_index(drop=True)
        df['dataset'] = df['dataset'].replace({'norman':'norman_5'})
       
        grouped = df.groupby(['epoch', 'mode','dataset']).agg(
        metric_mean=(metric, 'mean'),
        metric_std=(metric, 'std')
        ).reset_index()

        # Loop over each method and plot the corresponding data
        for method, color in zip(methods, colors):
            # Filter data for the current method
            grouped_method = grouped[grouped['mode'] == method]
            
            # Plot the mean metric as a function of the dataset (x-axis)
            plt.plot(grouped_method['dataset'], grouped_method['metric_mean'], '--^', label=method, color=color, markersize=10)
            
            # Fill the area between (mean - std) and (mean + std) for error bars
            plt.fill_between(
                grouped_method['dataset'], 
                grouped_method['metric_mean'] - grouped_method['metric_std'], 
                grouped_method['metric_mean'] + grouped_method['metric_std'], 
                color=color, 
                alpha=0.2
            )

        # Add labels and title
        plt.xlabel('Dataset', fontsize=14)
        plt.title('Metric vs. Dataset for Different Methods', fontsize=16)

    plt.ylabel('Metric Mean', fontsize=14)
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the legend
    plt.legend()
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_{name}_ablation_{mode}_{metric}_{dataset[-1]}.pdf'))
    plt.cla()
    plt.clf()
    plt.close()

"""latent correlation"""
def plot_latent_correlation(mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', dataset = 'norman', epoch = 5, structure='ae'):

    ## load data
    with open(os.path.join('./../../result/ablation_study',f'{structure}_{modeltype}',f'{"autoencoder" if structure=="ae" else "vae"}_{modeltype}_ablation_{analysis}_{mode}_{dataset}.pickle'), 'rb') as handle:
        results = pickle.load(handle)

    #subset
    seed_0_df = results[0]
    subset_epoch = seed_0_df[seed_0_df['epoch']==epoch]
    
    # Melt the DataFrame to plot both input_zdiff and latent_zdiff in a single plot
    df_melted = subset_epoch.melt(value_vars=['input_zdiff', 'latent_zdiff'], var_name='Type', value_name='z_diff')

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(data=df_melted, x='Type', y='z_diff', palette='Set3', width=0.5)

    # Set titles and labels
    plt.title(f'Comparison of Input and Latent z_diff - Epoch {epoch}', fontsize=16)
    plt.xlabel('Type', fontsize=14)
    plt.ylabel('z_diff', fontsize=14)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.savefig(os.path.join('./../../figures','ablation_study',f'{structure}_{modeltype}_ablation_{mode}_{analysis}_{epoch}_{dataset}.pdf'))
    plt.cla()
    plt.clf()
    plt.close()

def compute_metrics(mode = '1layer', methods = [], dataset = 'norman', metric = 'recall_at_25', structure='ae', analysis='interpretability'):

    def build_dataset(structure, beta):

        ##
        arch_l = []
        for arch in methods:
            
            if structure == 'vae':
                arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'vae_{arch}_ablation_{analysis}_{mode}_{dataset}_beta_{beta}.tsv'), sep='\t', index_col=0)
            else:
                arch_test_mse = pd.read_csv(os.path.join('./../../result','ablation_study',f'{structure}_{arch}',f'autoencoder_{arch}_ablation_{analysis}_{mode}_{dataset}.tsv'), sep='\t', index_col=0)
            arch_l.append(arch_test_mse)
        
        df = pd.concat(arch_l)
        return df

    beta = float(structure.split('_')[-1]) if 'vae' in structure else 0
    structure = structure.split('_')[0]
    df = build_dataset(structure, beta)

    grouped = df.groupby(['epoch', 'mode']).agg(
            metric_mean=(metric, 'mean'),
            metric_std=(metric, 'std')
            ).reset_index()
    
    subset_lepoch = grouped[grouped['epoch'] == grouped['epoch'].max()].reset_index(drop=True)

    print(f"Number of layers: {mode}")
    for i in range(subset_lepoch.shape[0]):
        print(f'{subset_lepoch["mode"].iloc[i]}, {metric}: {subset_lepoch["metric_mean"].iloc[i]} +- {subset_lepoch["metric_std"].iloc[i]}')


"""AE"""
def _call_ae(layers='1layer'):

    if layers == '1layer': #"""1layer""" #compare sena vs regular

        methods = ['sena_0', 'sena_1', 'sena_2', 'sena_3', 'regular', 'l1_3', 'l1_5']
        #plot_mse_analysis(mode = '1layer', methods = methods, dataset = 'norman')
        compute_metrics(mode='1layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency')

        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2', 'sena_delta_3', 'regular', 'l1_3', 'l1_5']
        compute_metrics(mode='2layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency')
        #plot_sparsity_analysis(mode = '1layer', methods=methods, dataset = 'norman')
        #compute_time_consumption(mode = '1layer', methods=methods, dataset = 'norman')

        # methods = ['sena_0', 'sena_1', 'sena_2', 'sena_3', 'sena_0.3']
        # plot_outlier_analysis(mode='1layer', metric = 'recall_at_25', methods=methods, dataset = ['norman_1','norman_2','norman_3','norman_4','norman'])
        # plot_outlier_analysis(mode='1layer', metric = 'recall_at_100', methods=methods, dataset = ['norman_1','norman_2','norman_3','norman_4','norman'])
        # #compute_metrics(mode='2layer', metric = 'recall_at_100', methods = methods, dataset = 'norman')
        #compute_metrics(mode='2layer', metric = 'recall_at_25', methods = methods, dataset = 'norman')

    elif layers == '2layer': #"""2layer""" #sena-delta
    
        methods = ['sena_delta_0', 'sena_delta_1','sena_delta_2','sena_delta_3', 'sena_delta_5', 'regular', 'l1_3','l1_5']
        plot_mse_analysis(mode = '2layer', methods = methods, dataset = 'norman')
        #plot_sparsity_analysis(mode = '2layer', methods=methods, dataset = 'norman')

        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2','sena_delta_3', 'sena_delta_5', 'sena_delta_0.3']
        plot_outlier_analysis(mode='2layer', metric = 'recall_at_25', methods = methods, dataset = ['norman'])
        plot_outlier_analysis(mode='2layer', metric = 'recall_at_100', methods = methods, dataset = ['norman'])
        #compute_metrics(mode='2layer', metric = 'recall_at_100', methods = methods, dataset = 'norman')
        #compute_metrics(mode='2layer', metric = 'recall_at_25', methods = methods, dataset = 'norman')

    elif layers == 'combined':

        """combined plots"""
        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2', 'sena_delta_3', 'sena_delta_0.3']
        plot_outlier_analysis_combined(mode=['1layer','2layer'], metric = 'recall_at_100', methods=methods, dataset = ['norman'], structure=['ae','vae_1.0'])

        #analyze single architecture (e.g. sena) between "mean of affected expression DE" and "latent space DE" at a specific epochs
        #plot_latent_correlation(epoch=45, mode = '1layer', analysis = 'lcorr', modeltype = 'sena_0', dataset = 'norman')

    elif layers == 'table':

        methods = ['sena_0', 'sena_1', 'sena_2', 'sena_3', 'regular', 'l1_3', 'l1_5']
        compute_metrics(mode='1layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency')
        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2', 'sena_delta_3', 'regular', 'l1_3', 'l1_5']
        compute_metrics(mode='2layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency')
        
def _call_vae(layers='1layer', beta=1.0):

    if layers == '1layer':

        """single layer"""
        #compare sena vs regular
        methods = ['regular', 'sena_0', 'sena_1', 'sena_2', 'sena_3', 'l1_3']
        #plot_mse_analysis(mode = '1layer', methods = methods, dataset = 'norman', structure = f'vae_{beta}', metric = 'test_mse')
        #plot_mse_analysis(mode = '1layer', methods = methods, dataset = 'norman', structure = f'vae_{beta}', metric = 'test_KL')

        methods = ['sena_0','sena_1','sena_2','sena_3', 'senadec_0','sena_dec1','senadec_2','sena_dec3']
        plot_outlier_analysis(mode='1layer', metric = 'recall_at_25', methods = methods, dataset = ['norman'], structure=f'vae_{beta}')
        plot_outlier_analysis(mode='1layer', metric = 'recall_at_100', methods = methods, dataset = ['norman'], structure=f'vae_{beta}')

    elif layers == '2layer':

        """two layers layer"""
        methods = ['regular', 'sena_delta_0','sena_delta_1','sena_delta_3', 'l1_3']
        #plot_mse_analysis(mode = '2layer', methods = methods, dataset = 'norman', structure=f'vae_{beta}', metric='test_mse')
        #plot_mse_analysis(mode = '2layer', methods = methods, dataset = 'norman', structure=f'vae_{beta}', metric='test_KL')
        #compute_metrics(mode='2layer', metric = 'recall_at_100', methods=methods, dataset = 'norman', structure=f'vae_{beta}')
        #compute_metrics(mode='2layer', metric = 'recall_at_25', methods=methods, dataset = 'norman', , structure=f'vae_{beta}')

        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2', 'sena_delta_3']
        plot_outlier_analysis(mode='2layer', metric = 'recall_at_25', methods=methods, dataset = ['norman'], structure=f'vae_{beta}')
        plot_outlier_analysis(mode='2layer', metric = 'recall_at_100', methods=methods, dataset = ['norman'], structure=f'vae_{beta}')

    elif layers == 'table':

        methods = ['sena_0', 'sena_1', 'sena_2', 'sena_3', 'regular', 'l1_3', 'l1_5']
        #compute_metrics(mode='1layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency', structure=f'vae_{beta}')
        #compute_metrics(mode='1layer', metric = 'test_KL', methods = methods, dataset = 'norman', analysis='efficiency', structure=f'vae_{beta}')

        methods = ['sena_delta_0', 'sena_delta_1', 'sena_delta_2', 'sena_delta_3', 'regular', 'l1_3', 'l1_5']
        compute_metrics(mode='2layer', metric = 'test_mse', methods = methods, dataset = 'norman', analysis='efficiency', structure=f'vae_{beta}')
        compute_metrics(mode='2layer', metric = 'test_KL', methods = methods, dataset = 'norman', analysis='efficiency', structure=f'vae_{beta}')


    
if __name__ == '__main__':

    """ae"""

    #recall@100 
    #_call_ae(layers='combined')

    #mse
    _call_ae(layers='1layer')
    
    #_call_ae(layers='table')
    #_call_vae(layers='table')

    #_call_vae(layers='1layer', beta=1.0)
    #_call_vae(layers='1layer', beta=0.1)
    #_call_vae(layers='1layer', beta=0.01)

    #_call_vae(layers='2layer', beta=10)
    #_call_vae(layers='2layer', beta=0.01)
    #_call_vae(layers='2layer', beta=1)