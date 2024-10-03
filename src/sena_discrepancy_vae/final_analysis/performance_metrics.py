import pandas as pd
import os

df = pd.read_csv(os.path.join('./../../','result','uhler','post_analysis','performance_metrics','performance_table.tsv'),sep='\t',index_col=0)

#build double
for mode in ['double']:
    df_mode = df[df['mode'] == mode]
    results_df = df_mode.groupby(['Metric','latdim','model_name'])['Values'].mean().reset_index()
    results_df.columns = ['Metric','latdim','model_name','mean']
    results_df['std'] = df_mode.groupby(['Metric','latdim','model_name'])['Values'].std().reset_index()['Values']

    #round
    results_df['mean'] = results_df['mean'].round(5)
    results_df['std'] = results_df['std'].round(6)

    for model in ['full_go_regular','full_go_sena_delta_0','full_go_sena_delta_1','full_go_sena_delta_2','full_go_sena_delta_3']:
        if any(results_df['model_name'].isin([model])):
            model_df = results_df[results_df['model_name'] == model]
            print(model_df)

    #results_df.to_csv(os.path.join('./../../','result','uhler','post_analysis','performance_metrics',f'performance_table_{mode}.tsv'))
