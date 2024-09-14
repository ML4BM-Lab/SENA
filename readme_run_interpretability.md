
## ---------------------------------------------- interpretability ---------------------------------------------

## ------------------------------------------------ AE --------------------------------------------------------

# sena
nohup python3 -u regular_ae.py sena_0 interpretability norman > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_3.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_2 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_2.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_4 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_4.log &

nohup python3 -u regular_ae.py sena_1 interpretability norman > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_3.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_2 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_2.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_4 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_4.log &

nohup python3 -u regular_ae.py sena_2 interpretability norman > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_3.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_2 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_2.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_4 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_4.log &

nohup python3 -u regular_ae.py sena_3 interpretability norman > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_3.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_2 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_2.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_4 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_4.log &

nohup python3 -u regular_ae.py sena_0.3 interpretability norman > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_3.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_2 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_2.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_4 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_4.log &

nohup python3 -u regular_ae.py sena_delta_0 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_1 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_1_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_2 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_2_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_3 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_5 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_5_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_0.3 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_03_2layer_interpretability_norman.log &


## ## ------------------------------------------------ VAE --------------------------------------------------------


# sena
nohup python3 -u variational_ae.py sena_0 interpretability norman 1 > ./../../logs/ablation_study/vae_sena_0_1layer_interpretability_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_0 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_interpretability_norman_beta1.log &

nohup python3 -u variational_ae.py sena_1 interpretability norman 1 > ./../../logs/ablation_study/vae_sena_1_1layer_interpretability_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_1 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_interpretability_norman_beta1.log &

nohup python3 -u variational_ae.py sena_2 interpretability norman 1 > ./../../logs/ablation_study/vae_sena_2_1layer_interpretability_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_2 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_2_2layer_interpretability_norman_beta1.log &

nohup python3 -u variational_ae.py sena_3 interpretability norman 1 > ./../../logs/ablation_study/vae_sena_3_1layer_interpretability_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_3 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_interpretability_norman_beta1.log &

nohup python3 -u variational_ae.py sena_0.3 interpretability norman 1 > ./../../logs/ablation_study/vae_sena_03_1layer_interpretability_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_0.3 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_03_2layer_interpretability_norman_beta1.log &