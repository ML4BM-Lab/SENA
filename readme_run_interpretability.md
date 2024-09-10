
## ---------------------------------------------- interpretability ---------------------------------------------

## ------------------------------------------------ AE --------------------------------------------------------
# regular
nohup python3 -u regular_ae.py regular interpretability norman > ./../../logs/ablation_study/ae_regular_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py regular interpretability norman 2 > ./../../logs/ablation_study/ae_regular_2layer_interpretability_norman.log &

# sena
nohup python3 -u regular_ae.py sena_0 interpretability norman > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_0 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman_3.log &

nohup python3 -u regular_ae.py sena_1 interpretability norman > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman_3.log &

nohup python3 -u regular_ae.py sena_2 interpretability norman > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_2 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_2_1layer_interpretability_norman_3.log &


nohup python3 -u regular_ae.py sena_3 interpretability norman > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman_3.log &


nohup python3 -u regular_ae.py sena_0.3 interpretability norman > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_1.log &
nohup python3 -u regular_ae.py sena_0.3 interpretability norman_3 > ./../../logs/ablation_study/ae_sena_03_1layer_interpretability_norman_3.log &


nohup python3 -u regular_ae.py sena_0.01 interpretability norman > ./../../logs/ablation_study/ae_sena_001_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_0.01 interpretability norman_1 > ./../../logs/ablation_study/ae_sena_001_1layer_interpretability_norman_1.log &


nohup python3 -u regular_ae.py sena_delta_0 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_1 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_1_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_3 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_interpretability_norman.log &


## l1
nohup python3 -u regular_ae.py l1_3 interpretability norman > ./../../logs/ablation_study/ae_l1_3_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py l1_5 interpretability norman > ./../../logs/ablation_study/ae_l1_5_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py l1_7 interpretability norman > ./../../logs/ablation_study/ae_l1_7_1layer_interpretability_norman.log &

nohup python3 -u regular_ae.py l1_3 interpretability norman 2 > ./../../logs/ablation_study/ae_l1_3_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py l1_5 interpretability norman 2 > ./../../logs/ablation_study/ae_l1_5_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py l1_7 interpretability norman 2 > ./../../logs/ablation_study/ae_l1_7_2layer_interpretability_norman.log &

## ## ------------------------------------------------ VAE --------------------------------------------------------

# regular
nohup python3 -u variational_ae.py regular interpretability norman > ./../../logs/ablation_study/vae_regular_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py regular interpretability norman 2 > ./../../logs/ablation_study/vae_regular_2layer_interpretability_norman.log &

# sena
nohup python3 -u variational_ae.py sena_0 interpretability norman > ./../../logs/ablation_study/vae_sena_0_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py sena_delta_0 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_interpretability_norman.log &

nohup python3 -u variational_ae.py sena_1 interpretability norman > ./../../logs/ablation_study/vae_sena_1_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py sena_delta_1 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_interpretability_norman.log &

nohup python3 -u variational_ae.py sena_3 interpretability norman > ./../../logs/ablation_study/vae_sena_3_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py sena_delta_3 interpretability norman 2 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_interpretability_norman.log &

# l1
nohup python3 -u variational_ae.py l1_3 interpretability norman > ./../../logs/ablation_study/vae_l1_3_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py l1_5 interpretability norman > ./../../logs/ablation_study/vae_l1_5_1layer_interpretability_norman.log &
nohup python3 -u variational_ae.py l1_7 interpretability norman > ./../../logs/ablation_study/vae_l1_7_1layer_interpretability_norman.log &

nohup python3 -u variational_ae.py l1_3 interpretability norman 2 > ./../../logs/ablation_study/vae_l1_3_2layer_interpretability_norman.log &
nohup python3 -u variational_ae.py l1_5 interpretability norman 2 > ./../../logs/ablation_study/vae_l1_5_2layer_interpretability_norman.log &
nohup python3 -u variational_ae.py l1_7 interpretability norman 2 > ./../../logs/ablation_study/vae_l1_7_2layer_interpretability_norman.log &
