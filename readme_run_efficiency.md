

## ------------------------------------------ ablation study (norman) ------------------------------------------------
## ablation study
docker exec -it causal bash
cd /wdir/src/sena_ablation_study


## ---------------------------------------------------- efficiency --------------------------------------------------------

## ------------------------------------------------------- AE --------------------------------------------------------------

# regular
nohup python3 -u regular_ae.py regular efficiency norman > ./../../logs/ablation_study/ae_regular_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py regular efficiency norman 2 > ./../../logs/ablation_study/ae_regular_2layer_efficiency_norman.log &

# sena
nohup python3 -u regular_ae.py sena_0 efficiency norman > ./../../logs/ablation_study/ae_sena_0_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_1 efficiency norman > ./../../logs/ablation_study/ae_sena_1_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_2 efficiency norman > ./../../logs/ablation_study/ae_sena_2_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_3 efficiency norman > ./../../logs/ablation_study/ae_sena_3_1layer_efficiency_norman.log &

nohup python3 -u regular_ae.py sena_delta_0 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_1 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_1_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_2 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_2_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_3 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_5 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_efficiency_norman.log &

# l1
nohup python3 -u regular_ae.py l1_3 efficiency norman > ./../../logs/ablation_study/ae_l1_3_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py l1_5 efficiency norman > ./../../logs/ablation_study/ae_l1_5_1layer_efficiency_norman.log &

nohup python3 -u regular_ae.py l1_3 efficiency norman 2 > ./../../logs/ablation_study/ae_l1_3_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py l1_5 efficiency norman 2 > ./../../logs/ablation_study/ae_l1_5_2layer_efficiency_norman.log &

## -------------------------------------------------------- VAE -------------------------------------------------------------------

# regular
nohup python3 -u variational_ae.py regular efficiency norman 1 1 > ./../../logs/ablation_study/vae_regular_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py regular efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_regular_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py regular efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_regular_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py regular efficiency norman 1 10 > ./../../logs/ablation_study/vae_regular_1layer_efficiency_norman_beta10.log &

nohup python3 -u variational_ae.py regular efficiency norman 2 1 > ./../../logs/ablation_study/vae_regular_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py regular efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_regular_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py regular efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_regular_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py regular efficiency norman 2 10 > ./../../logs/ablation_study/vae_regular_2layer_efficiency_norman_beta10.log &

# sena
nohup python3 -u variational_ae.py sena_0 efficiency norman 1 1 > ./../../logs/ablation_study/vae_sena_0_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_0 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_sena_0_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_0 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_sena_0_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_0 efficiency norman 1 10 > ./../../logs/ablation_study/vae_sena_0_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py sena_delta_0 efficiency norman 2 1 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_0 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_delta_0 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_delta_0 efficiency norman 2 10 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_efficiency_norman_beta10.log &

nohup python3 -u variational_ae.py sena_1 efficiency norman 1 1 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_1 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_1 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_1 efficiency norman 1 10 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py sena_delta_1 efficiency norman 2 1 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_1 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_delta_1 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_delta_1 efficiency norman 2 10 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta10.log &

nohup python3 -u variational_ae.py sena_2 efficiency norman 1 1 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_2 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_2 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_2 efficiency norman 1 10 > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py sena_delta_2 efficiency norman 2 1 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_2 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_delta_2 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_delta_2 efficiency norman 2 10 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman_beta10.log &

nohup python3 -u variational_ae.py sena_3 efficiency norman 1 1 > ./../../logs/ablation_study/vae_sena_3_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_3 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_sena_3_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_3 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_sena_3_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_3 efficiency norman 1 10 > ./../../logs/ablation_study/vae_sena_3_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py sena_delta_3 efficiency norman 2 1 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py sena_delta_3 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py sena_delta_3 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py sena_delta_3 efficiency norman 2 10 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_efficiency_norman_beta10.log &

# l1 -> 3
nohup python3 -u variational_ae.py l1_3 efficiency norman 1 1 > ./../../logs/ablation_study/vae_l1_3_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_l1_3_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_l1_3_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 1 10 > ./../../logs/ablation_study/vae_l1_3_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py l1_3 efficiency norman 2 1 > ./../../logs/ablation_study/vae_l1_3_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_l1_3_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_l1_3_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 2 10 > ./../../logs/ablation_study/vae_l1_3_2layer_efficiency_norman_beta10.log &


# l1 -> 5
nohup python3 -u variational_ae.py l1_5 efficiency norman 1 1 > ./../../logs/ablation_study/vae_l1_5_1layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 1 0.1 > ./../../logs/ablation_study/vae_l1_5_1layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 1 0.01 > ./../../logs/ablation_study/vae_l1_5_1layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 1 10 > ./../../logs/ablation_study/vae_l1_5_1layer_efficiency_norman_beta10.log &


nohup python3 -u variational_ae.py l1_5 efficiency norman 2 1> ./../../logs/ablation_study/vae_l1_5_2layer_efficiency_norman_beta1.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 2 0.1 > ./../../logs/ablation_study/vae_l1_5_2layer_efficiency_norman_beta01.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 2 0.01 > ./../../logs/ablation_study/vae_l1_5_2layer_efficiency_norman_beta001.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 2 10 > ./../../logs/ablation_study/vae_l1_5_2layer_efficiency_norman_beta10.log &



# -------------------------------------------------------------- MISCELLANEOUS ----------------------------------------------------
 
## lcorr
nohup python3 -u regular_ae.py sena lcorr norman > ./../../logs/ablation_study/ae_sena_1layer_lcorr_norman.log &

# --> raw

## interpretability
nohup python3 -u regular_ae.py regular interpretability raw > ./../../logs/ablation_study/ae_regular_1layer_interpretability_raw.log &
nohup python3 -u regular_ae.py sena interpretability raw > ./../../logs/ablation_study/ae_sena_1layer_interpretability_raw.log &

