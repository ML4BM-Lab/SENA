
## wandb: a992ed9825f2cb72c4f188d134d94e59da6fa00f

## -------------------------------------------------- uhler's architecture --------------------------------------------------------
docker exec -it causal bash
cd /wdir/src/uhler

# regular architecture
#✓
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 35 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim35.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 10 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim10.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 5 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim5.out & 

#✓
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13.out &
#✓
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 7 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_7.out & 

# our proposed encoder
#✓
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_42.out &
#✓
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13.out &
#✓
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 7 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_7.out & 

# sena delta lambda = 0.1
#✓
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_sena_delta_1_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 35 --seed 42 > ./../../logs/uhler/full_go_sena_delta_1_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 10 --seed 42 > ./../../logs/uhler/full_go_sena_delta_1_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 5 --seed 42 > ./../../logs/uhler/full_go_sena_delta_1_seed_42.out &


#✓
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 13 > ./../../logs/uhler/full_go_sena_delta_1_seed_13.out &
#✓
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 7 > ./../../logs/uhler/full_go_sena_delta_1_seed_7.out &

# sena delta lambda = 0.001
#✓
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_sena_delta_3_seed_42.out &
#✓
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 70 --seed 13 > ./../../logs/uhler/full_go_sena_delta_3_seed_13.out &
#✓
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 70 --seed 7 > ./../../logs/uhler/full_go_sena_delta_3_seed_7.out &

# some other experiments
# nohup python3 -u run.py --trainmode "NA_NA" --latdim 16 > ./../../logs/uhler/full_go_NA_NA_.out &

## ------------------------------------------ ablation study (norman) ------------------------------------------------
## ablation study
docker exec -it causal bash
cd /wdir/src/sena_ablation_study


## ---------------------------------------------- interpretability ---------------------------------------------

## ------------------------------------------------ AE --------------------------------------------------------
# regular
nohup python3 -u regular_ae.py regular interpretability norman > ./../../logs/ablation_study/ae_regular_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py regular interpretability norman 2 > ./../../logs/ablation_study/ae_regular_2layer_interpretability_norman.log &

# sena
nohup python3 -u regular_ae.py sena_0 interpretability norman > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_1 interpretability norman > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_3 interpretability norman > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_norman.log &

nohup python3 -u regular_ae.py sena_delta_0 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_1 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_1_2layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_delta_3 interpretability norman 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_interpretability_norman.log &

nohup python3 -u regular_ae.py sena_bias_0 interpretability norman > ./../../logs/ablation_study/ae_sena_bias_0_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_bias_1 interpretability norman > ./../../logs/ablation_study/ae_sena_bias_1_1layer_interpretability_norman.log &
nohup python3 -u regular_ae.py sena_bias_3 interpretability norman > ./../../logs/ablation_study/ae_sena_bias_3_1layer_interpretability_norman.log &

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

## ---------------------------------------------------- efficiency --------------------------------------------------------

## ------------------------------------------------------- AE --------------------------------------------------------------

# regular
nohup python3 -u regular_ae.py regular efficiency norman > ./../../logs/ablation_study/ae_regular_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py regular efficiency norman 2 > ./../../logs/ablation_study/ae_regular_2layer_efficiency_norman.log &

# sena
nohup python3 -u regular_ae.py sena_0 efficiency norman > ./../../logs/ablation_study/ae_sena_0_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_1 efficiency norman > ./../../logs/ablation_study/ae_sena_1_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_2 efficiency norman > ./../../logs/ablation_study/ae_sena_2_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_0.3 efficiency norman > ./../../logs/ablation_study/ae_sena_03_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_0.01 efficiency norman > ./../../logs/ablation_study/ae_sena_001_1layer_efficiency_norman.log &

nohup python3 -u regular_ae.py sena_delta_0 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_1 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_1_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_2 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_2_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_0.3 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_03_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py sena_delta_0.01 efficiency norman 2 > ./../../logs/ablation_study/ae_sena_delta_001_2layer_efficiency_norman.log &

# l1
nohup python3 -u regular_ae.py l1_3 efficiency norman > ./../../logs/ablation_study/ae_l1_3_1layer_efficiency_norman.log &
nohup python3 -u regular_ae.py l1_5 efficiency norman > ./../../logs/ablation_study/ae_l1_5_1layer_efficiency_norman.log &

nohup python3 -u regular_ae.py l1_3 efficiency norman 2 > ./../../logs/ablation_study/ae_l1_3_2layer_efficiency_norman.log &
nohup python3 -u regular_ae.py l1_5 efficiency norman 2 > ./../../logs/ablation_study/ae_l1_5_2layer_efficiency_norman.log &

## -------------------------------------------------------- VAE -------------------------------------------------------------------
# regular
nohup python3 -u variational_ae.py regular efficiency norman > ./../../logs/ablation_study/vae_regular_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py regular efficiency norman 2 > ./../../logs/ablation_study/vae_regular_2layer_efficiency_norman.log &

# sena
nohup python3 -u variational_ae.py sena_0 efficiency norman > ./../../logs/ablation_study/vae_sena_0_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py sena_delta_0 efficiency norman 2 > ./../../logs/ablation_study/vae_sena_delta_0_2layer_efficiency_norman.log &

nohup python3 -u variational_ae.py sena_1 efficiency norman > ./../../logs/ablation_study/vae_sena_1_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py sena_delta_1 efficiency norman 2 > ./../../logs/ablation_study/vae_sena_delta_1_2layer_efficiency_norman.log &

nohup python3 -u variational_ae.py sena_3 efficiency norman > ./../../logs/ablation_study/vae_sena_3_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py sena_delta_3 efficiency norman 2 > ./../../logs/ablation_study/vae_sena_delta_3_2layer_efficiency_norman.log &

# l1
nohup python3 -u variational_ae.py l1_3 efficiency norman > ./../../logs/ablation_study/vae_l1_3_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py l1_3 efficiency norman 2 > ./../../logs/ablation_study/vae_l1_3_2layer_efficiency_norman.log &

nohup python3 -u variational_ae.py l1_5 efficiency norman > ./../../logs/ablation_study/vae_l1_5_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py l1_5 efficiency norman 2 > ./../../logs/ablation_study/vae_l1_5_2layer_efficiency_norman.log &

nohup python3 -u variational_ae.py l1_7 efficiency norman > ./../../logs/ablation_study/vae_l1_7_1layer_efficiency_norman.log &
nohup python3 -u variational_ae.py l1_7 efficiency norman 2 > ./../../logs/ablation_study/vae_l1_7_2layer_efficiency_norman.log &


# -------------------------------------------------------------- MISCELLANEOUS ----------------------------------------------------
 
## lcorr
nohup python3 -u regular_ae.py sena lcorr norman > ./../../logs/ablation_study/ae_sena_1layer_lcorr_norman.log &

# --> raw

## interpretability
nohup python3 -u regular_ae.py regular interpretability raw > ./../../logs/ablation_study/ae_regular_1layer_interpretability_raw.log &
nohup python3 -u regular_ae.py sena interpretability raw > ./../../logs/ablation_study/ae_sena_1layer_interpretability_raw.log &

