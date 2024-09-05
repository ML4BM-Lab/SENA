
## wandb: a992ed9825f2cb72c4f188d134d94e59da6fa00f

## uhler's architecture
docker exec -it causal bash
cd /wdir/src/uhler

# regular architecture
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_regular_seed_42.out &
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 13 > ./../../logs/uhler/full_go_regular_seed_13.out &
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 7 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_7.out & 


# our proposed encoder
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_sena_delta_0_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 13 > ./../../logs/uhler/full_go_sena_delta_0_seed_13.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 7 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_7.out & 


nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_sena_delta_1_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 13 > ./../../logs/uhler/full_go_sena_delta_1_seed_13.out &


nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 70 --seed 42 > ./../../logs/uhler/full_go_sena_delta_3_seed_42.out &

# some other experiments
# nohup python3 -u run.py --trainmode "NA_NA" --latdim 16 > ./../../logs/uhler/full_go_NA_NA_.out &


## ------------------------------------------ ablation study ------------------------------------------------
## ablation study
docker exec -it causal bash
cd /wdir/src/sena_ablation_study

# --> topgo

## interpretability

# regular_orig
nohup python3 -u regular_ae.py regular_orig interpretability topgo > ./../../logs/ablation_study/ae_regular_orig_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py regular_orig interpretability topgo 2 > ./../../logs/ablation_study/ae_regular_orig_2layer_interpretability_topgo.log &

# regular_reduced
nohup python3 -u regular_ae.py regular interpretability topgo > ./../../logs/ablation_study/ae_regular_1layer_interpretability_topgo.log &

# sena
nohup python3 -u regular_ae.py sena_0 interpretability topgo > ./../../logs/ablation_study/ae_sena_0_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_1 interpretability topgo > ./../../logs/ablation_study/ae_sena_1_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_3 interpretability topgo > ./../../logs/ablation_study/ae_sena_3_1layer_interpretability_topgo.log &

nohup python3 -u regular_ae.py sena_delta_0 interpretability topgo 2 > ./../../logs/ablation_study/ae_sena_delta_0_2layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_delta_1 interpretability topgo 2 > ./../../logs/ablation_study/ae_sena_delta_1_2layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_delta_3 interpretability topgo 2 > ./../../logs/ablation_study/ae_sena_delta_3_2layer_interpretability_topgo.log &

nohup python3 -u regular_ae.py sena_bias_0 interpretability topgo > ./../../logs/ablation_study/ae_sena_bias_0_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_bias_1 interpretability topgo > ./../../logs/ablation_study/ae_sena_bias_1_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py sena_bias_3 interpretability topgo > ./../../logs/ablation_study/ae_sena_bias_3_1layer_interpretability_topgo.log &

# l1 
nohup python3 -u regular_ae.py l1_3 interpretability topgo > ./../../logs/ablation_study/ae_l1_3_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py l1_5 interpretability topgo > ./../../logs/ablation_study/ae_l1_5_1layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py l1_7 interpretability topgo > ./../../logs/ablation_study/ae_l1_7_1layer_interpretability_topgo.log &

nohup python3 -u regular_ae.py l1_3 interpretability topgo 2 > ./../../logs/ablation_study/ae_l1_3_2layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py l1_5 interpretability topgo 2 > ./../../logs/ablation_study/ae_l1_5_2layer_interpretability_topgo.log &
nohup python3 -u regular_ae.py l1_7 interpretability topgo 2 > ./../../logs/ablation_study/ae_l1_7_2layer_interpretability_topgo.log &

## efficiency
nohup python3 -u regular_ae.py regular_orig efficiency topgo > ./../../logs/ablation_study/ae_regular_orig_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py regular efficiency topgo > ./../../logs/ablation_study/ae_regular_1layer_efficiency_topgo.log &

nohup python3 -u regular_ae.py sena_0 efficiency topgo > ./../../logs/ablation_study/ae_sena_0_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py sena_1 efficiency topgo > ./../../logs/ablation_study/ae_sena_1_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py sena_3 efficiency topgo > ./../../logs/ablation_study/ae_sena_3_1layer_efficiency_topgo.log &

nohup python3 -u regular_ae.py sena_bias_0 efficiency topgo > ./../../logs/ablation_study/ae_sena_bias_0_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py sena_bias_1 efficiency topgo > ./../../logs/ablation_study/ae_sena_bias_1_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py sena_bias_3 efficiency topgo > ./../../logs/ablation_study/ae_sena_bias_3_1layer_efficiency_topgo.log &


nohup python3 -u regular_ae.py l1_3 efficiency topgo > ./../../logs/ablation_study/ae_l1_3_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py l1_5 efficiency topgo > ./../../logs/ablation_study/ae_l1_5_1layer_efficiency_topgo.log &
nohup python3 -u regular_ae.py l1_7 efficiency topgo > ./../../logs/ablation_study/ae_l1_7_1layer_efficiency_topgo.log &

# nohup python3 -u regular_ae.py l1_4 efficiency topgo > ./../../logs/ablation_study/ae_l1_4_1layer_efficiency_topgo.log &
# nohup python3 -u regular_ae.py l1_6 efficiency topgo > ./../../logs/ablation_study/ae_l1_6_1layer_efficiency_topgo.log &


## lcorr
nohup python3 -u regular_ae.py sena lcorr topgo > ./../../logs/ablation_study/ae_sena_1layer_lcorr_topgo.log &

# --> raw

## interpretability
nohup python3 -u regular_ae.py regular interpretability raw > ./../../logs/ablation_study/ae_regular_1layer_interpretability_raw.log &
nohup python3 -u regular_ae.py sena interpretability raw > ./../../logs/ablation_study/ae_sena_1layer_interpretability_raw.log &

