
## wandb: a992ed9825f2cb72c4f188d134d94e59da6fa00f

## -------------------------------------------------- uhler's architecture --------------------------------------------------------
docker exec -it causal bash
cd /wdir/src/uhler


# regular architecture
nohup python3 -u run.py --trainmode "regular" --latdim 105 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13_latdim105.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13_latdim70.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 35 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13_latdim35.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 10 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13_latdim10.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 5 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_13_latdim5.out & 

# our proposed encoder LAMBDA = 0
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 105 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13_latdim105.out & 
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13_latdim70.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 35 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13_latdim35.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 10 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13_latdim10.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 5 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13_latdim5.out &

# our proposed encoder LAMBDA = 10^-3
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 105 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_3_seed_13_latdim105.out & 
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_3_seed_13_latdim70.out &
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 35 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_3_seed_13_latdim35.out &
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 10 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_3_seed_13_latdim10.out &
nohup python3 -u run.py --trainmode "sena_delta_3" --latdim 5 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_3_seed_13_latdim5.out &

# our proposed encoder LAMBDA = 10^-2
nohup python3 -u run.py --trainmode "sena_delta_2" --latdim 105 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_2_seed_13_latdim105.out &
nohup python3 -u run.py --trainmode "sena_delta_2" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_2_seed_13_latdim70.out &
nohup python3 -u run.py --trainmode "sena_delta_2" --latdim 35 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_2_seed_13_latdim35.out &
nohup python3 -u run.py --trainmode "sena_delta_2" --latdim 10 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_2_seed_13_latdim10.out &
nohup python3 -u run.py --trainmode "sena_delta_2" --latdim 5 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_2_seed_13_latdim5.out &

# our proposed encoder LAMBDA = 10^-1
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 105 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_1_seed_13_latdim105.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_1_seed_13_latdim70.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 35 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_1_seed_13_latdim35.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 10 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_1_seed_13_latdim10.out &
nohup python3 -u run.py --trainmode "sena_delta_1" --latdim 5 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_1_seed_13_latdim5.out &