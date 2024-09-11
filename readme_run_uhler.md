
## wandb: a992ed9825f2cb72c4f188d134d94e59da6fa00f

## -------------------------------------------------- uhler's architecture --------------------------------------------------------
docker exec -it causal bash
cd /wdir/src/uhler

# regular architecture
nohup python3 -u run.py --trainmode "regular" --latdim 105 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim105.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 70 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim70.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 35 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim35.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 10 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim10.out & 
nohup python3 -u run.py --trainmode "regular" --latdim 5 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_regular_seed_42_latdim5.out & 

# our proposed encoder
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 42 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_42.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 13 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_13.out &
nohup python3 -u run.py --trainmode "sena_delta_0" --latdim 70 --seed 7 --epochs 100 > ./../../logs/uhler/full_go_sena_delta_0_seed_7.out & 

# some other experiments
# nohup python3 -u run.py --trainmode "NA_NA" --latdim 16 > ./../../logs/uhler/full_go_NA_NA_.out &