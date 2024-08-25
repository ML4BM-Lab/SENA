
## regular architecture
nohup python3 -u run.py --device "cuda:0" --name "full_go_regular" --trainmode "regular" --latdim 70 > ./../../logs/full_go_regular.out &

## our proposed encoder
nohup python3 -u run.py --device "cuda:0" --name "full_go_NA_NA" --trainmode "NA_NA" --latdim 16 > ./../../logs/full_go_NA_NA_.out &

nohup python3 -u run.py --device "cuda:0" --name "full_go_NA+deltas" --trainmode "NA+deltas" --latdim 70 > ./../../logs/full_go_NA+deltas.out &

## a992ed9825f2cb72c4f188d134d94e59da6fa00f

## ablation study
docker exec -it causal bash
cd /wdir/src/sena_ablation_study


nohup python3 -u regular_ae.py regular outlier interpretability > ./../../logs/ablation_study/ae_regular_1layer_outlier.log &
nohup python3 -u regular_ae.py sena outlier interpretability > ./../../logs/ablation_study/ae_sena_1layer_outlier.log &

nohup python3 -u regular_ae.py regular outlier efficiency > ./../../logs/ablation_study/ae_regular_1layer_outlier_efficiency.log &
nohup python3 -u regular_ae.py sena outlier efficiency > ./../../logs/ablation_study/ae_sena_1layer_outlier_efficiency.log &