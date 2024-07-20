
## regular architecture
nohup python3 -u run.py --device "cuda:0" --name "full_go_regular" --trainmode "regular" > ./../../logs/full_go_regular.out &
nohup python3 -u run.py --device "cuda:0" --name "raw_go_regular" --trainmode "regular" > ./../../logs/raw_go_regular.out &

## our proposed encoder
nohup python3 -u run.py --device "cuda:0" --name "raw_go_NA+deltas" --trainmode "NA+deltas" > ./../../logs/raw_go_NA+deltas.out &


## a992ed9825f2cb72c4f188d134d94e59da6fa00f