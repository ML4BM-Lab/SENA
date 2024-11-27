Build Docker image:
```shell
docker build -t gears .
```

Run Docker container:
```shell
docker run -it -v $(pwd):/workspace --gpus all gears bash
```

Training:
```shell
python3 src/train_predict_evaluate.py --hidden_size=HIDDEN_SIZE --seed=SEED > results/gears_norman_split_no_test_seed_SEED_hidden_size_HIDDEN_SIZE_log.txt 2>&1
```
