Build Docker image:
```shell
docker build -t gears .
```

Run Docker container:
```shell
docker run -it -v $(pwd):/workspace --gpus all gears bash
```

In the container, run everything with the given parameters:
```shell
python3 -m train_predict_evaluate --seed 42 --hidden_size 64 > log.txt 2>&1
```
