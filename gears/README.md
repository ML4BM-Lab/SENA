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
python3 src/training.py > results/gears_norman_no_test_training.log 2>&1
```

Inference:
```shell
python3 src/inference.py
```

Evaluation (outside of Docker container):
```shell
python  src/evaluation.py
```
