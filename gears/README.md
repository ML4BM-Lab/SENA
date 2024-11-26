Build Docker image:
```shell
docker build -t gears .
```

Run Docker container:
```shell
docker run -it -v $(pwd):/workspace gears bash
```
