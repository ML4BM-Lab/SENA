docker run --gpus all -v /home/jfuente/data_a/jfuente/causal/:/wdir/ --name causal -dt pyg/gennius bash 
docker restart causal 
docker exec -it causal bash