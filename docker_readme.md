docker run --gpus all -v /home/jfuente/data/jfuente/causal_representation_learning/:/wdir/ --name causal -dt pyg/belka bash 
docker restart causal 
docker exec -it causal bash