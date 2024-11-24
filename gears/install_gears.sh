#!/usr/bin/env bash

# Install packages from GEARS' requirements.txt.
pip install numpy pandas tqdm scikit-learn torch torch_geometric scanpy networkx dcor

# Install GEARS package.
pip install cell-gears
