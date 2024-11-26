FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes \
        curl \
        git \
        wget

RUN pip install --upgrade pip

# Install PyG with optional dependencies.
RUN pip install torch-geometric==2.6.1
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv --find-links https://data.pyg.org/whl/torch-2.4.0+cu118.html

# Install GEARS.
RUN pip install cell-gears
