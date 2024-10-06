
# Autoencoder/VAE Evaluator

This repository contains a Python script for evaluating autoencoder (AE) and variational autoencoder (VAE) models on a dataset. The script is flexible to use different model architectures and datasets, with support for multiple random seeds.

## Usage

Within the docker container, run:

```bash
python src/ablation_study/main.py --mode {ae|vae} --encoder_name {sena|mlp|l1} [other options]
```

### Arguments

- `--mode`: The mode of the model. Choose between 'ae' (autoencoder) or 'vae' (variational autoencoder). Default is `ae`.
- `--encoder_name`: The encoder architecture. Options are 'sena', 'mlp', or 'l1'. Default is `sena`.
- `--nseeds`: Number of random seeds to run. Default is 5.
- `--batch_size`: Batch size for training. Default is 128.
- `--nlayers`: Number of layers in the model. Default is 1.
- `--dataset`: Dataset to use. Default is `Norman2019_raw`.
- `--num_gene_th`: Threshold for the minimum number of genes within each BP in the Norman dataset. Default is 5.
- `--beta`: Beta parameter for the VAE model. Default is 1.0.
- `--lambda_sena`: λ parameter for the Sena model. Default is 0.
- `--lambda_l1`: L1 regularization parameter. Default is 1e-5.
- `--epochs`: Epochs for training. Default is 250.

### Example

```bash
python src/ablation_study/main.py --mode vae --encoder_name sena --nseeds 3 --nlayers 2 --dataset Norman2019_raw
```

### Output

Results will be saved to the `results/ablation_study/` directory, with a file name based on the selected mode, number of layers, dataset, and beta parameter (if applicable). Results are saved as a `.tsv` file.
