
# SENA-discrepancy-VAE

## Abstract 

<div style="display: flex; align-items: flex-start;">
<div style="flex: 1;">
Predicting the impact of genomic and drug perturbations in cellular function is
crucial for understanding gene functions and drug effects, ultimately leading to
improved therapies. To this end, Causal Representation Learning (CRL) constitutes
one of the most promising approaches, as it aims to identify the latent factors
that causally govern biological systems, thus facilitating the prediction of the
effect of unseen perturbations. Yet, current CRL methods fail in reconciling
their principled latent representations with known biological processes, leading
to models that are not interpretable. To address this major issue, in this work
we present SENA-discrepancy-VAE, a model based on the recently proposed
CRL method discrepancy-VAE, that produces representations where each latent
factor can be interpreted as the (linear) combination of the activity of a (learned)
set of biological processes. To this extent, we present an encoder, SENA-δ, that
efficiently compute and map biological processes’ activity levels to the latent causal
factors. We show that SENA-discrepancy-VAE achieves predictive performances
on unseen combinations of interventions that are comparable with its original, noninterpretable counterpart, 
while inferring causal latent factors that are biologically meaningful.
</div>
<div style="flex: 1;">
<img src="imgs/model_overview.png" alt="Model overview" style="max-width:100%;">
</div>
</div>


## Overview

SENA-discrepancy-VAE is a **Causal Representation Learning (CRL)** model designed to predict the impact of genomic and drug perturbations on cellular function by mapping biological processes to latent causal factors. The model improves interpretability by leveraging biological processes (BPs) as prior knowledge, allowing the prediction of unseen perturbations while inferring biologically meaningful causal factors.

### Key Features
- **Interpretability:** Latent factors represent a linear combination of biological processes' activity levels.
- **Prediction of unseen perturbations:** Comparable performance to non-interpretable models.
- **Integration of prior biological knowledge**: Employs biological processes (BPs) as prior knowledge to map causal factors.

## Architecture

The SENA-discrepancy-VAE modifies the encoder architecture of the standard discrepancy-VAE by introducing a **SENA-δ encoder**. This encoder is biologically driven, incorporating BPs as masks that guide the mapping from gene expressions to causal factors.

- **SENA Layer:** Summarizes gene expression data to infer BP activity levels.
- **Two-Layer Encoder:** The second layer combines BP activity levels to produce latent factors used in the VAE framework.

For more details on the architecture and methodology, refer to our paper.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ML4BM-Lab/SENA
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset

The model has been evaluated on a large-scale **Perturb-seq** dataset, which profiles gene expression changes in leukemia cells under genetic perturbations.

1. Download the dataset (if applicable) or preprocess your own Perturb-seq data.
2. Ensure that the dataset is preprocessed with filtering, normalization, and log-transformation as described in the paper.

### Running the Model

To train the model on your data:
```bash
python train.py --config configs/default.yaml
```

### Configuration Options

The model is highly configurable using YAML files. Key settings include:
- **Biological Process Masks:** Specify the pathway masks used for training.
- **Model Parameters:** Adjust latent dimension sizes, regularization coefficients (`λ`), and other hyperparameters.

## Results

SENA-discrepancy-VAE achieves:
- Comparable performance to non-interpretable counterparts on unseen combinations of perturbations.
- Enhanced biological interpretability, making latent causal factors traceable to biological processes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
