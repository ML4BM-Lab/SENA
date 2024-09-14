import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('./../sena_ablation_study/')
import sena_tools as st

class SENAVAE(nn.Module):
    """
    SENA VAE model with SENA layers in the encoder.
    """
    def __init__(
        self, 
        input_size: int, 
        latent_size: int, 
        relation_dict: Dict[str, Any], 
        device: str = 'cuda', 
        sp: float = 0.0
    ):
        super(SENAVAE, self).__init__()

        # Encoder with SENA layers
        self.encoder = st.NetActivity_layer(
            input_size, latent_size, relation_dict, device=device, sp=sp
        )
        self.encoder_var = nn.Linear(input_size, latent_size)  # Log variance for latent space

        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.encoder(x)
        var = F.softplus(self.encoder_var(x))  # Ensure variance is positive
        return mean, var

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var

class SENAVAE_Decoder(nn.Module):
    """
    SENA VAE model with SENA layers in the decoder.
    """
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        relation_dict: Dict[str, Any],
        device: str = 'cuda',
        sp: float = 0.0
    ):
        super(SENAVAE_Decoder, self).__init__()

        # Encoder
        self.encoder_mean = nn.Linear(input_size, latent_size)
        self.encoder_var = nn.Linear(input_size, latent_size)

        # Decoder with SENA layers
        self.decoder = st.NetActivity_layer_rev(
            latent_size, input_size, relation_dict, device=device, sp=sp
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.encoder_mean(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var

    def encoder(self, x):
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        return z

class SENADeltaVAE(nn.Module):
    """
    SENA Delta VAE model with SENA layers in the encoder.
    """
    def __init__(
        self, 
        input_size: int, 
        latent_size: int, 
        relation_dict: Dict[str, Any], 
        device: str = 'cuda', 
        sp: float = 0.0
    ):
        super(SENADeltaVAE, self).__init__()

        # Activation Function
        self.lrelu = nn.LeakyReLU()

        # Encoder with SENA layers
        self.encoder = st.NetActivity_layer(
            input_size, latent_size, relation_dict, device=device, sp=sp
        )
        self.encoder_mean = nn.Linear(latent_size, latent_size)
        self.encoder_var = nn.Linear(latent_size, latent_size)

        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = self.lrelu(x)
        mean = self.encoder_mean(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var

class SENADeltaVAE_Decoder(nn.Module):
    """
    SENA Delta VAE model with SENA layers in the decoder.
    """
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        relation_dict: Dict[str, Any],
        device: str = 'cuda',
        sp: float = 0.0
    ):
        super(SENADeltaVAE_Decoder, self).__init__()

        # Activation Function
        self.lrelu = nn.LeakyReLU()

        # Encoder
        self.encoder_mlp = nn.Linear(input_size, latent_size)
        self.encoder_mean = nn.Linear(latent_size, latent_size)
        self.encoder_var = nn.Linear(latent_size, latent_size)

        # Decoder with SENA layers
        self.decoder = st.NetActivity_layer_rev(
            latent_size, input_size, relation_dict, device=device, sp=sp
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder_mlp(x)
        x = self.lrelu(x)
        mean = self.encoder_mean(x)
        var = F.softplus(self.encoder_var(x))
        return mean, var

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, var
    
    def encoder(self, x):
        mean, var = self.encode(x)
        z = self.reparameterize(mean, var)
        return x

def vae_loss(
    reconstructed_x: torch.Tensor, 
    x: torch.Tensor, 
    mean: torch.Tensor, 
    var: torch.Tensor, 
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')

    # KL Divergence
    kl_divergence = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean.pow(2) - var, dim=1))

    # Total VAE loss
    return reconstruction_loss, beta * kl_divergence

def run_model(
    mode: str, 
    seed: int, 
    analysis: str = 'interpretability', 
    beta: float = 1.0
) -> pd.DataFrame:

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Measure time
    start_time = time.time()

    # Load data
    adata, ptb_targets, ptb_targets_ens, gos, rel_dict, gene_go_dict, ens_gene_dict = st.load_norman_2019_dataset()

    # Split train/test data
    train_data, test_data = train_test_split(
        torch.tensor(adata.X.todense()).float(), 
        stratify=adata.obs['guide_ids'], 
        test_size=0.1
    )
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Initialize model based on the mode
    sp = 0.0  # Default sparsity parameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse sparsity parameter from mode string
    sp_parts = mode.split('_')
    sp_index = 2 if 'delta' in mode or 'decoder' in mode else 1
    sp_num = float(sp_parts[sp_index]) if len(sp_parts) > sp_index else 0
    sp = 10 ** -sp_num if sp_num > 0 else 0.0

    # Select model class based on mode
    if 'enc' in mode:
        if nlayers == 1:
            ModelClass = SENAVAE
        else:
            ModelClass = SENADeltaVAE
    else:
        if nlayers == 1:
            ModelClass = SENAVAE_Decoder
        else:
            ModelClass = SENADeltaVAE_Decoder        

    model = ModelClass(
        input_size=adata.X.shape[1],
        latent_size=len(gos),
        relation_dict=rel_dict,
        device=device,
        sp=sp
    ).to(device)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Training parameters
    epochs = 250
    report_epoch = 10
    results = []

    # Training loop
    for epoch in range(epochs):
        epoch_train_mse = []
        epoch_train_kl = []

        for batched_exp in train_loader:
            
            optimizer.zero_grad()
            reconstructed_x, mean, var = model(batched_exp.to(device))
            train_mse, train_kl = vae_loss(
                reconstructed_x, batched_exp.to(device), mean, var, beta=beta
            )

            total_loss = train_mse + train_kl
            total_loss.backward()
            optimizer.step()

            epoch_train_mse.append(train_mse.item())
            epoch_train_kl.append(train_kl.item())

        # Reporting
        if epoch % report_epoch == 0:
            if analysis == 'interpretability':
                ttest_df = st.compute_activation_df(
                    model, adata, gos, scoretype='mu_diff', mode=mode,
                    gene_go_dict=gene_go_dict, ensembl_genename_dict=ens_gene_dict,
                    ptb_targets=ptb_targets
                )
                summary_analysis_ep = st.compute_outlier_activation_analysis(ttest_df, mode=mode)
                summary_analysis_ep['epoch'] = epoch
            elif analysis == 'efficiency':
                with torch.no_grad():
                    reconstructed_x, mean, var = model(test_data.to(device))
                    test_mse, test_kl = vae_loss(
                        reconstructed_x.cpu(), test_data.cpu(), mean.cpu(), var.cpu(), beta=beta
                    )
                sparsity = 0  # Placeholder for sparsity computation
                summary_analysis_ep = pd.DataFrame({
                    'epoch': epoch,
                    'train_mse': np.mean(epoch_train_mse),
                    'test_mse': test_mse.item(),
                    'train_kl': np.mean(epoch_train_kl),
                    'test_kl': test_kl.item(),
                    'mode': mode,
                    'sparsity': sparsity
                }, index=[0])
                print(f'Epoch {epoch+1}, Test MSE: {test_mse.item()}, Test KL: {test_kl.item()}')

            results.append(summary_analysis_ep)

        print(f'Epoch {epoch+1}, Train MSE: {np.mean(epoch_train_mse)}, Train KL: {np.mean(epoch_train_kl)}')

    # Compile results
    results_df = pd.concat(results)
    results_df['seed'] = seed
    results_df['time'] = time.time() - start_time

    return results_df

if __name__ == '__main__':

    # Define inputs
    model_type = sys.argv[1]
    analysis = sys.argv[2]
    dataset = sys.argv[3]
    num_gene_th = 5 if '_' not in dataset else int(dataset.split('_')[-1])
    nlayers = 1 if len(sys.argv) < 5 else sys.argv[4]
    beta = 1.0 if len(sys.argv) < 5 else float(sys.argv[4])

    # Define seeds
    n_seeds = 2 if analysis == 'efficiency' else 3

    # Initialize paths
    fpath = os.path.join('../../result/encoder_vs_decoder/', f'vae_{model_type}')
    fname = os.path.join(
        fpath, f'vae_{model_type}_encvsdec_{analysis}_{nlayers}layer_{dataset}_beta_{beta}'
    )

    # Create directory if it doesn't exist
    os.makedirs(fpath, exist_ok=True)

    # Run the model for each seed
    all_results = []
    for i in range(n_seeds):
        result_df = run_model(
            mode=model_type, seed=i, analysis=analysis, beta=beta
        )
        all_results.append(result_df)

    # Save results
    final_results_df = pd.concat(all_results).reset_index(drop=True)
    print(final_results_df)
    final_results_df.to_csv(fname + '.tsv', sep='\t', index=False)
