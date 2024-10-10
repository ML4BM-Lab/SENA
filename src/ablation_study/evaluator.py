import time
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Define the custom NetworkActivityLayer
class NetworkActivityLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        relation_dict: Dict[int, List[int]],
        lambda_parameter: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create sparse weight mask according to relationships
        mask = torch.zeros((input_size, output_size), device=self.device)

        # Set to 1 where there is a relation
        for i in range(input_size):
            for latent_idx in relation_dict.get(i, []):
                mask[i, latent_idx] = 1

        # Include lambda parameter
        self.mask = mask
        self.mask[self.mask == 0] = lambda_parameter

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((output_size, input_size), device=self.device))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, device=self.device))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x @ ((self.weight * self.mask.T).T)
        if self.bias is not None:
            return output + self.bias
        return output

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

# Evaluator class
class Evaluator:
    def __init__(
        self,
        beta: float = 1.0,
        device: Optional[torch.device] = None,
        logging = None,
        epochs = None,
        report_epoch = 50,
    ):
        self.beta = beta
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer: optim.Optimizer = None
        self.model: nn.Module = None
        self.logging = logging
        self.epochs = epochs
        self.report_epoch = report_epoch


    def initialize_optimizer(self) -> None:
        """Initializes the optimizer."""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

    def compute_loss(self, batch: torch.Tensor, output: Any) -> torch.Tensor:

        """Computes the loss for a given batch."""
        if self.mode == "ae":
            recon_x = output
            loss = nn.functional.mse_loss(recon_x, batch)
        elif self.mode == "vae":
            recon_x, mean, var = output
            recon_loss = nn.functional.mse_loss(recon_x, batch)
            kl_div = -0.5 * torch.mean(1 + torch.log(var) - mean.pow(2) - var)
            loss = recon_loss + self.beta * kl_div
        else:
            raise ValueError(f"Unsupported model: {self.mode}")

        if 'l1' in self.encoder_name:
            # Apply L1 regularization over the first layer
            l1_lambda_num = float(self.enc.split('_')[1])
            l1_lambda = int(l1_lambda_num)
            l1_norm = self.model.encoder.weight.abs().sum()
            loss += l1_lambda * l1_norm

        return loss

    def evaluate(self, test_data: torch.Tensor) -> Dict[str, float]:
        """Evaluates the model on the test data and returns metrics."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(test_data.to(self.device))
            if self.mode == "ae":
                recon_x = output
                test_loss = nn.functional.mse_loss(recon_x, test_data.to(self.device)).item()
                metrics = {'test_loss': test_loss}
            elif self.mode == "vae":
                recon_x, mean, var = output
                recon_loss = nn.functional.mse_loss(recon_x, test_data.to(self.device)).item()
                kl_div = -0.5 * torch.mean(1 + torch.log(var) - mean.pow(2) - var).item()
                total_loss = recon_loss + self.beta * kl_div
                metrics = {'test_loss': total_loss, 'test_recon_loss': recon_loss, 'test_kl_loss': kl_div}
            else:
                raise ValueError(f"Unsupported model: {self.model}")
        return metrics

    def run(
        self, model: nn.Module, seed: int, train_loader: DataLoader, test_data: torch.Tensor
    ) -> pd.DataFrame:
        """
        Runs the training and evaluation process.

        Args:
            model (nn.Module): The model to train.
            seed (int): Random seed for reproducibility.
            train_data (torch.Tensor): Training data.
            test_data (torch.Tensor): Testing data.

        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        self.model = model.to(self.device)
        self.mode = self.model.mode
        self.encoder_name = self.model.encoder_name
        self.initialize_optimizer()
        start_time = time.time()
        results = []

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        for epoch in tqdm(range(self.epochs), desc = 'Training'):
            self.model.train()
            epoch_losses = []

            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.compute_loss(batch, output)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            # Evaluation and metrics
            if epoch % self.report_epoch == 0:
                metrics = self.evaluate(test_data)
                metrics.update({
                    'epoch': epoch,
                    'train_loss': np.mean(epoch_losses),
                    'encoder_name': self.encoder_name,
                    'mode': self.mode
                })
                results.append(metrics)

                # Print progress
                self.logging.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {metrics['train_loss']:.4f}, Test Loss: {metrics['test_loss']:.4f}")

        # Compile results into DataFrame
        results_df = pd.DataFrame(results)
        results_df['seed'] = seed
        results_df['time'] = time.time() - start_time

        return results_df

# Sena Model class
class SenaModel(nn.Module):
    def __init__(
        self,
        mode: str, 
        input_size: int,
        latent_size: int,
        nlayers: int,
        relation_dict: Dict[int, List[int]],
        lambda_sena: float = 0.0,
        device: Optional[torch.device] = None,
        encoder_name: str = "sena",
    ):
        super(SenaModel, self).__init__()
        self.mode = mode
        self.encoder_name = encoder_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlayers = nlayers
        self.lrelu = nn.LeakyReLU()

        if self.mode == "ae":
       
            self.first_layer = NetworkActivityLayer(
                input_size = input_size,
                output_size = latent_size,
                relation_dict = relation_dict,
                lambda_parameter = lambda_sena,
                device=self.device
            )
            if self.nlayers == 2:
                self.encoder_hidden = nn.Linear(latent_size, latent_size)

        elif self.mode == "vae":
            
            if self.nlayers == 1:
                self.first_layer = self.encoder_mean = NetworkActivityLayer(
                                                        input_size = input_size,
                                                        output_size = latent_size,
                                                        relation_dict = relation_dict,
                                                        lambda_parameter = lambda_sena,
                                                        device=self.device
                                                        )   
                self.encoder_var = nn.Linear(input_size, latent_size)
            elif self.nlayers == 2:
                self.encoder_mean = nn.Linear(latent_size, latent_size)
                self.encoder_var = nn.Linear(latent_size, latent_size)

        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Any:

        if self.mode == "ae":
            x = self.lrelu(self.first_layer(x))
            if self.nlayers == 2:
                x = self.lrelu(self.encoder_hidden(x))
            x = self.decoder(x)
            return x
        
        elif self.mode == "vae":

            if self.nlayers == 2:
                x = self.lrelu(self.first_layer(x))
            mean = self.encoder_mean(x)
            var = F.softplus(self.encoder_var(x))
            z = self.reparameterize(mean, var)
            x = self.decoder(z)
            return x, mean, var


# MLP/L1 Model class
class MLPModel(nn.Module):
    def __init__(
        self,
        mode: str,  # 'AE' or 'VAE'
        encoder_name:str,
        input_size: int,
        latent_size: int,
        nlayers: int,
        lambda_l1: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super(MLPModel, self).__init__()
        self.mode = mode
        self.encoder_name = encoder_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlayers = nlayers
        self.lambda_l1 = lambda_l1

        # Encoder
        self.lrelu = nn.LeakyReLU()

        if self.mode == "ae":
            self.first_layer = nn.Linear(input_size, latent_size)
            if self.nlayers == 2:
                self.encoder_hidden = nn.Linear(latent_size, latent_size)

        if self.mode == "vae":
            if nlayers == 1:
                self.first_layer = self.encoder_mean = nn.Linear(input_size, latent_size)
                self.encoder_var = nn.Linear(input_size, latent_size)
            else:
                self.encoder_mean = nn.Linear(latent_size, latent_size)
                self.encoder_var = nn.Linear(latent_size, latent_size)

        # Decoder
        self.decoder = nn.Linear(latent_size, input_size)

    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x: torch.Tensor) -> Any:

        if self.mode == "ae":
            x = self.lrelu(self.first_layer(x))
            if self.nlayers == 2:
                x = self.lrelu(self.encoder_hidden(x))
            x = self.decoder(x)
            return x
        
        elif self.mode == "vae":

            if self.nlayers == 2:
                x = self.lrelu(self.first_layer(x))
            mean = self.encoder_mean(x)
            var = F.softplus(self.encoder_var(x))
            z = self.reparameterize(mean, var)
            x = self.decoder(z)
            return x, mean, var