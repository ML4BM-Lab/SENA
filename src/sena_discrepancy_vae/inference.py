import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import MMD_loss, LossFunction


def evaluate_generated_samples(
    model: torch.nn.Module,
    loss_f,
    dataloader: DataLoader,
    device: torch.device,
    temp: float,
    numint: int = 1,
    mode: str = "double",
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
    batch_size: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the given dataloader and compute metrics.

    Returns:
        MMD (float): Mean Maximum Mean Discrepancy
        MSE (float): Mean Squared Error
        KLD (float): Kullback-Leibler Divergence
        L1 (float): L1 loss
    """
    model = model.to(device)
    model.eval()

    pred_x_list, gt_x_list = [], []
    gt_y_list, pred_y_list = [], []
    c_y_list, mu_list, var_list = [], [], []
    MSE_l, KLD_l, L1_l, MMD_l = [], [], [], []
    
    #Initialize MMD loss
    mmd_loss_func = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

    for i, X in enumerate(tqdm(dataloader, desc="evaluating loader")):

        x, y, c = X[0].to(device), X[1], X[2].to(device)

        if numint == 2:
            idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
            c1 = torch.zeros_like(c).to(device)
            c1[:, idx[0]] = 1
            c2 = torch.zeros_like(c).to(device)
            c2[:, idx[1]] = 1
        else:
            c1 = c2 = c

        with torch.no_grad():
            y_hat, x_recon, z_mu, z_var, G, _ = model(
                x, c1, c2, num_interv=numint, temp=temp
            )

        gt_x_list.append(x.cpu())
        pred_x_list.append(x_recon.cpu())

        gt_y_list.append(y)
        pred_y_list.append(y_hat.cpu())

        c_y_list.append(c.cpu())
        mu_list.append(z_mu.cpu())
        var_list.append(z_var.cpu())

        if not i % batch_size:

            # Stack tensors
            gt_x = torch.vstack(gt_x_list)
            pred_x = torch.vstack(pred_x_list)
            gt_y = torch.vstack(gt_y_list)
            pred_y = torch.vstack(pred_y_list)
            c_y = torch.vstack(c_y_list)
            mu = torch.vstack(mu_list)
            var = torch.vstack(var_list)
            G = model.G.cpu()

            # Compute metrics
            _, MSE, KLD, L1 = loss_f.compute_loss(
                pred_y,
                gt_y,
                pred_x,
                gt_x,
                mu,
                var,
                G
            )

            # Compute MMD
            MMD = mmd_loss_func(pred_y, gt_y)

            MSE_l.append(MSE.item())
            KLD_l.append(KLD.item())
            L1_l.append(L1.item())
            MMD_l.append(MMD.item())

            # Reset lists
            pred_x_list, gt_x_list = [], []
            gt_y_list, pred_y_list = [], []
            c_y_list, mu_list, var_list = [], [], []


    return np.mean(MMD_l), np.mean(MSE_l), np.mean(KLD_l), np.mean(L1_l)

def evaluate_model_generic(
    model: torch.nn.Module,
    loss_f,
    savedir: str,
    device: torch.device,
    mode: str,
    temp: float = 1000.0,
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the given data type (single left-out, single train, or double perturbation).

    Args:
        model (torch.nn.Module): The model to evaluate.
        loss_f: Loss function class
        savedir (str): Directory where the data files are stored.
        device (torch.device): Device on which the model is run.
        mode (str): Evaluation mode (e.g., 'test', 'train', etc.).
        data_type (str): The type of data to evaluate ('single_leftout', 'single_train', 'double').
        temp (float): Temperature value for evaluation.
        MMD_sigma (float): Sigma value for MMD calculation.
        kernel_num (int): Number of kernels for MMD.

    Returns:
        Tuple: MMD, MSE, KLD, L1 losses.
    """
    # Define file paths based on data type
    data_file_map = {
        "train": "train_data.pkl",
        "test": "test_data_single_node.pkl",
        "double": "double_data.pkl",
    }

    if mode not in data_file_map:
        raise ValueError(f"Invalid data type: {mode}. Expected 'train', 'test', or 'double'.")

    # Construct the full path to the data file
    data_path = os.path.join(savedir, data_file_map[mode])

    # Load the data
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            dataloader = pickle.load(f)
    else:
        raise FileNotFoundError(f"{mode} data file not found at {data_path}")

    # Determine the number of interventions (numint) based on data type
    numint = 1 if mode in ["train", "test"] else 2

    # Evaluate the model using the loaded data
    return evaluate_generated_samples(
        model,
        loss_f,
        dataloader,
        device,
        temp,
        numint=numint,
        mode=mode,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
    )

def evaluate_model(
    model: torch.nn.Module,
    mode: str,
    loss_f,
    savedir: str,
    device: torch.device,
    temp: float,
    MMD_sigma: float,
    kernel_num: int,
) -> pd.DataFrame:
    """
    Evaluate the model and return metrics in a DataFrame.

    Parameters:
        mode (str): 'train', 'test', or 'double'

    Returns:
        pd.DataFrame: DataFrame containing the metrics.
    """

    if mode not in ['train','test','double']:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected 'train', 'test', or 'double'."
        )

    #compute losses
    MMD, MSE, KLD, L1 = evaluate_model_generic(
        model,
        loss_f,
        savedir,
        device,
        mode,
        temp=temp,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
    )

    #build dataframe
    data = {"Metric": ["MMD", "MSE", "KLD", "L1"], "Values": [MMD, MSE, KLD, L1]}
    df = pd.DataFrame(data)

    return df

def compute_metrics(
    savedir: str, evaluation: List[str] = ["double"]
) -> pd.DataFrame:
    """
    Compute metrics for a given model.

    Parameters:
        savedir (str): Path to the saved model and config files.
        evaluation (list): list of folds to compute metrics for (['train','test','double'])
    Returns:
        pd.DataFrame: DataFrame containing the computed metrics.
    """
    # Load the model
    model_path = os.path.join(savedir, "best_model.pt")
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load config from the savedir
    config_path = os.path.join(savedir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # If config file does not exist, use default values or raise an error
        config = {}
        print(f"Warning: Config file not found in {savedir}. Using default parameters.")

    # Extract parameters from config
    MMD_sigma = config.get("MMD_sigma", 200.0)
    kernel_num = config.get("kernel_num", 10)
    matched_IO = config.get("matched_IO", False)
    temp = config.get("temp", 1000.0)
    seed = config.get("seed", 42)
    latdim = config.get("latdim", 105)
    model_name = config.get("name", "example")

    #init loss function class
    loss_f = LossFunction(MMD_sigma=MMD_sigma, kernel_num=kernel_num, matched_IO=matched_IO)

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # init list
    df_list = []

    for mode in evaluation:
        print(f"Evaluating on {mode} data...")
        df = evaluate_model(
            model=model,
            mode=mode,
            loss_f=loss_f,
            savedir=savedir,
            device=device,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
        df["mode"] = mode  # Add the mode as a column to the DataFrame
        df_list.append(df)  # Append the results to the list

    # Collect results
    df = pd.concat(df_list).reset_index(drop=True)
    df["seed"] = seed
    df["latdim"] = latdim
    df["model_name"] = model_name

    # Save results
    output_path = os.path.join(savedir, f"{model_name}_metrics_summary.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Metrics saved to {output_path}")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute metrics for a trained model.")
    parser.add_argument(
        "--savedir",
        type=str,
        default="./results/example",
        help="Path to the saved model and config files.",
    )
    parser.add_argument(
        "--evaluation",
        nargs="+",
        default=["double"],
        help="Which folds to evaluate (train, test and/or double)",
    )
    args = parser.parse_args()

    metrics_df = compute_metrics(
        savedir=args.savedir, evaluation=args.evaluation
    )
