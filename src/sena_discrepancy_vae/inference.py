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
from train import loss_function
from utils import MMD_loss


def evaluate_generated_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temp: float,
    numint: int = 1,
    mode: str = "CMVAE",
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
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

    for X in tqdm(dataloader, desc="evaluating loader"):

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
    _, MSE, KLD, L1 = loss_function(
        pred_y,
        gt_y,
        pred_x,
        gt_x,
        mu,
        var,
        G,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
        matched_IO=True,
    )

    # Compute MMD by batches
    mmd_loss_func = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    batch_size = 16
    num_batches = pred_y.shape[0] // batch_size
    MMD_list = []

    for i in tqdm(range(num_batches), desc="computing MMD by batches"):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        mmd_value = mmd_loss_func(
            pred_y[start_idx:end_idx], gt_y[start_idx:end_idx]
        ).item()
        MMD_list.append(mmd_value)

    MMD = np.mean(MMD_list)

    return MMD, MSE.item(), KLD.item(), L1.item()


def evaluate_single_leftout(
    model: torch.nn.Module,
    savedir: str,
    device: torch.device,
    mode: str,
    temp: float = 1000.0,
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the test data (single left-out).

    Returns:
        MMD, MSE, KLD, L1
    """
    data_path = os.path.join(savedir, "test_data_single_node.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            dataloader = pickle.load(f)
    else:
        raise FileNotFoundError(f"Test data file not found at {data_path}")

    return evaluate_generated_samples(
        model,
        dataloader,
        device,
        temp,
        numint=1,
        mode=mode,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
    )


def evaluate_single_train(
    model: torch.nn.Module,
    savedir: str,
    device: torch.device,
    mode: str,
    temp: float = 1000.0,
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the training data.

    Returns:
        MMD, MSE, KLD, L1
    """
    data_path = os.path.join(savedir, "train_data.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            dataloader = pickle.load(f)
    else:
        raise FileNotFoundError(f"Train data file not found at {data_path}")

    return evaluate_generated_samples(
        model,
        dataloader,
        device,
        temp,
        numint=1,
        mode=mode,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
    )


def evaluate_double(
    model: torch.nn.Module,
    savedir: str,
    device: torch.device,
    mode: str,
    temp: float = 1000.0,
    MMD_sigma: float = 200.0,
    kernel_num: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the double perturbation data.

    Returns:
        MMD, MSE, KLD, L1
    """
    # Get data
    double_data_path = os.path.join(savedir, "double_data.pkl")
    if os.path.exists(double_data_path):
        with open(double_data_path, "rb") as f:
            dataloader = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Double perfurbation data file not found at {double_data_path}"
        )

    return evaluate_generated_samples(
        model,
        dataloader,
        device,
        temp,
        numint=2,
        mode=mode,
        MMD_sigma=MMD_sigma,
        kernel_num=kernel_num,
    )


def evaluate_model(
    model: torch.nn.Module,
    mode: str,
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
    if mode == "test":
        MMD, MSE, KLD, L1 = evaluate_single_leftout(
            model,
            savedir,
            device,
            mode,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
    elif mode == "train":
        MMD, MSE, KLD, L1 = evaluate_single_train(
            model,
            savedir,
            device,
            mode,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
    elif mode == "double":
        MMD, MSE, KLD, L1 = evaluate_double(
            model,
            savedir,
            device,
            mode,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Expected 'train', 'test', or 'double'."
        )

    data = {"Metric": ["MMD", "MSE", "KLD", "L1"], "Values": [MMD, MSE, KLD, L1]}

    df = pd.DataFrame(data)
    return df


def compute_metrics(
    savedir: str, evaluation_types: List[str] = ["double"]
) -> pd.DataFrame:
    """
    Compute metrics for a given model.

    Parameters:
        savedir (str): Path to the saved model and config files.
        evaluation_types (list): list of folds to compute metrics for (['train','test','double'])
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
    temp = config.get("temp", 1000.0)
    seed = config.get("seed", 42)
    latdim = config.get("latdim", 105)
    model_name = config.get("name", "example")

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # init empty
    df_double, df_train, df_test = None, None, None

    # Evaluate model
    if "double" in evaluation_types:
        print("Evaluating on double perturbation data...")
        df_double = evaluate_model(
            model=model,
            mode="double",
            savedir=savedir,
            device=device,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
        df_double["mode"] = "double"

    if "train" in evaluation_types:
        print("Evaluating on training data...")
        df_train = evaluate_model(
            model=model,
            mode="train",
            savedir=savedir,
            device=device,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
        df_train["mode"] = "train"

    if "test" in evaluation_types:
        print("Evaluating on test data...")
        df_test = evaluate_model(
            model=model,
            mode="test",
            savedir=savedir,
            device=device,
            temp=temp,
            MMD_sigma=MMD_sigma,
            kernel_num=kernel_num,
        )
        df_test["mode"] = "test"

    # Collect results
    df = pd.concat([df_double, df_train, df_test]).reset_index(drop=True)
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
        "--evaluation_types",
        nargs="+",
        default=["double"],
        help="Which folds to evaluate (train, test and/or double)",
    )
    args = parser.parse_args()

    metrics_df = compute_metrics(
        savedir=args.savedir, evaluation_types=args.evaluation_types
    )
