import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from train import train
from utils import Norman2019DataLoader, Wessel2023HEK293DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  # Logs to a file
        logging.StreamHandler(),  # Also logs to the console
    ],
)


@dataclass
class Options:
    name: str = "example"
    model: str = "sena"
    dataset_name: str = "Norman2019_reduced"
    batch_size: int = 32
    sena_lambda: float = 0
    lr: float = 1e-3
    epochs: int = 100
    grad_clip: bool = False
    mxAlpha: float = 1.0
    mxBeta: float = 1.0
    mxTemp: float = 100.0
    lmbda: float = 0.1
    MMD_sigma: float = 200.0
    kernel_num: int = 10
    matched_IO: bool = False
    latdim: int = 105
    seed: int = 42
    dim: Optional[int] = None
    cdim: Optional[int] = None
    log: bool = False
    mlflow_port: int = 5678


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="./results/",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the training."
    )
    parser.add_argument(
        "--model", type=str, default="sena", help="Model to use for training."
    )
    parser.add_argument("--name", type=str, default="example", help="Name of the run.")
    parser.add_argument("--dataset", type=str, default="Norman2019_reduced", help="Name of the run.")

    parser.add_argument("--latdim", type=int, default=105, help="Latent dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--sena_lambda", type=float, default=0, help="Sena λ value")
    parser.add_argument(
        "--log", action='store_true', help="flow server log system"
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_config(opts: Options, save_dir: str) -> None:
    """Save the configuration options to a JSON file."""
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(opts), f, indent=4)


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args: argparse.Namespace) -> None:
    """Main training function."""

    # Create Options instance with arguments overriding defaults
    opts = Options(
        epochs=args.epochs,
        latdim=args.latdim,
        seed=args.seed,
        model=args.model,
        sena_lambda=args.sena_lambda,
        name=args.name,
        dataset_name=args.dataset,
        log=args.log
    )

    logging.info(f"Configuration: {opts}")

    # Set random seeds
    set_seeds(opts.seed)

    logging.info(f"Loading {opts.dataset_name} dataset")
    if 'Norman2019' in opts.dataset_name:
        data_handler = Norman2019DataLoader(batch_size=opts.batch_size)
    elif opts.dataset_name == 'wessel_hefk293':
        data_handler = Wessel2023HEK293DataLoader(batch_size=opts.batch_size)

    # Get data from single-gene perturbation
    (
        dataloader,
        dataloader2,
        dim,
        cdim,
        ptb_targets,
    ) = data_handler.get_data(mode="train")

    # Get data from double perturbation
    dataloader_double, _, _, _ = data_handler.get_data(mode="test")

    opts.dim = dim
    opts.cdim = cdim

    # Update latent dimension if not specified
    if opts.latdim is None:
        opts.latdim = opts.cdim

    logging.info("Saving configuration dict...")

    # Save configurations and data
    save_config(opts, args.savedir)
    save_pickle(ptb_targets, os.path.join(args.savedir, "ptb_targets.pkl"))
    save_pickle(dataloader2, os.path.join(args.savedir, "test_data_single_node.pkl"))
    save_pickle(dataloader, os.path.join(args.savedir, "train_data.pkl"))
    save_pickle(dataloader_double, os.path.join(args.savedir, "double_data.pkl"))

    # Train the model
    train(
        dataloader=dataloader,
        opts=opts,
        device=args.device,
        savedir=args.savedir,
        logger=logging,
        data_handler=data_handler,
    )


if __name__ == "__main__":
    args = parse_args()

    # Build the save directory path
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    main(args)
