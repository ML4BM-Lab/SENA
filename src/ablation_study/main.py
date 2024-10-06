import argparse
import os
import pickle
from typing import List
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from evaluator import Evaluator, SenaModel, MLPModel
from utils import Norman2019DataLoader
import logging

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    parser = argparse.ArgumentParser(description="Autoencoder/VAE Evaluator")
    parser.add_argument("--mode", type=str, default="ae", choices=["ae", "vae"], help="mode type")
    parser.add_argument("--encoder_name", type=str, default="sena", choices=["sena", "mlp", "l1"])
    parser.add_argument("--nseeds", type=int, default=3, help="Number of random seeds to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--nlayers", type=int, default=1, help="Number of layers in the model")
    parser.add_argument("--dataset", type=str, default="Norman2019_raw", help="Dataset to use")
    parser.add_argument("--num_gene_th", type=int, default=5, help="Number of genes threshold for norman dataset")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for VAE")
    parser.add_argument("--lambda_sena", type=float, default=0, help="Sena λ value")
    parser.add_argument("--lambda_l1", type=float, default=1e-5, help="L1 λ value")
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")

    # Define filename
    filename = os.path.join(
        "results",
        "ablation_study",
        f"{args.mode}_ablation_{args.nlayers}layer_{args.dataset}",
    )
    if args.mode == "vae":
        filename += f"_beta_{args.beta}"
    os.makedirs(filename, exist_ok=True)
    logging.info(f"Results will be saved to {filename}.pickle")

    # Load data
    logging.info(f"Loading dataset: {args.dataset}")
    if "Norman" in args.dataset:
        data_handler = Norman2019DataLoader(dataname=args.dataset)
        data_handler.load_norman_2019_dataset()
        logging.info(
            f"Data loaded: {data_handler.adata.shape[0]} samples, {data_handler.adata.shape[1]} features."
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    # Run evaluator for each seed
    all_results: List[pd.DataFrame] = []

    for seed in range(args.nseeds):
        logging.info(f"Running evaluation for seed {seed}")

        # Split data into training and testing sets
        train_data, test_data = train_test_split(
            torch.tensor(data_handler.adata.X.todense()).float(),
            stratify=data_handler.adata.obs["guide_ids"],
            test_size=0.1,
            random_state=seed,
        )
        logging.info(
            f"Split data: {len(train_data)} training samples, {len(test_data)} testing samples."
        )

        if args.encoder_name == "sena":
            logging.info(
                f"Initializing SenaModel with mode={args.mode}, encoder_name={args.encoder_name}, "
                f"input_size={data_handler.adata.shape[1]}, latent_size={len(data_handler.gos)}, "
                f"nlayers={args.nlayers}, lambda_sena={args.lambda_sena}"
            )
            model = SenaModel(
                mode=args.mode,
                encoder_name=args.encoder_name,
                input_size=data_handler.adata.shape[1],
                latent_size=len(data_handler.gos),
                nlayers=args.nlayers,
                relation_dict=data_handler.rel_dict,
                lambda_sena=args.lambda_sena,
            )
        else:
            logging.info(
                f"Initializing MLPModel with mode={args.mode}, encoder_name={args.encoder_name}, "
                f"input_size={data_handler.adata.shape[1]}, latent_size={len(data_handler.gos)}, "
                f"nlayers={args.nlayers}, lambda_l1={args.lambda_l1}"
            )
            model = MLPModel(
                mode=args.mode,
                encoder_name=args.encoder_name,
                input_size=data_handler.adata.shape[1],
                latent_size=len(data_handler.gos),
                nlayers=args.nlayers,
                lambda_l1=args.lambda_l1,
            )

        # Set up evaluator
        logging.info(f"Setting up evaluator with beta={args.beta}")
        evaluator = Evaluator(beta=args.beta)

        # Run evaluation
        logging.info(f"Starting evaluation for seed {seed}")
        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True
        )
        results_df = evaluator.run(
            model=model,
            seed=seed,
            train_loader=train_loader,
            test_data=test_data,
        )
        logging.info(f"Completed evaluation for seed {seed}")
        all_results.append(results_df)

    #saving
    fpath = os.path.join(filename,'results_summary.tsv')
    logging.info(f"Saving results to {fpath}")
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(fpath,sep='\t')

    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()
