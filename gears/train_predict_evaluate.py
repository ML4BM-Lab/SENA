import argparse  # noqa: D100
import itertools
import os

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mmd_loss import MMDLoss  # type: ignore # noqa: E402
from scipy.sparse import csr_matrix

from gears import GEARS, PertData

DATA_DIR_PATH = "data"
MODELS_DIR_PATH = "models"
RESULTS_DIR_PATH = "results"

PREDICT_SINGLE = False
PREDICT_DOUBLE = True
PREDICT_COMBO = False


def train(
    pert_data: PertData, split: str, seed: int, hidden_size: int, device: str
) -> str:
    """Set up, train, and save GEARS model."""
    print("Training GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.model_initialize(hidden_size=hidden_size)
    gears_model.train(epochs=20)
    model_name = (
        f"gears_norman_"
        f"split_{split}_"
        f"seed_{str(seed)}_"
        f"hidden_size_{str(hidden_size)}"
    )
    gears_model.save_model(path=os.path.join(MODELS_DIR_PATH, model_name))
    return model_name


def predict(pert_data: PertData, device: str, model_name: str) -> None:
    """Predict with GEARS model."""
    # Load the model.
    print("Loading GEARS model.")
    gears_model = GEARS(pert_data=pert_data, device=device)
    gears_model.load_pretrained(path=os.path.join(MODELS_DIR_PATH, model_name))

    # Get all single perturbations.
    single_perturbations = set(
        [
            c.strip("+ctrl")
            for c in pert_data.adata.obs["condition"]
            if ("ctrl+" in c) or ("+ctrl" in c)
        ]
    )
    print(f"Number of single perturbations: {len(single_perturbations)}")

    # Get all double perturbations.
    double_perturbations = set(
        [c for c in pert_data.adata.obs["condition"] if "ctrl" not in c]
    )
    print(f"Number of double perturbations: {len(double_perturbations)}")

    # Generate all possible double perturbations (combos).
    combo_perturbations = []
    for g1 in single_perturbations:
        for g2 in single_perturbations:
            if g1 == g2:
                continue
            combo_perturbations.append(sorted([g1, g2]))
    combo_perturbations.sort()
    combo_perturbations = list(k for k, _ in itertools.groupby(combo_perturbations))
    print(f"Number of combo perturbations: {len(combo_perturbations)}")

    # Get the names of all measured genes as comma-separated list.
    var_names_str = ",".join(map(str, list(pert_data.adata.var_names)))

    if PREDICT_SINGLE:
        # Predict all single perturbations.
        single_results_file_path = os.path.join(
            RESULTS_DIR_PATH, f"{model_name}_single.csv"
        )
        with open(file=single_results_file_path, mode="w") as f:
            print(f"single,{var_names_str}", file=f)
            for i, g in enumerate(single_perturbations):
                print(f"Predicting single {i+1}/{len(single_perturbations)}: {g}")
                prediction = gears_model.predict(pert_list=[[g]])
                single = next(iter(prediction.keys()))
                expressions = prediction[single]
                expressions_str = ",".join(map(str, expressions))
                print(f"{single},{expressions_str}", file=f)

    if PREDICT_DOUBLE:
        # Predict all double perturbations.
        double_results_file_path = os.path.join(
            RESULTS_DIR_PATH, f"{model_name}_double.csv"
        )
        with open(file=double_results_file_path, mode="w") as f:
            print(f"double,{var_names_str}", file=f)
            for i, d in enumerate(double_perturbations):
                print(f"Predicting double {i+1}/{len(double_perturbations)}: {d}")
                prediction = gears_model.predict(pert_list=[d.split("+")])
                double = next(iter(prediction.keys()))
                expressions = prediction[double]
                expressions_str = ",".join(map(str, expressions))
                print(f"{double},{expressions_str}", file=f)

    if PREDICT_COMBO:
        # Predict all combo perturbations.
        combo_results_file_path = os.path.join(
            RESULTS_DIR_PATH, f"{model_name}_combo.csv"
        )
        with open(file=combo_results_file_path, mode="w") as f:
            print(f"combo,{var_names_str}", file=f)
            for i, c in enumerate(combo_perturbations):
                print(f"Predicting combo {i+1}/{len(combo_perturbations)}: {c}")
                prediction = gears_model.predict(pert_list=[c])
                combo = next(iter(prediction.keys()))
                expressions = prediction[combo]
                expressions_str = ",".join(map(str, expressions))
                print(f"{combo},{expressions_str}", file=f)


def evaluate_double(adata: AnnData, model_name: str):
    """Evaluate the predicted GEPs of double perturbations."""
    # Load predicted GEPs.
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(RESULTS_DIR_PATH, f"{model_name}_double.csv")
    )

    # Make results file path.
    results_file_path = os.path.join(
        RESULTS_DIR_PATH, f"{model_name}_double_metrics.csv"
    )

    with open(file=results_file_path, mode="w") as f:
        print(
            "double,num_cells,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred",
            file=f,
        )

        for i, double in enumerate(df["double"]):
            # Get the predicted GEP for the current double perturbation.
            pred_gep = df.loc[df["double"] == double]
            pred_gep = df.iloc[0, 1:].tolist()

            # Get all the true GEPs with the current double perturbation.
            double = double.replace("_", "+")
            print(f"Evaluating double {i+1}/{len(df['double'])}: {double}")
            true_geps = adata[adata.obs["condition"] == double]
            n = true_geps.n_obs

            # Get n random control GEPs.
            all_ctrl_geps = adata[adata.obs["condition"] == "ctrl"]
            random_indices = np.random.choice(
                all_ctrl_geps.n_obs, size=n, replace=False
            )
            ctrl_geps = all_ctrl_geps[random_indices, :]

            # Copy the predicted GEP n times.
            pred_geps = csr_matrix(np.tile(pred_gep, reps=(n, 1)))

            # Convert to tensors.
            true_geps_tensor = torch.tensor(true_geps.X.toarray())
            ctrl_geps_tensor = torch.tensor(ctrl_geps.X.toarray())
            pred_geps_tensor = torch.tensor(pred_geps.toarray())

            # MMD setup.
            mmd_sigma = 200.0
            kernel_num = 10
            mmd_loss = MMDLoss(fix_sigma=mmd_sigma, kernel_num=kernel_num)

            # Compute MMD for the entire batch.
            mmd_true_vs_ctrl = mmd_loss.forward(
                source=ctrl_geps_tensor, target=true_geps_tensor
            )
            mmd_true_vs_pred = mmd_loss.forward(
                source=pred_geps_tensor, target=true_geps_tensor
            )

            # Compute MSE for the entire batch.
            mse_true_vs_ctrl = torch.mean(
                (true_geps_tensor - ctrl_geps_tensor) ** 2
            ).item()
            mse_true_vs_pred = torch.mean(
                (true_geps_tensor - pred_geps_tensor) ** 2
            ).item()

            print(f"MMD (true vs. control):   {mmd_true_vs_ctrl:10.6f}")
            print(f"MMD (true vs. predicted): {mmd_true_vs_pred:10.6f}")
            print(f"MSE (true vs. control):   {mse_true_vs_ctrl:10.6f}")
            print(f"MSE (true vs. predicted): {mse_true_vs_pred:10.6f}")
            print(
                f"{double},{n},{mmd_true_vs_ctrl},{mmd_true_vs_pred},{mse_true_vs_ctrl},{mse_true_vs_pred}",
                file=f,
            )


def main():  # noqa: D103
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Train, predict, evaluate GEARS.")
    parser.add_argument("--split", type=str, default="no_test", help="Data split.")
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for data splitting."
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Size of the hidden layers."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    args = parser.parse_args()

    # Print all arguments.
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create directories.
    os.makedirs(name=DATA_DIR_PATH, exist_ok=True)
    os.makedirs(name=MODELS_DIR_PATH, exist_ok=True)
    os.makedirs(name=RESULTS_DIR_PATH, exist_ok=True)

    # Load "norman" data.
    print("Loading 'norman' data.")
    norman = PertData(data_path=DATA_DIR_PATH)
    norman.load(data_name="norman")

    # Split data and get dataloaders. This is the same procedure as used for Figure 4 in
    # the GEARS paper.
    # See also: ttps://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py
    print("Preparing data split.")
    norman.prepare_split(split=args.split, seed=args.seed)
    norman.get_dataloader(batch_size=32)

    # Train.
    model_name = train(
        pert_data=norman,
        split=args.split,
        seed=args.seed,
        hidden_size=args.hidden_size,
        device=args.device,
    )

    # Predict.
    predict(pert_data=norman, device=args.device, model_name=model_name)

    # Evaluate.
    evaluate_double(adata=norman.adata, model_name=model_name)


if __name__ == "__main__":
    main()
