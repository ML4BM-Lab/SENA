import os
import sys

import numpy as np
import pandas as pd

from gears import PertData

sys.path.append(os.path.abspath(os.path.join("..", "src", "sena_discrepancy_vae")))

import torch  # noqa: E402
from utils import MMD_loss  # noqa: E402

MODEL_NAME = "gears_norman_no_test_seed_832_hidden_size_105"

file_name = os.path.basename(__file__)

# Set the paths.
data_dir_path = "data"
models_dir_path = "models"
results_dir_path = "results"

# Load "norman" data.
print(f"[{file_name}] Loading 'norman' data from: {data_dir_path}")
norman = PertData(data_path=data_dir_path)
norman.load(data_name="norman")

# Load predicted GEPs.
df = pd.read_csv(
    filepath_or_buffer=os.path.join(results_dir_path, f"{MODEL_NAME}_double.csv")
)

results_file_path = os.path.join(results_dir_path, f"{MODEL_NAME}_double_metrics.csv")

with open(file=results_file_path, mode="w") as f:
    print(
        "double,num_cells,mmd_true_vs_ctrl,mmd_true_vs_pred,mse_true_vs_ctrl,mse_true_vs_pred",
        file=f,
    )

    for i, double in enumerate(df["double"]):
        mmd_values_true_vs_ctrl = []
        mmd_values_true_vs_pred = []
        mse_values_true_vs_ctrl = []
        mse_values_true_vs_pred = []

        # Get the predicted GEPs for the current double perturbation.
        pred_gep = df.loc[df["double"] == double]
        pred_gep = df.iloc[0, 1:].tolist()

        # Get all the true GEPs with the current double perturbation.
        double = double.replace("_", "+")
        print(f"[{file_name}] Double perturbation {i+1}/{len(df['double'])}: {double}")
        true_geps = norman.adata[norman.adata.obs["condition"] == double]
        N = true_geps.n_obs

        # Get N random control GEPs.
        all_ctrl_geps = norman.adata[norman.adata.obs["condition"] == "ctrl"]
        random_indices = np.random.choice(all_ctrl_geps.n_obs, size=N, replace=False)
        ctrl_geps = all_ctrl_geps[random_indices, :]

        # Compute the MMD and MSE between:
        # - True GEPs and control GEPs.
        # - True GEPs and predicted GEPs.
        for true_gep, ctrl_gep in zip(true_geps, ctrl_geps):
            true_gep = true_gep.X[0, :].toarray().flatten().tolist()
            ctrl_gep = ctrl_gep.X[0, :].toarray().flatten().tolist()

            # MMD setup.
            MMD_sigma: float = 200.0
            kernel_num: int = 10
            mmd_loss = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)

            # Compute the MMD between the true GEP and the control GEP.
            mmd = mmd_loss.forward(
                source=torch.tensor(ctrl_gep).unsqueeze(0),
                target=torch.tensor(true_gep).unsqueeze(0),
            )
            mmd_values_true_vs_ctrl.append(mmd.item())

            # Compute the MSE between the true GEP and the control GEP.
            mse = np.mean((np.array(true_gep) - np.array(ctrl_gep)) ** 2)
            mse_values_true_vs_ctrl.append(mse)

            # Compute the MMD between the true GEP and the predicted GEP.
            mmd = mmd_loss.forward(
                source=torch.tensor(pred_gep).unsqueeze(0),
                target=torch.tensor(true_gep).unsqueeze(0),
            )
            mmd_values_true_vs_pred.append(mmd.item())

            # Compute the MSE between the true GEP and the predicted GEP.
            mse = np.mean((np.array(true_gep) - np.array(pred_gep)) ** 2)
            mse_values_true_vs_pred.append(mse)

        mmd_avg_true_vs_ctrl = np.mean(mmd_values_true_vs_ctrl)
        mmd_avg_true_vs_pred = np.mean(mmd_values_true_vs_pred)
        mse_avg_true_vs_ctrl = np.mean(mse_values_true_vs_ctrl)
        mse_avg_true_vs_pred = np.mean(mse_values_true_vs_pred)
        print(f"[{file_name}] MMD (true vs. control):   {mmd_avg_true_vs_ctrl:10.6f}")
        print(f"[{file_name}] MMD (true vs. predicted): {mmd_avg_true_vs_pred:10.6f}")
        print(f"[{file_name}] MSE (true vs. control):   {mse_avg_true_vs_ctrl:10.6f}")
        print(f"[{file_name}] MSE (true vs. predicted): {mse_avg_true_vs_pred:10.6f}")
        print(
            f"{double},{N},{mmd_avg_true_vs_ctrl},{mmd_avg_true_vs_pred},{mse_avg_true_vs_ctrl},{mse_avg_true_vs_pred}",
            file=f,
        )
