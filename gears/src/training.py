import os

import numpy as np

from gears import GEARS, PertData

file_name = os.path.basename(__file__)

# Create directories.
data_dir_path = "/workspace/data"
models_dir_path = "/workspace/models"
results_dir_path = "/workspace/results"
os.makedirs(name=data_dir_path, exist_ok=True)
os.makedirs(name=models_dir_path, exist_ok=True)
os.makedirs(name=results_dir_path, exist_ok=True)

# Load "norman" data.
print(f"[{file_name}] Loading 'norman' data from: {data_dir_path}")
norman = PertData(data_path=data_dir_path)
norman.load(data_name="norman")

for i in range(1):
    seed = np.random.randint(low=0, high=1000)
    print(f"[{file_name}] Seed: {seed}")

    # Split data and get dataloaders. This is the same
    # [procedure](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py) as
    # used for Figure 4 in the GEARS paper.
    print("[{file_name}] Preparing data split.")
    norman.prepare_split(split="no_test", seed=seed)
    norman.get_dataloader(batch_size=32, test_batch_size=128)

    hidden_sizes = [5, 10, 35, 70, 105]
    for hidden_size in hidden_sizes:
        # Set up, train, and save GEARS model.
        device = "cuda"
        print(f"[{file_name}] Device: {device}")
        print(f"[{file_name}] Training GEARS model.")
        gears_model = GEARS(pert_data=norman, device=device)
        gears_model.model_initialize(hidden_size=hidden_size)
        gears_model.train(epochs=20)
        model_name = (
            f"gears_norman_no_test_seed_{str(seed)}_hidden_size_{str(hidden_size)}"
        )
        gears_model.save_model(
            path=os.path.join(
                models_dir_path,
                model_name,
            )
        )
