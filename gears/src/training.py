import os

from gears import GEARS, PertData

file_name = os.path.basename(__file__)

# Create directories.
data_dir_path = "data"
models_dir_path = "models"
results_dir_path = "results"
os.makedirs(name=data_dir_path, exist_ok=True)
os.makedirs(name=models_dir_path, exist_ok=True)
os.makedirs(name=results_dir_path, exist_ok=True)

# Load "norman" data.
print(f"[{file_name}] Loading 'norman' data from: {data_dir_path}")
norman = PertData(data_path=data_dir_path)
norman.load(data_name="norman")

# Split data and get dataloaders. This is the same
# [procedure](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py) as
# used for Figure 4 in the GEARS paper.
print("[{file_name}] Preparing data split.")
norman.prepare_split(split="no_test", seed=42)  # Used in Fig. 4.
norman.get_dataloader(batch_size=32, test_batch_size=128)

# Set up, train, and save GEARS model.
device = "cpu"
print(f"[{file_name}] Device: {device}")
print(f"[{file_name}] Training GEARS model.")
gears_model = GEARS(pert_data=norman, device=device)
gears_model.model_initialize()
gears_model.train(epochs=1)
gears_model.save_model(path=os.path.join(models_dir_path, "gears_norman_no_test"))
