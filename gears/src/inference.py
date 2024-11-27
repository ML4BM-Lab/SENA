import itertools
import os

from gears import GEARS, PertData

file_name = os.path.basename(__file__)

# Set the paths.
data_dir_path = "/workspace/data"
models_dir_path = "/workspace/models"
results_dir_path = "/workspace/results"

# Load "norman" data.
print(f"[{file_name}] Loading 'norman' data from: {data_dir_path}")
norman = PertData(data_path=data_dir_path)
norman.load(data_name="norman")

# Split data and get dataloaders. This is the same
# [procedure](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py) as
# used for Figure 4 in the GEARS paper.
print("[{file_name}] Preparing data split.")
norman.prepare_split(split="no_test", seed=42)
norman.get_dataloader(batch_size=32, test_batch_size=128)

# Load the model.
device = "cuda"
print(f"[{file_name}] Device: {device}")
print(f"[{file_name}] Loading GEARS model.")
gears_model = GEARS(pert_data=norman, device=device)
gears_model.load_pretrained(path=os.path.join(models_dir_path, "gears_norman_no_test"))

# Get all single perturbations.
single_perturbations = set(
    [
        c.strip("+ctrl")
        for c in norman.adata.obs["condition"]
        if ("ctrl+" in c) or ("+ctrl" in c)
    ]
)
print(f"[{file_name}] Number of single perturbations: {len(single_perturbations)}")

# Get all double perturbations.
double_perturbations = set(
    [c for c in norman.adata.obs["condition"] if "ctrl" not in c]
)
print(f"[{file_name}] Number of double perturbations: {len(double_perturbations)}")

# Generate all possible double perturbations (combos).
combo_perturbations = []
for g1 in single_perturbations:
    for g2 in single_perturbations:
        if g1 == g2:
            continue
        combo_perturbations.append(sorted([g1, g2]))
combo_perturbations.sort()
combo_perturbations = list(k for k, _ in itertools.groupby(combo_perturbations))
print(f"[{file_name}] Number of combo perturbations: {len(combo_perturbations)}")

# Get the names of all measured genes as comma-separated list.
var_names_str = ",".join(map(str, list(norman.adata.var_names)))

# Predict all single perturbations.
single_results_file_path = os.path.join(
    results_dir_path, "gears_norman_no_test_single.csv"
)
with open(file=single_results_file_path, mode="w") as f:
    print(f"single,{var_names_str}", file=f)
    for i, g in enumerate(single_perturbations):
        print(f"[{file_name}] Predicting single {i}/{len(single_perturbations)}: {g}")
        prediction = gears_model.predict(pert_list=[[g]])
        single = next(iter(prediction.keys()))
        expressions = prediction[single]
        expressions_str = ",".join(map(str, expressions))
        print(f"{single},{expressions_str}", file=f)

# Predict all double perturbations.
double_results_file_path = os.path.join(
    results_dir_path, "gears_norman_no_test_double.csv"
)
with open(file=double_results_file_path, mode="w") as f:
    print(f"double,{var_names_str}", file=f)
    for i, d in enumerate(double_perturbations):
        print(f"[{file_name}] Predicting double {i}/{len(double_perturbations)}: {d}")
        prediction = gears_model.predict(pert_list=[d.split("+")])
        double = next(iter(prediction.keys()))
        expressions = prediction[double]
        expressions_str = ",".join(map(str, expressions))
        print(f"{double},{expressions_str}", file=f)

# Predict all combo perturbations.
combo_results_file_path = os.path.join(
    results_dir_path, "gears_norman_no_test_combo.csv"
)
with open(file=combo_results_file_path, mode="w") as f:
    print(f"combo,{var_names_str}", file=f)
    for i, c in enumerate(combo_perturbations):
        print(f"[{file_name}] Predicting combo {i}/{len(combo_perturbations)}: {c}")
        prediction = gears_model.predict(pert_list=[c])
        combo = next(iter(prediction.keys()))
        expressions = prediction[combo]
        expressions_str = ",".join(map(str, expressions))
        print(f"{combo},{expressions_str}", file=f)
