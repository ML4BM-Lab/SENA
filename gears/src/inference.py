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
norman.prepare_split(split="no_test", seed=42)  # Used in Fig. 4.
norman.get_dataloader(batch_size=32, test_batch_size=128)

# Load the model.
device = "cuda"
print(f"[{file_name}] Device: {device}")
print(f"[{file_name}] Loading GEARS model.")
gears_model = GEARS(pert_data=norman, device=device)
gears_model.load_pretrained(path=os.path.join(models_dir_path, "gears_norman_no_test"))

# Get all single perturbations.
genes_of_interest = set(
    [
        c.strip("+ctrl")
        for c in norman.adata.obs["condition"]
        if ("ctrl+" in c) or ("+ctrl" in c)
    ]
)
genes_of_interest = [g for g in genes_of_interest if g in list(norman.pert_names)]

# Generate all possible double perturbations (combos).
all_possible_combos = []
for g1 in genes_of_interest:
    for g2 in genes_of_interest:
        if g1 == g2:
            continue
        all_possible_combos.append(sorted([g1, g2]))
all_possible_combos.sort()
all_possible_combos = list(k for k, _ in itertools.groupby(all_possible_combos))

# Get the names of all measured genes as comma-separated list.
var_names_str = ",".join(map(str, list(norman.adata.var_names)))

# Predict all single perturbations.
single_results_file_path = os.path.join(
    results_dir_path, "gears_norman_no_test_single.txt"
)
with open(file=single_results_file_path, mode="w") as f:
    print(f"single,{var_names_str}", file=f)
    for it, g in enumerate(genes_of_interest):
        print(f"[{file_name}] Predicting single {it}/{len(genes_of_interest)}: {g}")
        prediction = gears_model.predict(pert_list=[[g]])
        single = next(iter(prediction.keys()))
        expressions = prediction[single]
        expressions_str = ",".join(map(str, expressions))
        print(f"{single},{expressions_str}", file=f)

# Predict all combo perturbations.
combo_results_file_path = os.path.join(
    results_dir_path, "gears_norman_no_test_combo.txt"
)
with open(file=combo_results_file_path, mode="w") as f:
    print(f"combo,{var_names_str}", file=f)
    for it, c in enumerate(all_possible_combos):
        print(f"[{file_name}] Predicting combo {it}/{len(all_possible_combos)}: {c}")
        prediction = gears_model.predict(pert_list=[c])
        combo = next(iter(prediction.keys()))
        expressions = prediction[combo]
        expressions_str = ",".join(map(str, expressions))
        print(f"{combo},{expressions_str}", file=f)
