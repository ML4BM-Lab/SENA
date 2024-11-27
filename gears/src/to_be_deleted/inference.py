import argparse
import itertools
import os

from gears import GEARS, PertData

PREDICT_SINGLE = False
PREDICT_DOUBLE = True
PREDICT_COMBO = False


def main():
    parser = argparse.ArgumentParser(description="Predict with GEARS model.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the trained model."
    )
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Model name: {model_name}")

    # Set the paths.
    data_dir_path = "/workspace/data"
    models_dir_path = "/workspace/models"
    results_dir_path = "/workspace/results"

    # Load "norman" data.
    print(f"Loading 'norman' data from: {data_dir_path}")
    norman = PertData(data_path=data_dir_path)
    norman.load(data_name="norman")

    # Split data and get dataloaders. This is the same
    # [procedure](https://github.com/yhr91/GEARS_misc/blob/main/paper/Fig4_UMAP_train.py) as
    # used for Figure 4 in the GEARS paper.
    print("Preparing data split.")
    norman.prepare_split(split="no_test", seed=42)
    norman.get_dataloader(batch_size=32, test_batch_size=128)

    # Load the model.
    device = "cuda"
    print(f"Device: {device}")
    print("Loading GEARS model.")
    gears_model = GEARS(pert_data=norman, device=device)
    gears_model.load_pretrained(path=os.path.join(models_dir_path, model_name))

    # Get all single perturbations.
    single_perturbations = set(
        [
            c.strip("+ctrl")
            for c in norman.adata.obs["condition"]
            if ("ctrl+" in c) or ("+ctrl" in c)
        ]
    )
    print(f"Number of single perturbations: {len(single_perturbations)}")

    # Get all double perturbations.
    double_perturbations = set(
        [c for c in norman.adata.obs["condition"] if "ctrl" not in c]
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
    var_names_str = ",".join(map(str, list(norman.adata.var_names)))

    if PREDICT_SINGLE:
        # Predict all single perturbations.
        single_results_file_path = os.path.join(
            results_dir_path, f"{model_name}_single.csv"
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
            results_dir_path, f"{model_name}_double.csv"
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
            results_dir_path, f"{model_name}_combo.csv"
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


if __name__ == "__main__":
    main()
