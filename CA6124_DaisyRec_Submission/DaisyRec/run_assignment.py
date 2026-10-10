"""Run the configured DaisyRec assignment workflow from PyCharm or a terminal."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "assignment_config.json"


def run(command, label):
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def option_args(values):
    """Convert a parameter dict to DaisyRec's --key=value CLI format."""
    return [f"--{key}={value}" for key, value in values.items() if value is not None]


def main():
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = json.load(stream)

    common = {
        "prepro": config["prepro"],
        "positive_threshold": config["positive_threshold"],
        "test_method": config["test_method"],
        "val_method": config["validation_method"],
        "topk": config["topk"],
        "cand_num": config["candidate_count"],
        "epochs": config["epochs"],
        "gpu": config.get("gpu", ""),
        "sample_method": config["sample_method"],
        "sample_ratio": config["sample_ratio"],
        "loss_type": config["loss_type"],
        "optimizer": config["optimizer"],
    }

    print(f"Python interpreter: {sys.executable}", flush=True)
    for dataset in config["datasets"]:
        for model in config["models"]:
            search_space = config["search_spaces"][model]
            tune_params = {
                **common,
                "algo_name": model,
                "dataset": dataset,
                "hyperopt_trail": config["trials"],
                "optimization_metric": config["optimization_metric"],
                "tune_pack": json.dumps(search_space, separators=(",", ":")),
            }
            run(
                [sys.executable, "tune_hyperopt.py", *option_args(tune_params)],
                f"HyperOpt: {model} / {dataset}",
            )

            best_path = ROOT / "tune_res" / (
                f"best_params_{config['loss_type']}_{model}_{dataset}_"
                f"{config['prepro']}_{config['validation_method']}.json"
            )
            with best_path.open(encoding="utf-8") as stream:
                best_params = json.load(stream)["best_params"]

            test_params = {
                **common,
                "algo_name": model,
                "dataset": dataset,
                **best_params,
            }
            run(
                [sys.executable, "test.py", *option_args(test_params)],
                f"Final test: {model} / {dataset}",
            )

    print("\nAll configured model-dataset experiments completed.")


if __name__ == "__main__":
    main()
