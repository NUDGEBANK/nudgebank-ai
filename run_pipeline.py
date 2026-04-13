from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

from consumption_dataset import main as dataset_main
from export_source_tables import main as export_main
from predict_next_month_consumption import main as predict_main
from train_consumption_xgboost import main as train_main


def run_step(title: str, func) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    func()


def main() -> None:
    run_step("1. Export Source Tables", export_main)
    run_step("2. Build Prediction Datasets", dataset_main)
    run_step("3. Train XGBoost Model", train_main)
    run_step("4. Predict Next Month Consumption", predict_main)


if __name__ == "__main__":
    main()
