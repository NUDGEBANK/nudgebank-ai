from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from consumption_dataset import (
    DEFAULT_DATASET_DIR,
    build_latest_inference_dataset,
    save_latest_inference_dataset,
)
from train_consumption_xgboost import ARTIFACT_DIR


def _load_latest_feature_rows(dataset_dir: Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    dataset_path = dataset_dir / "latest_inference_dataset.csv"
    if dataset_path.exists():
        dataset = pd.read_csv(dataset_path, encoding="utf-8-sig")
    else:
        dataset = build_latest_inference_dataset()
        save_latest_inference_dataset(dataset, dataset_dir)
    return dataset


def predict_next_month_total(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    model_path = artifact_dir / "consumption_xgboost.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Model artifact not found. Run model/train_consumption_xgboost.py first."
        )

    model = joblib.load(model_path)
    feature_rows = _load_latest_feature_rows(dataset_dir)
    predictions = model.predict(feature_rows)

    result = feature_rows[[column for column in ("consumer_id", "year_month", "age_group", "sex") if column in feature_rows.columns]].copy()
    result["predicted_next_month_total_spending"] = predictions

    output_path = artifact_dir / "next_month_consumption_predictions.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    result = predict_next_month_total()
    print("Next month consumption prediction complete.")
    print(result.head(10).to_string(index=False))
    print(
        json.dumps(
            {"output_path": str(ARTIFACT_DIR / "next_month_consumption_predictions.csv")},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
