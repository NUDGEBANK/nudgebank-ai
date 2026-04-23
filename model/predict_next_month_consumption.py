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

ROLLING_3M_LOWER_MULTIPLIER = 0.5
ROLLING_3M_UPPER_MULTIPLIER = 2.5
CURRENT_MONTH_UPPER_MULTIPLIER = 2.8
CURRENT_MONTH_LOWER_MULTIPLIER = 0.3
HIGH_SPENDING_SURGE_THRESHOLD = 1.2
SURGE_LOWER_MULTIPLIER = 0.8


def _load_latest_feature_rows(dataset_dir: Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    dataset = build_latest_inference_dataset()
    save_latest_inference_dataset(dataset, dataset_dir)
    return dataset


def _clip_with_guardrails(feature_rows: pd.DataFrame, predictions: pd.Series) -> pd.Series:
    raw = pd.to_numeric(predictions, errors="coerce").fillna(0)
    rolling_3m_avg = pd.to_numeric(
        feature_rows.get("rolling_3m_avg_spending", pd.Series(0, index=feature_rows.index)),
        errors="coerce",
    ).fillna(0)
    current_month_total = pd.to_numeric(
        feature_rows.get(
            "monthly_current_month_spending",
            feature_rows.get("total_spending", pd.Series(0, index=feature_rows.index)),
        ),
        errors="coerce",
    ).fillna(0)

    lower_bound = rolling_3m_avg * ROLLING_3M_LOWER_MULTIPLIER
    current_month_floor = current_month_total * CURRENT_MONTH_LOWER_MULTIPLIER
    surge_condition = current_month_total > (rolling_3m_avg * HIGH_SPENDING_SURGE_THRESHOLD)
    surge_floor = rolling_3m_avg * SURGE_LOWER_MULTIPLIER
    lower_bound = lower_bound.where(~surge_condition, surge_floor)
    lower_bound = pd.concat([lower_bound, current_month_floor], axis=1).max(axis=1)
    rolling_upper_bound = rolling_3m_avg * ROLLING_3M_UPPER_MULTIPLIER
    current_month_cap = current_month_total * CURRENT_MONTH_UPPER_MULTIPLIER
    has_rolling_upper_bound = rolling_upper_bound > 0
    has_current_month_cap = current_month_cap > 0

    upper_bound = pd.Series(float("inf"), index=feature_rows.index, dtype="float64")
    available_caps = pd.concat(
        [
            rolling_upper_bound.where(has_rolling_upper_bound),
            current_month_cap.where(has_current_month_cap),
        ],
        axis=1,
    )
    chosen_upper = available_caps.max(axis=1, skipna=True)
    upper_bound = upper_bound.where(chosen_upper.isna(), chosen_upper)

    bounded = raw.clip(lower=lower_bound)
    bounded = bounded.clip(upper=upper_bound)
    return bounded.round(2)


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
    raw_predictions = pd.Series(model.predict(feature_rows), index=feature_rows.index)
    predictions = _clip_with_guardrails(feature_rows, raw_predictions)

    result = feature_rows[[column for column in ("consumer_id", "year_month", "age_group", "sex") if column in feature_rows.columns]].copy()
    result["raw_predicted_next_month_total_spending"] = raw_predictions.round(2)
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
