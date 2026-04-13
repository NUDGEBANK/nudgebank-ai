from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from consumption_dataset import DEFAULT_DATASET_DIR, build_prediction_dataset, save_prediction_dataset


ARTIFACT_DIR = DEFAULT_DATASET_DIR / "xgboost_artifacts"
LABEL_COLUMN = "label_next_month_total_spending"
EXCLUDE_COLUMNS = {
    "consumer_id",
    "year_month",
    LABEL_COLUMN,
}


def _load_or_create_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> pd.DataFrame:
    dataset_path = dataset_dir / "consumption_prediction_dataset.csv"
    if dataset_path.exists():
        return pd.read_csv(dataset_path, encoding="utf-8-sig")
    dataset = build_prediction_dataset()
    save_prediction_dataset(dataset, dataset_dir)
    return dataset


def _prepare_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    X = dataset.drop(columns=[LABEL_COLUMN]).copy()
    y = dataset[LABEL_COLUMN].copy()

    categorical_columns = [column for column in X.columns if X[column].dtype == "object" and column not in EXCLUDE_COLUMNS]
    numeric_columns = [
        column for column in X.columns
        if column not in EXCLUDE_COLUMNS and column not in categorical_columns
    ]

    X = X[numeric_columns + categorical_columns]
    return X, y, numeric_columns, categorical_columns


def _time_based_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_months = sorted(dataset["year_month"].astype(str).unique())
    if len(sorted_months) < 2:
        raise ValueError("Need at least two distinct months to create train/validation split.")

    validation_month = sorted_months[-1]
    train_df = dataset[dataset["year_month"].astype(str) < validation_month].copy()
    valid_df = dataset[dataset["year_month"].astype(str) == validation_month].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Time-based split failed because train or validation set is empty.")

    return train_df, valid_df


def train_model(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict[str, object]:
    dataset = _load_or_create_dataset(dataset_dir)
    train_df, valid_df = _time_based_split(dataset)

    X_train, y_train, numeric_columns, categorical_columns = _prepare_features(train_df)
    X_valid, y_valid, _, _ = _prepare_features(valid_df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_columns),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
        "validation_rows": int(len(valid_df)),
        "train_rows": int(len(train_df)),
        "validation_month": str(valid_df["year_month"].iloc[0]),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "consumption_xgboost.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    feature_spec_path = ARTIFACT_DIR / "feature_spec.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    feature_spec_path.write_text(
        json.dumps(
            {
                "label_column": LABEL_COLUMN,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_spec_path": str(feature_spec_path),
        "metrics": metrics,
    }


def main() -> None:
    result = train_model()
    print("Consumption prediction model training complete.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
