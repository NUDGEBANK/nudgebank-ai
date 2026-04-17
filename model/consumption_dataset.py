from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "model" / "artifacts" / "consumption_prediction"
DEFAULT_SOURCE_DIR = Path(os.getenv("NUDGEBANK_SOURCE_DIR", REPO_ROOT / "data_source"))

TABLE_FILES = {
    "card_transaction": "card_transaction.csv",
    "consumer_baseline": "consumer_baseline.csv",
    "consumer_monthly_analysis": "consumer_monthly_analysis.csv",
    "age_group_baseline": "age_group_baseline.csv",
}

ID_CANDIDATES = ("consumer_id", "member_id", "user_id")
DATE_CANDIDATES = ("approved_at", "transaction_at", "payment_date", "used_at", "ta_ymd")
AMOUNT_CANDIDATES = ("amount", "amt", "approved_amount", "payment_amount")
AGE_GROUP_CANDIDATES = ("age_group", "age_band")
SEX_CANDIDATES = ("sex", "gender")
L1_CATEGORY_CANDIDATES = ("card_tpbuz_nm_1", "category_l1", "merchant_category_l1")
L2_CATEGORY_CANDIDATES = ("card_tpbuz_nm_2", "category_l2", "merchant_category_l2")
SPENDING_TYPE_CANDIDATES = ("spending_type",)

SPENDING_TYPES = ("essential", "normal", "discretionary", "risk")


@dataclass
class DatasetArtifacts:
    dataset_path: str
    metadata_path: str
    row_count: int
    feature_count: int
    label_column: str


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise KeyError(f"Expected one of columns {tuple(candidates)}, found {tuple(df.columns)}")
    return None


def _coerce_year_month(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"[^0-9]", "", regex=True)
    return text.str.slice(0, 6)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).fillna(0.0)

def _parse_transaction_dates(series: pd.Series) -> pd.Series:
    # Handle mixed datetime formats (with/without fractional seconds, timezone offsets).
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=True)


def load_source_tables(source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for table_name, file_name in TABLE_FILES.items():
        table_path = source_dir / file_name
        if table_path.exists():
            tables[table_name] = pd.read_csv(table_path, encoding="utf-8")
    if "card_transaction" not in tables:
        raise FileNotFoundError(
            "card_transaction.csv is required to build the prediction dataset. "
            f"Expected it under {source_dir}."
        )
    return tables


def _prepare_card_transactions(card_transaction: pd.DataFrame) -> pd.DataFrame:
    df = card_transaction.copy()
    consumer_id_col = _pick_column(df, ID_CANDIDATES)
    date_col = _pick_column(df, DATE_CANDIDATES)
    amount_col = _pick_column(df, AMOUNT_CANDIDATES)
    age_group_col = _pick_column(df, AGE_GROUP_CANDIDATES, required=False)
    sex_col = _pick_column(df, SEX_CANDIDATES, required=False)
    l1_col = _pick_column(df, L1_CATEGORY_CANDIDATES, required=False)
    l2_col = _pick_column(df, L2_CATEGORY_CANDIDATES, required=False)
    spending_type_col = _pick_column(df, SPENDING_TYPE_CANDIDATES, required=False)

    df = df.rename(
        columns={
            consumer_id_col: "consumer_id",
            date_col: "transaction_date",
            amount_col: "amount",
            age_group_col or "": "age_group",
            sex_col or "": "sex",
            l1_col or "": "category_l1",
            l2_col or "": "category_l2",
            spending_type_col or "": "spending_type",
        }
    )

    df["transaction_date"] = _parse_transaction_dates(df["transaction_date"])
    df = df.dropna(subset=["consumer_id", "transaction_date", "amount"]).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"]).copy()
    df["year_month"] = df["transaction_date"].dt.strftime("%Y%m")
    df["day_of_week"] = df["transaction_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour"] = df["transaction_date"].dt.hour.fillna(12).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    return df


def _build_monthly_transaction_features(card_df: pd.DataFrame) -> pd.DataFrame:
    grouped = card_df.groupby(["consumer_id", "year_month"], as_index=False)
    monthly = grouped.agg(
        total_spending=("amount", "sum"),
        transaction_count=("amount", "size"),
        avg_transaction_amount=("amount", "mean"),
        max_transaction_amount=("amount", "max"),
        min_transaction_amount=("amount", "min"),
        weekend_spending=("amount", lambda s: s[card_df.loc[s.index, "is_weekend"] == 1].sum()),
        night_spending=("amount", lambda s: s[card_df.loc[s.index, "is_night"] == 1].sum()),
    )
    monthly["weekend_spending_ratio"] = _safe_divide(monthly["weekend_spending"], monthly["total_spending"])
    monthly["night_spending_ratio"] = _safe_divide(monthly["night_spending"], monthly["total_spending"])
    monthly = monthly.drop(columns=["weekend_spending", "night_spending"])

    for categorical_column in ("spending_type", "category_l1"):
        if categorical_column not in card_df.columns:
            continue
        pivot = (
            card_df.groupby(["consumer_id", "year_month", categorical_column])["amount"]
            .sum()
            .unstack(fill_value=0)
        )
        ratio = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        ratio.columns = [f"{categorical_column}_{column}_ratio" for column in ratio.columns]
        monthly = monthly.merge(
            ratio.reset_index(),
            on=["consumer_id", "year_month"],
            how="left",
        )

    for spending_type in SPENDING_TYPES:
        column = f"spending_type_{spending_type}_ratio"
        if column not in monthly.columns:
            monthly[column] = 0.0

    return monthly.fillna(0)


def _attach_consumer_baseline(monthly_df: pd.DataFrame, consumer_baseline: pd.DataFrame | None) -> pd.DataFrame:
    if consumer_baseline is None or consumer_baseline.empty:
        return monthly_df

    baseline = consumer_baseline.copy()
    consumer_id_col = _pick_column(baseline, ID_CANDIDATES)
    age_group_col = _pick_column(baseline, AGE_GROUP_CANDIDATES, required=False)
    sex_col = _pick_column(baseline, SEX_CANDIDATES, required=False)
    baseline = baseline.rename(
        columns={
            consumer_id_col: "consumer_id",
            age_group_col or "": "age_group",
            sex_col or "": "sex",
        }
    )

    keep_columns = ["consumer_id"]
    for optional in ("age_group", "sex"):
        if optional in baseline.columns:
            keep_columns.append(optional)

    return monthly_df.merge(baseline[keep_columns].drop_duplicates("consumer_id"), on="consumer_id", how="left")


def _attach_consumer_monthly_analysis(
    monthly_df: pd.DataFrame,
    consumer_monthly_analysis: pd.DataFrame | None,
) -> pd.DataFrame:
    if consumer_monthly_analysis is None or consumer_monthly_analysis.empty:
        return monthly_df

    analysis = consumer_monthly_analysis.copy()
    consumer_id_col = _pick_column(analysis, ID_CANDIDATES)
    month_col = _pick_column(analysis, ("year_month", "target_month", "analysis_month"), required=False)
    if month_col is None:
        return monthly_df

    analysis = analysis.rename(columns={consumer_id_col: "consumer_id", month_col: "year_month"})
    analysis["year_month"] = _coerce_year_month(analysis["year_month"])

    join_columns = [column for column in analysis.columns if column not in {"consumer_id", "year_month"}]
    rename_map = {column: f"monthly_{column}" for column in join_columns}
    analysis = analysis.rename(columns=rename_map)
    return monthly_df.merge(analysis, on=["consumer_id", "year_month"], how="left")


def _attach_age_group_baseline(monthly_df: pd.DataFrame, age_group_baseline: pd.DataFrame | None) -> pd.DataFrame:
    if age_group_baseline is None or age_group_baseline.empty or "age_group" not in monthly_df.columns:
        return monthly_df

    baseline = age_group_baseline.copy()
    age_group_col = _pick_column(baseline, AGE_GROUP_CANDIDATES, required=False)
    if age_group_col is None:
        return monthly_df

    baseline = baseline.rename(columns={age_group_col: "age_group"})
    join_columns = [column for column in baseline.columns if column != "age_group"]
    rename_map = {column: f"age_group_{column}" for column in join_columns}
    baseline = baseline.rename(columns=rename_map)
    return monthly_df.merge(baseline, on="age_group", how="left")


def _add_time_series_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy().sort_values(["consumer_id", "year_month"])
    consumer_group = df.groupby("consumer_id")

    df["prev_month_total_spending"] = consumer_group["total_spending"].shift(1)
    df["prev_2month_total_spending"] = consumer_group["total_spending"].shift(2)
    df["spending_diff_1m"] = df["total_spending"] - df["prev_month_total_spending"]
    df["spending_growth_rate_1m"] = _safe_divide(df["spending_diff_1m"], df["prev_month_total_spending"])
    df["rolling_3m_avg_spending"] = (
        consumer_group["total_spending"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_3m_std_spending"] = (
        consumer_group["total_spending"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).std())
    ).fillna(0.0)
    df["months_seen"] = consumer_group.cumcount() + 1
    df["label_next_month_total_spending"] = consumer_group["total_spending"].shift(-1)
    return df


def _finalize_feature_frame(source_dir: Path = DEFAULT_SOURCE_DIR) -> pd.DataFrame:
    tables = load_source_tables(source_dir)
    card_df = _prepare_card_transactions(tables["card_transaction"])
    monthly_df = _build_monthly_transaction_features(card_df)
    monthly_df = _attach_consumer_baseline(monthly_df, tables.get("consumer_baseline"))
    monthly_df = _attach_consumer_monthly_analysis(monthly_df, tables.get("consumer_monthly_analysis"))
    monthly_df = _attach_age_group_baseline(monthly_df, tables.get("age_group_baseline"))
    monthly_df = _add_time_series_features(monthly_df)

    if "sex" in monthly_df.columns:
        monthly_df["sex"] = monthly_df["sex"].astype(str).fillna("UNKNOWN")
    if "age_group" in monthly_df.columns:
        monthly_df["age_group"] = monthly_df["age_group"].astype(str).fillna("UNKNOWN")

    monthly_df = monthly_df.fillna(0)
    return monthly_df


def build_prediction_dataset(source_dir: Path = DEFAULT_SOURCE_DIR) -> pd.DataFrame:
    monthly_df = _finalize_feature_frame(source_dir)
    monthly_df = monthly_df.dropna(subset=["label_next_month_total_spending"]).copy()
    return monthly_df


def build_latest_inference_dataset(source_dir: Path = DEFAULT_SOURCE_DIR) -> pd.DataFrame:
    monthly_df = _finalize_feature_frame(source_dir)
    latest_rows = (
        monthly_df.sort_values(["consumer_id", "year_month"])
        .groupby("consumer_id", as_index=False)
        .tail(1)
        .copy()
    )
    return latest_rows.drop(columns=["label_next_month_total_spending"], errors="ignore")


def save_prediction_dataset(
    dataset: pd.DataFrame,
    output_dir: Path = DEFAULT_DATASET_DIR,
) -> DatasetArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "consumption_prediction_dataset.csv"
    metadata_path = output_dir / "consumption_prediction_dataset_meta.json"

    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    metadata = {
        "row_count": int(len(dataset)),
        "feature_count": int(len(dataset.columns) - 1),
        "label_column": "label_next_month_total_spending",
        "columns": dataset.columns.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return DatasetArtifacts(
        dataset_path=str(dataset_path),
        metadata_path=str(metadata_path),
        row_count=int(len(dataset)),
        feature_count=int(len(dataset.columns) - 1),
        label_column="label_next_month_total_spending",
    )


def save_latest_inference_dataset(
    dataset: pd.DataFrame,
    output_dir: Path = DEFAULT_DATASET_DIR,
) -> DatasetArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "latest_inference_dataset.csv"
    metadata_path = output_dir / "latest_inference_dataset_meta.json"

    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    metadata = {
        "row_count": int(len(dataset)),
        "feature_count": int(len(dataset.columns)),
        "label_column": "",
        "columns": dataset.columns.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return DatasetArtifacts(
        dataset_path=str(dataset_path),
        metadata_path=str(metadata_path),
        row_count=int(len(dataset)),
        feature_count=int(len(dataset.columns)),
        label_column="",
    )


def main() -> None:
    training_dataset = build_prediction_dataset()
    latest_inference_dataset = build_latest_inference_dataset()
    training_artifacts = save_prediction_dataset(training_dataset)
    inference_artifacts = save_latest_inference_dataset(latest_inference_dataset)
    print("Consumption prediction datasets created.")
    print(
        json.dumps(
            {
                "training_dataset": asdict(training_artifacts),
                "latest_inference_dataset": asdict(inference_artifacts),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
