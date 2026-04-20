from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from export_source_tables import build_database_url, load_environment
from predict_next_month_consumption import predict_next_month_total
from train_consumption_xgboost import ARTIFACT_DIR

DEFAULT_NEW_WEIGHT = 0.8
DEFAULT_OLD_WEIGHT = 0.2
SURGE_NEW_WEIGHT = 0.7
SURGE_OLD_WEIGHT = 0.3
SURGE_GROWTH_THRESHOLD = 0.5


def _to_month_start(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 6)
    return pd.to_datetime(text + "01", format="%Y%m%d", errors="coerce")


def _to_numeric_scalar(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) == 0:
            return None
        value = value[0]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def _prepare_prediction_rows(predictions: pd.DataFrame, model_version: str) -> pd.DataFrame:
    prepared = predictions.copy()
    prepared = prepared.rename(columns={"consumer_id": "member_id"})
    prepared["member_id"] = pd.to_numeric(prepared["member_id"], errors="coerce").astype("Int64")
    prepared["analysis_year_month"] = _to_month_start(prepared["year_month"])
    prepared["predicted_year_month"] = prepared["analysis_year_month"] + pd.offsets.MonthBegin(1)
    prepared["predicted_total_spending"] = pd.to_numeric(
        prepared["predicted_next_month_total_spending"],
        errors="coerce",
    ).round(2)
    prepared["model_version"] = model_version

    prepared = prepared.dropna(
        subset=["member_id", "analysis_year_month", "predicted_year_month", "predicted_total_spending"]
    ).copy()

    prepared["member_id"] = prepared["member_id"].astype(int)
    return prepared[
        [
            "member_id",
            "analysis_year_month",
            "predicted_year_month",
            "predicted_total_spending",
            "model_version",
        ]
    ]


def _apply_prediction_smoothing(connection, rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    existing_df = pd.read_sql(
        text(
            """
            SELECT
                member_id,
                analysis_year_month,
                predicted_total_spending
            FROM consumer_prediction
            """
        ),
        connection,
    )
    if existing_df.empty:
        return rows

    existing_df["member_id"] = pd.to_numeric(existing_df["member_id"], errors="coerce").astype("Int64")
    existing_df["analysis_year_month"] = pd.to_datetime(existing_df["analysis_year_month"], errors="coerce")
    existing_df["predicted_total_spending"] = pd.to_numeric(existing_df["predicted_total_spending"], errors="coerce")
    existing_df = existing_df.dropna(subset=["member_id", "analysis_year_month", "predicted_total_spending"]).copy()
    if existing_df.empty:
        return rows

    growth_df = pd.read_sql(
        text(
            """
            WITH ranked AS (
                SELECT
                    member_id,
                    analysis_year_month,
                    current_month_spending,
                    LAG(current_month_spending) OVER (
                        PARTITION BY member_id
                        ORDER BY analysis_year_month
                    ) AS prev_month_spending
                FROM consumer_monthly_analysis
            )
            SELECT
                member_id,
                analysis_year_month,
                current_month_spending,
                prev_month_spending
            FROM ranked
            """
        ),
        connection,
    )
    growth_df["member_id"] = pd.to_numeric(growth_df["member_id"], errors="coerce").astype("Int64")
    growth_df["analysis_year_month"] = pd.to_datetime(growth_df["analysis_year_month"], errors="coerce")
    growth_df["current_month_spending"] = pd.to_numeric(growth_df["current_month_spending"], errors="coerce")
    growth_df["prev_month_spending"] = pd.to_numeric(growth_df["prev_month_spending"], errors="coerce")
    growth_df = growth_df.dropna(subset=["member_id", "analysis_year_month"]).copy()

    merged = rows.merge(
        existing_df.rename(columns={"predicted_total_spending": "existing_prediction"}),
        on=["member_id", "analysis_year_month"],
        how="left",
    ).merge(
        growth_df,
        on=["member_id", "analysis_year_month"],
        how="left",
    )
    merged["predicted_total_spending"] = merged["predicted_total_spending"].map(_to_numeric_scalar)
    merged["existing_prediction"] = merged["existing_prediction"].map(_to_numeric_scalar)

    growth_rate = (
        (merged["current_month_spending"] - merged["prev_month_spending"])
        / merged["prev_month_spending"].where(merged["prev_month_spending"] > 0)
    )
    is_surge = growth_rate.fillna(0) >= SURGE_GROWTH_THRESHOLD

    new_weight = pd.Series(DEFAULT_NEW_WEIGHT, index=merged.index, dtype="float64")
    old_weight = pd.Series(DEFAULT_OLD_WEIGHT, index=merged.index, dtype="float64")
    new_weight = new_weight.where(~is_surge, SURGE_NEW_WEIGHT)
    old_weight = old_weight.where(~is_surge, SURGE_OLD_WEIGHT)

    has_existing = merged["existing_prediction"].notna()
    merged.loc[has_existing, "predicted_total_spending"] = (
        merged.loc[has_existing, "predicted_total_spending"] * new_weight[has_existing]
        + merged.loc[has_existing, "existing_prediction"] * old_weight[has_existing]
    ).round(2)

    return merged[
        [
            "member_id",
            "analysis_year_month",
            "predicted_year_month",
            "predicted_total_spending",
            "model_version",
        ]
    ]


def persist_predictions(model_version: str = "xgboost-v1") -> dict[str, object]:
    load_environment()
    predictions = predict_next_month_total()
    rows = _prepare_prediction_rows(predictions, model_version)

    engine = create_engine(build_database_url())
    upsert_sql = text(
        """
        INSERT INTO consumer_prediction (
            member_id,
            analysis_year_month,
            predicted_year_month,
            predicted_total_spending,
            model_version,
            created_at,
            updated_at
        ) VALUES (
            :member_id,
            :analysis_year_month,
            :predicted_year_month,
            :predicted_total_spending,
            :model_version,
            NOW(),
            NOW()
        )
        ON CONFLICT (member_id, analysis_year_month)
        DO UPDATE SET
            predicted_year_month = EXCLUDED.predicted_year_month,
            predicted_total_spending = EXCLUDED.predicted_total_spending,
            model_version = EXCLUDED.model_version,
            updated_at = NOW()
        """
    )

    with engine.begin() as connection:
        rows = _apply_prediction_smoothing(connection, rows)
        payload = rows.to_dict(orient="records")
        if payload:
            connection.execute(upsert_sql, payload)

    return {
        "row_count": len(payload),
        "table": "consumer_prediction",
        "model_version": model_version,
    }


def main() -> None:
    result = persist_predictions()
    print("Consumer prediction persistence complete.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
