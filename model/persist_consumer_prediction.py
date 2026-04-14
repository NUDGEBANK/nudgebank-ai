from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import create_engine, text

from export_source_tables import build_database_url, load_environment
from predict_next_month_consumption import predict_next_month_total
from train_consumption_xgboost import ARTIFACT_DIR


def _to_month_start(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 6)
    return pd.to_datetime(text + "01", format="%Y%m%d", errors="coerce")


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

    payload = rows.to_dict(orient="records")
    with engine.begin() as connection:
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
