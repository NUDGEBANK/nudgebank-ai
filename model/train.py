from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analysis import (
    DEFAULT_DATA_GLOB,
    OUTPUT_DIR,
    load_data,
    prepare_dataframe,
    prepare_logit_data,
    run_logistic_regression,
    sample_for_logit,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model" / "artifacts"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_data(DEFAULT_DATA_GLOB)
    df, _, _ = prepare_dataframe(df)
    try:
        logit_df = sample_for_logit(prepare_logit_data(df))
    except ValueError as error:
        (ARTIFACT_DIR / "logit_summary.txt").write_text(
            f"Model training skipped.\nReason: {error}\n"
            "Use a larger raw dataset via NUDGEBANK_DATA_GLOB to fit the model.",
            encoding="utf-8",
        )
        print("Model training skipped.")
        print(error)
        return

    result, odds_ratio = run_logistic_regression(logit_df)

    odds_ratio.to_csv(ARTIFACT_DIR / "odds_ratio.csv", index=False, encoding="utf-8-sig")
    (ARTIFACT_DIR / "logit_summary.txt").write_text(result.summary().as_text(), encoding="utf-8")

    print("Model training complete.")
    print(f"Artifacts saved to: {ARTIFACT_DIR}")
    print(f"Analysis outputs remain available in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
