from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_TRANSACTION_PATH = REPO_ROOT / "data_source" / "card_transaction.csv"


def create_mock_history(month_offsets: tuple[int, ...] = (-3, -2, -1)) -> pd.DataFrame:
    if not CARD_TRANSACTION_PATH.exists():
        raise FileNotFoundError(f"Missing source file: {CARD_TRANSACTION_PATH}")

    base_df = pd.read_csv(CARD_TRANSACTION_PATH, encoding="utf-8-sig")
    if base_df.empty:
        raise ValueError("card_transaction.csv is empty.")

    if "approved_at" not in base_df.columns:
        raise KeyError("card_transaction.csv must include approved_at column.")

    base_df["approved_at"] = pd.to_datetime(base_df["approved_at"], utc=True, errors="coerce")
    base_df = base_df.dropna(subset=["approved_at"]).copy()
    if base_df.empty:
        raise ValueError("approved_at values could not be parsed.")

    amount_scale_by_offset = {
        -3: 0.72,
        -2: 0.83,
        -1: 0.91,
    }

    history_frames = [base_df.copy()]
    max_transaction_id = int(pd.to_numeric(base_df["transaction_id"], errors="coerce").max())

    for offset in month_offsets:
        mock_df = base_df.copy()
        mock_df["approved_at"] = mock_df["approved_at"] + pd.DateOffset(months=offset)
        scale = amount_scale_by_offset.get(offset, 0.85)
        mock_df["amount"] = (pd.to_numeric(mock_df["amount"], errors="coerce") * scale).round(2)
        if "transaction_id" in mock_df.columns:
            mock_df["transaction_id"] = range(max_transaction_id + 1, max_transaction_id + 1 + len(mock_df))
            max_transaction_id += len(mock_df)
        history_frames.append(mock_df)

    result = pd.concat(history_frames, ignore_index=True)
    result = result.sort_values("approved_at").reset_index(drop=True)
    result["approved_at"] = result["approved_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    result.to_csv(CARD_TRANSACTION_PATH, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    result = create_mock_history()
    print("Mock multi-month history created for local testing.")
    print(f"Updated rows: {len(result)}")
    print(result["approved_at"].astype(str).str[:7].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
