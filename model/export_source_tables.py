from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data_source"

EXPORT_QUERIES = {
    "card_transaction": """
        SELECT
            a.member_id AS consumer_id,
            ct.transaction_id,
            ct.card_id,
            ct.market_id,
            ct.category_id,
            ct.amount,
            ct.transaction_datetime AS approved_at,
            ct.menu_name,
            ct.quantity,
            mk.market_name,
            mc.category_name,
            mc.mcc,
            mem.gender AS sex,
            mem.birth
        FROM card_transaction ct
        JOIN card c
            ON ct.card_id = c.card_id
        JOIN account a
            ON c.account_id = a.account_id
        JOIN member mem
            ON a.member_id = mem.member_id
        LEFT JOIN market mk
            ON ct.market_id = mk.market_id
        LEFT JOIN market_category mc
            ON ct.category_id = mc.category_id
        ORDER BY ct.transaction_datetime
    """,
    "consumer_baseline": """
        SELECT
            member_id AS consumer_id,
            analysis_year_month,
            avg_spending,
            essential_ratio,
            normal_ratio,
            discretionary_ratio,
            risk_ratio,
            volatility,
            volatility_index,
            created_at,
            updated_at
        FROM consumer_baseline
        ORDER BY analysis_year_month, member_id
    """,
    "age_group_baseline": """
        SELECT
            age_group,
            avg_spending,
            essential_ratio,
            normal_ratio,
            discretionary_ratio,
            risk_ratio,
            volatility,
            volatility_index,
            repayment_action
        FROM age_group_baseline
        ORDER BY age_group
    """,
    "member": """
        SELECT
            member_id AS consumer_id,
            id AS login_id,
            name,
            birth,
            gender,
            phone_number,
            created_at
        FROM member
        ORDER BY member_id
    """,
    "account": """
        SELECT
            account_id,
            member_id AS consumer_id,
            account_name,
            account_number,
            balance,
            protected_balance,
            opened_at
        FROM account
        ORDER BY account_id
    """,
    "market_category": """
        SELECT
            category_id,
            category_name,
            mcc
        FROM market_category
        ORDER BY category_id
    """,
}


def load_environment() -> None:
    # In deployment, real environment variables should take precedence.
    # For local development, we allow loading from nearby .env files.
    candidate_env_files = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "bank-backend" / ".env",
    ]
    for env_file in candidate_env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)


def build_database_url() -> str:
    url = os.getenv("NUDGEBANK_DB_URL")
    if url:
        return url

    user = os.getenv("NUDGEBANK_DB_USER", os.getenv("DB_USERNAME", "postgres"))
    password = os.getenv("NUDGEBANK_DB_PASSWORD", os.getenv("DB_PASSWORD", ""))
    host = os.getenv("NUDGEBANK_DB_HOST", os.getenv("DB_HOST", "localhost"))
    port = os.getenv("NUDGEBANK_DB_PORT", "5432")
    database = os.getenv("NUDGEBANK_DB_NAME", "nudgebank")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def export_tables(table_names: list[str] | None = None, output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    table_names = table_names or list(EXPORT_QUERIES.keys())
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_engine(build_database_url())
    exported: dict[str, str] = {}

    with engine.connect() as connection:
        for table_name in table_names:
            if table_name not in EXPORT_QUERIES:
                raise KeyError(
                    f"Unsupported export target '{table_name}'. "
                    f"Available tables: {tuple(EXPORT_QUERIES.keys())}"
                )
            query = text(EXPORT_QUERIES[table_name])
            df = pd.read_sql(query, connection)
            output_path = output_dir / f"{table_name}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            exported[table_name] = str(output_path)

    return exported


def main() -> None:
    load_environment()
    exported = export_tables()
    print("Source tables exported to CSV.")
    print(json.dumps(exported, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
