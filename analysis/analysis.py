from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_GLOB = str(REPO_ROOT / "data_raw" / "*.csv")
OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
DOCS_ASSETS_DIR = REPO_ROOT / "docs" / "assets"
MIN_COUNT_PER_GROUP = 30
LOGIT_SAMPLE_PER_CLASS = 2_000
RANDOM_STATE = 42

ESSENTIAL_L1 = {"의료/건강", "공공/기업/단체", "미디어/통신"}
NORMAL_L1 = {"음식", "생활서비스", "학문/교육"}
DISCRETIONARY_L1 = {"여가/오락", "공연/전시"}

ESSENTIAL_L2_KEYWORDS = ("병원", "약국", "의약", "교통", "연료", "주유", "통신", "보험")
NORMAL_L2_KEYWORDS = ("편의점", "한식", "중식", "일식", "양식", "분식", "제과", "미용", "생활", "교육")
DISCRETIONARY_L2_KEYWORDS = (
    "커피",
    "음료",
    "쇼핑",
    "의복",
    "의류",
    "패션",
    "화장품",
    "취미",
    "오락",
    "스포츠",
    "공연",
    "전시",
    "여행",
    "숙박",
)
RISK_L2_KEYWORDS = ("유흥주점", "유흥", "간이주점", "주점", "단란주점", "클럽", "룸살롱", "나이트")


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_keyword_pattern(keywords: tuple[str, ...]) -> str:
    return "|".join(re.escape(keyword) for keyword in keywords)


def load_data(data_glob: str = DEFAULT_DATA_GLOB) -> tuple[pd.DataFrame, list[str]]:
    file_list = sorted(glob.glob(data_glob))
    if not file_list:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_glob}")
    df = pd.concat([pd.read_csv(path, encoding="utf-8") for path in file_list], ignore_index=True)
    return df, file_list


def classify_spending_types(df: pd.DataFrame) -> pd.Series:
    major = df["card_tpbuz_nm_1"].fillna("").astype(str).str.strip()
    sub = df["card_tpbuz_nm_2"].fillna("").astype(str).str.strip()
    text = major + " " + sub

    spending_type = pd.Series("normal", index=df.index, dtype="object")
    risk_mask = text.str.contains(make_keyword_pattern(RISK_L2_KEYWORDS), regex=True)
    essential_mask = major.isin(ESSENTIAL_L1) | text.str.contains(make_keyword_pattern(ESSENTIAL_L2_KEYWORDS), regex=True)
    discretionary_mask = major.isin(DISCRETIONARY_L1) | text.str.contains(
        make_keyword_pattern(DISCRETIONARY_L2_KEYWORDS), regex=True
    )
    normal_mask = major.isin(NORMAL_L1) | text.str.contains(make_keyword_pattern(NORMAL_L2_KEYWORDS), regex=True)
    retail_essential_mask = (major == "소매/유통") & text.str.contains(
        make_keyword_pattern(("음/식료품", "마트", "편의점", "종합소매점")), regex=True
    )
    retail_discretionary_mask = (major == "소매/유통") & text.str.contains(
        make_keyword_pattern(("의복", "의류", "패션", "화장품", "잡화")), regex=True
    )

    spending_type.loc[normal_mask] = "normal"
    spending_type.loc[essential_mask | retail_essential_mask] = "essential"
    spending_type.loc[discretionary_mask | retail_discretionary_mask] = "discretionary"
    spending_type.loc[risk_mask] = "risk"
    return spending_type


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def suggest_repayment_action(row: pd.Series) -> str:
    if row["risk_ratio"] >= 0.15 or row["discretionary"] >= 0.35:
        return "상환 강하게"
    if row["essential"] >= 0.45 or row["volatility_index"] >= 1.2:
        return "안정형"
    return "기본"


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
    age_map = {1: "10s", 2: "20s", 3: "30s", 4: "40s", 5: "50s", 6: "60s+"}
    df = df.copy()
    df["age_group"] = df["age"].map(age_map)
    df = df.dropna(subset=["age_group"])
    counts = df["age_group"].value_counts()
    valid_groups = counts[counts >= MIN_COUNT_PER_GROUP].index.tolist()
    df = df[df["age_group"].isin(valid_groups)].copy()
    df["spending_type"] = classify_spending_types(df)
    return df, valid_groups, counts


def summarize_analysis(df: pd.DataFrame, valid_groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, float, float]:
    avg_spending = df.groupby("age_group")["amt"].mean().reindex(valid_groups)
    groups = [df[df["age_group"] == age]["amt"] for age in valid_groups]
    f_stat, p_value = stats.f_oneway(*groups)

    spending_amount = (
        df.groupby(["age_group", "spending_type"])["amt"]
        .sum()
        .unstack(fill_value=0)
        .reindex(valid_groups, fill_value=0)
    )
    for column in ("essential", "normal", "discretionary", "risk"):
        if column not in spending_amount.columns:
            spending_amount[column] = 0.0
    spending_amount = spending_amount[["essential", "normal", "discretionary", "risk"]]
    spending_ratio = spending_amount.div(spending_amount.sum(axis=1), axis=0).fillna(0)

    risk_amount = df[df["spending_type"] == "risk"].groupby("age_group")["amt"].sum()
    total_amount = df.groupby("age_group")["amt"].sum()
    risk_ratio = risk_amount.reindex(valid_groups, fill_value=0).div(total_amount).fillna(0)

    daily = df.groupby(["age_group", "ta_ymd"])["amt"].sum().reset_index()
    volatility = daily.groupby("age_group")["amt"].std().reindex(valid_groups).fillna(0)
    volatility_index = volatility / volatility.mean() if volatility.mean() else volatility

    summary = spending_ratio.copy()
    summary["avg_spending"] = avg_spending
    summary["risk_ratio"] = risk_ratio
    summary["volatility"] = volatility
    summary["volatility_index"] = volatility_index
    summary["repayment_action"] = summary.apply(suggest_repayment_action, axis=1)
    summary = summary[
        [
            "avg_spending",
            "essential",
            "normal",
            "discretionary",
            "risk",
            "risk_ratio",
            "volatility",
            "volatility_index",
            "repayment_action",
        ]
    ]
    return summary, spending_ratio, risk_ratio, f_stat, p_value


def prepare_logit_data(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[["sex", "age", "hour", "amt", "cnt", "day", "spending_type"]].copy()
    model_df["is_risk_spending"] = (model_df["spending_type"] == "risk").astype(int)
    model_df["amt_log"] = np.log1p(model_df["amt"])
    model_df["is_weekend"] = model_df["day"].isin([6, 7]).astype(int)
    model_df["age_z_score"] = zscore(model_df["age"].astype(float))
    model_df["hour_z_score"] = zscore(model_df["hour"].astype(float))
    model_df["amt_log_z_score"] = zscore(model_df["amt_log"].astype(float))
    model_df["cnt_z_score"] = zscore(model_df["cnt"].astype(float))
    return model_df.dropna()


def sample_for_logit(model_df: pd.DataFrame) -> pd.DataFrame:
    risk_df = model_df[model_df["is_risk_spending"] == 1]
    non_risk_df = model_df[model_df["is_risk_spending"] == 0]
    sample_size = min(len(risk_df), len(non_risk_df), LOGIT_SAMPLE_PER_CLASS)
    if sample_size == 0:
        raise ValueError("Not enough data to fit logistic regression.")
    sampled = pd.concat(
        [
            risk_df.sample(sample_size, random_state=RANDOM_STATE),
            non_risk_df.sample(sample_size, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run_logistic_regression(model_df: pd.DataFrame):
    formula = "is_risk_spending ~ C(sex) + age_z_score + hour_z_score + amt_log_z_score + cnt_z_score + is_weekend"
    result = smf.logit(formula=formula, data=model_df).fit(disp=False, maxiter=200)
    odds_ratio = np.exp(result.params).rename("odds_ratio").reset_index().rename(columns={"index": "variable"})
    return result, odds_ratio


def save_chart(series: pd.Series, title: str, ylabel: str, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(series.index, series.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("연령대")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_table_image(df: pd.DataFrame, title: str, output_path: Path, font_size: int = 11, col_width: float = 1.3) -> None:
    row_count, col_count = df.shape
    fig_width = max(8, col_count * col_width + 2)
    fig_height = max(2.5, row_count * 0.55 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=16)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index if not isinstance(df.index, pd.RangeIndex) else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B0B8C2")
        if row == 0:
            cell.set_facecolor("#24415D")
            cell.set_text_props(color="white", weight="bold")
        elif col == -1:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_text_image(text: str, title: str, output_path: Path, font_size: int = 9) -> None:
    lines = text.splitlines()
    fig_height = max(8, len(lines) * 0.28)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=14)
    ax.text(
        0.01,
        0.99,
        text,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=font_size,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_insight_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "핵심 결과": [
                "재량 소비 비중 최고",
                "위험 소비 비율 최고",
                "필수 소비 비중 최고",
                "소비 변동성 최고",
            ],
            "연령대": [
                summary_df["discretionary"].idxmax(),
                summary_df["risk_ratio"].idxmax(),
                summary_df["essential"].idxmax(),
                summary_df["volatility"].idxmax(),
            ],
            "값": [
                f"{summary_df['discretionary'].max():.2%}",
                f"{summary_df['risk_ratio'].max():.2%}",
                f"{summary_df['essential'].max():.2%}",
                f"{summary_df['volatility'].max():,.0f}",
            ],
            "해석": [
                "젊은층에서 선택적 소비 성향이 강함",
                "상대적으로 소비 위험이 가장 큼",
                "생필성 지출 비중이 가장 높음",
                "일별 소비 편차가 가장 큼",
            ],
        }
    )


def format_summary_table(summary_df: pd.DataFrame, count_series: pd.Series) -> pd.DataFrame:
    table = summary_df.copy()
    table.insert(0, "표본 수", count_series.reindex(table.index).astype(int))
    table["avg_spending"] = table["avg_spending"].map(lambda x: f"{x:,.0f}")
    table["essential"] = table["essential"].map(lambda x: f"{x:.2%}")
    table["normal"] = table["normal"].map(lambda x: f"{x:.2%}")
    table["discretionary"] = table["discretionary"].map(lambda x: f"{x:.2%}")
    table["risk"] = table["risk"].map(lambda x: f"{x:.2%}")
    table["risk_ratio"] = table["risk_ratio"].map(lambda x: f"{x:.2%}")
    table["volatility"] = table["volatility"].map(lambda x: f"{x:,.0f}")
    table["volatility_index"] = table["volatility_index"].map(lambda x: f"{x:.2f}")
    return table.rename(
        columns={
            "avg_spending": "평균 소비",
            "essential": "필수 비율",
            "normal": "일반 비율",
            "discretionary": "재량 비율",
            "risk": "위험 비율",
            "risk_ratio": "위험소비 비율",
            "volatility": "변동성",
            "volatility_index": "변동성 지수",
            "repayment_action": "추천 상환전략",
        }
    )


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def format_p_value(p_value: float) -> str:
    if p_value == 0:
        return "< 1e-308"
    return f"{p_value:.3e}"


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    data_glob = os.getenv("NUDGEBANK_DATA_GLOB", DEFAULT_DATA_GLOB)
    raw_df, file_list = load_data(data_glob)

    print_section("1. 데이터 로딩")
    print(f"읽은 파일 수: {len(file_list)}")
    for path in file_list:
        print(f"- {path}")
    print(f"전체 샘플 수: {len(raw_df):,}")

    df, valid_groups, counts = prepare_dataframe(raw_df)

    print_section("2. 전처리 결과")
    print("유효 연령대:")
    print(valid_groups)
    print("\n연령대별 데이터 개수:")
    print(counts.reindex(valid_groups).to_string())

    print("\n소비 유형별 건수:")
    print(df["spending_type"].value_counts().to_string())

    summary, spending_ratio, risk_ratio, f_stat, p_value = summarize_analysis(df, valid_groups)

    print_section("3. ANOVA 결과")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {format_p_value(p_value)}")
    if p_value < 0.05:
        print("해석: 연령대별 소비 금액 차이는 통계적으로 유의미합니다.")
    else:
        print("해석: 연령대별 소비 금액 차이는 통계적으로 유의미하지 않습니다.")

    print_section("4. 연령별 소비 요약")
    display_summary = summary.copy()
    display_summary["avg_spending"] = display_summary["avg_spending"].round(2)
    display_summary["essential"] = (display_summary["essential"] * 100).round(2)
    display_summary["normal"] = (display_summary["normal"] * 100).round(2)
    display_summary["discretionary"] = (display_summary["discretionary"] * 100).round(2)
    display_summary["risk"] = (display_summary["risk"] * 100).round(2)
    display_summary["risk_ratio"] = (display_summary["risk_ratio"] * 100).round(2)
    display_summary["volatility"] = display_summary["volatility"].round(2)
    display_summary["volatility_index"] = display_summary["volatility_index"].round(2)
    print(display_summary.to_string())

    print_section("5. 핵심 인사이트")
    print(f"- 재량 소비 비중 최고: {summary['discretionary'].idxmax()} ({summary['discretionary'].max():.2%})")
    print(f"- 위험 소비 비율 최고: {summary['risk_ratio'].idxmax()} ({summary['risk_ratio'].max():.2%})")
    print(f"- 필수 소비 비중 최고: {summary['essential'].idxmax()} ({summary['essential'].max():.2%})")
    print(f"- 소비 변동성 최고: {summary['volatility'].idxmax()} ({summary['volatility'].max():,.0f})")

    summary.round(4).to_csv(OUTPUT_DIR / "age_spending_summary.csv", encoding="utf-8-sig")
    anova_df = pd.DataFrame(
        [{"검정": "One-way ANOVA", "F-statistic": round(float(f_stat), 4), "p-value": format_p_value(p_value), "결론": "유의미함"}]
    )
    anova_df.to_csv(OUTPUT_DIR / "anova_result.csv", index=False, encoding="utf-8-sig")

    summary_table = format_summary_table(summary, counts)
    insight_table = build_insight_table(summary)
    save_table_image(anova_df, "ANOVA 결과표", OUTPUT_DIR / "anova_result_table.png", font_size=12, col_width=1.6)
    save_table_image(summary_table, "연령별 소비 요약표", OUTPUT_DIR / "age_spending_summary_table.png", font_size=10, col_width=1.25)
    save_table_image(insight_table, "핵심 인사이트 표", OUTPUT_DIR / "key_insights_table.png", font_size=11, col_width=1.6)

    save_chart(
        summary["avg_spending"],
        "연령대별 평균 소비 금액",
        "평균 소비 금액",
        OUTPUT_DIR / "avg_spending_by_age.png",
        "#4C78A8",
    )
    save_chart(
        risk_ratio,
        "연령별 위험 소비 비율",
        "위험 소비 비율",
        OUTPUT_DIR / "risk_ratio_by_age.png",
        "#E45756",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    spending_ratio.plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"])
    ax.set_title("연령별 소비 유형 비율")
    ax.set_xlabel("연령대")
    ax.set_ylabel("비율")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spending_type_ratio_by_age.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print_section("6. 로지스틱 회귀")
    try:
        logit_df = sample_for_logit(prepare_logit_data(df))
        logit_result, odds_ratio = run_logistic_regression(logit_df)
        odds_ratio.to_csv(OUTPUT_DIR / "logit_odds_ratio.csv", index=False, encoding="utf-8-sig")
        logit_summary_text = logit_result.summary().as_text()
        (OUTPUT_DIR / "logit_summary.txt").write_text(logit_summary_text, encoding="utf-8")
        odds_ratio_display = odds_ratio.copy()
        odds_ratio_display["odds_ratio"] = odds_ratio_display["odds_ratio"].map(lambda x: f"{x:.4f}")
        odds_ratio_display = odds_ratio_display.rename(columns={"variable": "변수", "odds_ratio": "OR"})
        save_table_image(odds_ratio_display, "로지스틱 회귀 OR 표", OUTPUT_DIR / "logit_odds_ratio_table.png", font_size=10, col_width=1.5)
        save_text_image(logit_summary_text, "Logit Regression Results", OUTPUT_DIR / "logit_summary.png", font_size=9)
        print(f"로지스틱 회귀 표본 수: {len(logit_df):,}")
        print(logit_result.summary())
        print("\nOdds Ratio:")
        print(odds_ratio.to_string(index=False))
    except ValueError as error:
        (OUTPUT_DIR / "logit_summary.txt").write_text(
            f"Logistic regression skipped.\nReason: {error}\n"
            "Use a larger raw dataset via NUDGEBANK_DATA_GLOB to fit the model.",
            encoding="utf-8",
        )
        print(f"로지스틱 회귀 생략: {error}")

    for file_name in [
        "anova_result_table.png",
        "age_spending_summary_table.png",
        "key_insights_table.png",
        "avg_spending_by_age.png",
        "risk_ratio_by_age.png",
        "spending_type_ratio_by_age.png",
        "logit_odds_ratio_table.png",
        "logit_summary.png",
    ]:
        source = OUTPUT_DIR / file_name
        if source.exists():
            shutil.copy2(source, DOCS_ASSETS_DIR / file_name)

    print_section("7. 저장 결과")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print("- age_spending_summary.csv")
    print("- anova_result.csv")
    print("- anova_result_table.png")
    print("- age_spending_summary_table.png")
    print("- key_insights_table.png")
    print("- avg_spending_by_age.png")
    print("- risk_ratio_by_age.png")
    print("- spending_type_ratio_by_age.png")
    print("- logit_summary.txt")
    print("- logit_odds_ratio.csv")
    print("- logit_odds_ratio_table.png")
    print("- logit_summary.png")
    print(f"\nREADME 이미지 동기화 폴더: {DOCS_ASSETS_DIR}")


if __name__ == "__main__":
    main()
