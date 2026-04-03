from __future__ import annotations


def recommend_repayment_action(
    risk_ratio: float,
    essential_ratio: float,
    discretionary_ratio: float,
    volatility_index: float,
) -> str:
    if risk_ratio >= 0.15 or discretionary_ratio >= 0.35:
        return "상환 강하게"
    if essential_ratio >= 0.45 or volatility_index >= 1.2:
        return "안정형"
    return "기본"


def score_user(
    risk_ratio: float,
    essential_ratio: float,
    volatility_index: float,
    discretionary_ratio: float = 0.0,
) -> dict:
    score = risk_ratio * 0.5 + volatility_index * 0.3 - essential_ratio * 0.2
    return {
        "score": round(score, 4),
        "repayment_action": recommend_repayment_action(
            risk_ratio=risk_ratio,
            essential_ratio=essential_ratio,
            discretionary_ratio=discretionary_ratio,
            volatility_index=volatility_index,
        ),
    }


if __name__ == "__main__":
    example = score_user(
        risk_ratio=0.12,
        essential_ratio=0.40,
        volatility_index=1.05,
        discretionary_ratio=0.22,
    )
    print(example)
