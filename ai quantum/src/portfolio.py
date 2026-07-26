
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


@dataclass
class PortfolioSolution:
    selected: list[str]
    weights: pd.Series
    objective: float
    expected_return: float
    volatility: float
    sharpe: float
    concentration: float


def build_selection_qubo(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    cardinality: int,
    risk_aversion: float,
    penalty: float,
) -> np.ndarray:
    """QUBO chọn K tài sản; tỷ trọng cuối cùng được tối ưu ở bước classical."""
    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)
    k = cardinality

    Q = risk_aversion * cov / (k**2)
    Q = Q.copy()
    Q[np.diag_indices_from(Q)] += -mu / k
    Q[np.diag_indices_from(Q)] += penalty * (1 - 2 * k)

    for i in range(len(mu)):
        for j in range(i + 1, len(mu)):
            Q[i, j] += penalty
            Q[j, i] += penalty
    return Q


def selection_objective(
    bits: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> float:
    k = max(int(bits.sum()), 1)
    w = bits / k
    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)
    return float(risk_aversion * w @ cov @ w - mu @ w)


def exact_select(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    cardinality: int,
    risk_aversion: float,
) -> np.ndarray:
    n = len(expected_returns)
    best_value = np.inf
    best_bits = np.zeros(n, dtype=int)

    for idx in combinations(range(n), cardinality):
        bits = np.zeros(n, dtype=int)
        bits[list(idx)] = 1
        value = selection_objective(bits, expected_returns, covariance, risk_aversion)
        if value < best_value:
            best_value = value
            best_bits = bits
    return best_bits


def optimize_weights(
    selected_assets: list[str],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    max_weight: float,
) -> PortfolioSolution:
    if not selected_assets:
        raise ValueError("Danh sách tài sản được chọn đang rỗng.")

    mu = expected_returns.loc[selected_assets].to_numpy(dtype=float)
    cov = covariance.loc[selected_assets, selected_assets].to_numpy(dtype=float)
    n = len(selected_assets)
    feasible_cap = max(max_weight, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        return float(risk_aversion * w @ cov @ w - mu @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, feasible_cap) for _ in range(n)]
    x0 = np.repeat(1.0 / n, n)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    weights_arr = result.x if result.success else x0
    weights_arr = np.clip(weights_arr, 0.0, None)
    weights_arr = weights_arr / weights_arr.sum()

    weights = pd.Series(weights_arr, index=selected_assets, name="Weight")
    exp_return = float(weights_arr @ mu)
    volatility = float(np.sqrt(max(weights_arr @ cov @ weights_arr, 0.0)))
    sharpe = exp_return / volatility if volatility > 1e-12 else np.nan
    concentration = float(np.sum(weights_arr**2))

    return PortfolioSolution(
        selected=selected_assets,
        weights=weights,
        objective=objective(weights_arr),
        expected_return=exp_return,
        volatility=volatility,
        sharpe=sharpe,
        concentration=concentration,
    )


def evaluate_realized(
    daily_returns: pd.DataFrame,
    weights: pd.Series,
    evaluation_days: int = 60,
) -> dict[str, float]:
    common = [c for c in weights.index if c in daily_returns.columns]
    w = weights.loc[common].to_numpy(dtype=float)
    w = w / w.sum()
    r = daily_returns[common].tail(evaluation_days).to_numpy(dtype=float) @ w

    cumulative = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / peak - 1.0

    ann_return = float((cumulative[-1] ** (TRADING_DAYS / len(r))) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan

    return {
        "Realized Return": ann_return,
        "Realized Volatility": ann_vol,
        "Realized Sharpe": sharpe,
        "Maximum Drawdown": float(np.min(drawdown)),
    }


def equal_weight_solution(
    assets: list[str],
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> PortfolioSolution:
    weights = pd.Series(1.0 / len(assets), index=assets, name="Weight")
    mu = expected_returns.loc[assets].to_numpy(dtype=float)
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    w = weights.to_numpy()
    exp_return = float(w @ mu)
    volatility = float(np.sqrt(max(w @ cov @ w, 0.0)))

    return PortfolioSolution(
        selected=assets,
        weights=weights,
        objective=float(risk_aversion * w @ cov @ w - mu @ w),
        expected_return=exp_return,
        volatility=volatility,
        sharpe=exp_return / volatility if volatility > 1e-12 else np.nan,
        concentration=float(np.sum(w**2)),
    )
