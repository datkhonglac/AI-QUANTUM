
from pathlib import Path
import numpy as np

from src.data_utils import forecast_returns, load_price_data
from src.portfolio import build_selection_qubo, exact_select, optimize_weights
from src.qaoa_numpy import solve_qaoa


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_runs():
    prices = load_price_data(ROOT / "data" / "demo_prices.csv")
    forecast = forecast_returns(prices, lookback=126)

    k = 4
    Q = build_selection_qubo(
        forecast.expected_returns,
        forecast.covariance,
        cardinality=k,
        risk_aversion=3.0,
        penalty=4.0,
    )
    assert Q.shape == (len(prices.columns), len(prices.columns))
    assert np.allclose(Q, Q.T)

    bits = exact_select(
        forecast.expected_returns,
        forecast.covariance,
        cardinality=k,
        risk_aversion=3.0,
    )
    assert bits.sum() == k

    qaoa = solve_qaoa(Q, cardinality=k, grid_gamma=5, grid_beta=5)
    assert qaoa.bitstring.sum() == k

    selected = [a for a, b in zip(prices.columns, qaoa.bitstring) if b == 1]
    solution = optimize_weights(
        selected,
        forecast.expected_returns,
        forecast.covariance,
        risk_aversion=3.0,
        max_weight=0.5,
    )
    assert abs(solution.weights.sum() - 1.0) < 1e-6
    assert (solution.weights >= 0).all()
