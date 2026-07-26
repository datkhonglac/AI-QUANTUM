
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


TRADING_DAYS = 252


@dataclass
class ForecastResult:
    expected_returns: pd.Series
    model_scores: pd.DataFrame
    covariance: pd.DataFrame
    returns: pd.DataFrame


def load_price_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Đọc CSV dạng wide: Date, TICKER_1, TICKER_2, ..."""
    df = pd.read_csv(source)
    if "Date" not in df.columns:
        raise ValueError("CSV phải có cột Date.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Cột Date có giá trị không hợp lệ.")

    price_cols = [c for c in df.columns if c != "Date"]
    if len(price_cols) < 4:
        raise ValueError("Cần tối thiểu 4 mã tài sản để chạy demo.")

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .dropna(how="all")
    )

    df = df.dropna(axis=1, thresh=max(30, int(len(df) * 0.8)))
    df = df.ffill().dropna()
    if len(df) < 80:
        raise ValueError("Cần tối thiểu khoảng 80 phiên dữ liệu sau làm sạch.")
    if (df <= 0).any().any():
        raise ValueError("Giá phải lớn hơn 0.")
    return df


def _feature_frame(series: pd.Series) -> pd.DataFrame:
    ret = series.pct_change()
    feat = pd.DataFrame(index=series.index)
    feat["lag_1"] = ret.shift(1)
    feat["lag_2"] = ret.shift(2)
    feat["lag_5"] = ret.shift(5)
    feat["momentum_5"] = series.pct_change(5).shift(1)
    feat["momentum_10"] = series.pct_change(10).shift(1)
    feat["mean_5"] = ret.rolling(5).mean().shift(1)
    feat["vol_5"] = ret.rolling(5).std().shift(1)
    feat["vol_20"] = ret.rolling(20).std().shift(1)
    feat["target"] = ret
    return feat.dropna()


def forecast_returns(prices: pd.DataFrame, lookback: int = 252) -> ForecastResult:
    """Dự báo lợi nhuận ngày kế tiếp bằng Gradient Boosting cho từng tài sản."""
    daily_returns = prices.pct_change().dropna()
    window_returns = daily_returns.tail(min(lookback, len(daily_returns)))
    annual_cov = window_returns.cov() * TRADING_DAYS

    predictions: dict[str, float] = {}
    score_rows: list[dict[str, float | str]] = []

    for ticker in prices.columns:
        feat = _feature_frame(prices[ticker])
        if len(feat) < 50:
            pred_daily = float(window_returns[ticker].mean())
            mae = np.nan
            direction_acc = np.nan
        else:
            X = feat.drop(columns=["target"])
            y = feat["target"]
            split = max(35, int(len(X) * 0.8))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = GradientBoostingRegressor(
                random_state=42,
                n_estimators=80,
                max_depth=2,
                learning_rate=0.04,
                loss="huber",
            )
            model.fit(X_train, y_train)

            test_pred = model.predict(X_test) if len(X_test) else np.array([])
            mae = float(np.mean(np.abs(test_pred - y_test))) if len(test_pred) else np.nan
            direction_acc = (
                float(np.mean(np.sign(test_pred) == np.sign(y_test)))
                if len(test_pred)
                else np.nan
            )
            pred_daily = float(model.predict(X.iloc[[-1]])[0])

        hist_mean = float(window_returns[ticker].mean())
        shrunk_daily = 0.35 * pred_daily + 0.65 * hist_mean
        predictions[ticker] = shrunk_daily * TRADING_DAYS
        score_rows.append(
            {
                "Ticker": ticker,
                "MAE_daily": mae,
                "Directional_Accuracy": direction_acc,
                "Predicted_Annual_Return": predictions[ticker],
            }
        )

    expected = pd.Series(predictions, name="ExpectedReturn").reindex(prices.columns)
    scores = pd.DataFrame(score_rows).set_index("Ticker")
    return ForecastResult(
        expected_returns=expected,
        model_scores=scores,
        covariance=annual_cov.reindex(index=prices.columns, columns=prices.columns),
        returns=daily_returns,
    )
