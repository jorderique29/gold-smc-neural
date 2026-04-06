# src/features/technical.py
"""
Lag-reduced technical indicators for XAUUSD Transformer feature input.

RSI-Lagless : Ehlers' 2-pole Super Smoother pre-filter applied to closing
              price before standard RSI computation. Reduces phase lag vs
              standard EMA-based RSI while keeping output in [0, 100].
              Reference: Ehlers (2013) Cybernetic Analysis, ch. 3.

ATR         : 14-bar Average True Range with Wilder smoothing (EMA alpha=1/period).
"""
import math
import numpy as np
import pandas as pd


def add_rsi_lagless(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add an Ehlers Super Smoother RSI column ('rsi_lagless') to df.

    The Super Smoother (2-pole IIR filter) pre-smooths closing prices to
    eliminate high-frequency noise before RSI computation, cutting the lag
    inherent in standard EMA-based RSI.

    Args:
        df:     OHLCV DataFrame with a 'close' column.
        period: Lookback period (default 14). Controls both the Super Smoother
                cutoff frequency and the Wilder EMA window.

    Returns:
        Copy of df with added column 'rsi_lagless' (float, NaN for first `period` bars).

    Raises:
        ValueError: if 'close' column is missing.
    """
    if "close" not in df.columns:
        raise ValueError("add_rsi_lagless requires a 'close' column.")

    df = df.copy()
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    # ── Super Smoother coefficients ──────────────────────────────────────────
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    # ── Apply 2-pole Super Smoother recursively ──────────────────────────────
    ss = np.zeros(n, dtype=float)
    for i in range(2, n):
        ss[i] = (c1 * (close[i] + close[i - 1]) / 2.0
                 + c2 * ss[i - 1]
                 + c3 * ss[i - 2])

    # ── RSI on smoothed series ───────────────────────────────────────────────
    delta = np.diff(ss, prepend=ss[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain = (pd.Series(gain)
                .ewm(alpha=1.0 / period, min_periods=period, adjust=False)
                .mean()
                .to_numpy())
    avg_loss = (pd.Series(loss)
                .ewm(alpha=1.0 / period, min_periods=period, adjust=False)
                .mean()
                .to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        rs  = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan

    df["rsi_lagless"] = rsi
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add a 14-bar Average True Range column ('atr') to df.

    Uses Wilder smoothing (EMA with alpha=1/period).

    Args:
        df:     OHLCV DataFrame with 'high', 'low', 'close'.
        period: ATR period (default 14).

    Returns:
        Copy of df with added column 'atr' (float, NaN for first `period` bars).

    Raises:
        ValueError: if any of 'high', 'low', 'close' are missing.
    """
    missing = {"high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(
            f"add_atr requires 'high', 'low', 'close'. Missing: {missing}"
        )

    df = df.copy()
    high       = df["high"]
    low        = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    atr.iloc[:period] = np.nan
    df["atr"] = atr
    return df
