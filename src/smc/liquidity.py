# src/smc/liquidity.py
"""
Liquidity pool and session range detection for XAUUSD SMC analysis.

Detects:
  - Equal Highs / Equal Lows  (stop-loss cluster zones / liquidity pools)
  - Asia session range         (00:00–08:00 UTC), forward-filled to all bars
  - Judas Swings               (sweep of Asia range with intra-bar reversal)
"""
import numpy as np
import pandas as pd

PIP_SIZE = 0.10   # 1 pip = $0.10 for XAUUSD


def detect_equal_highs_lows(
    df: pd.DataFrame,
    tolerance_pips: float = 5.0,
    lookback: int = 50,
) -> pd.DataFrame:
    """
    Mark bars whose high or low matches a prior bar's high/low within tolerance.

    Args:
        df:              OHLCV DataFrame with 'high' and 'low'.
        tolerance_pips:  Max pip distance for two levels to be considered equal.
        lookback:        Number of prior bars to scan.

    Returns:
        Copy of df with added bool columns:
            equal_high_pool, equal_low_pool

    Raises:
        ValueError: if 'high' or 'low' columns are missing.
    """
    missing = {"high", "low"} - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_equal_highs_lows requires 'high' and 'low'. Missing: {missing}"
        )

    df = df.copy()
    tolerance = tolerance_pips * PIP_SIZE
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    n     = len(df)

    eq_high = np.zeros(n, dtype=bool)
    eq_low  = np.zeros(n, dtype=bool)

    for i in range(1, n):
        start = max(0, i - lookback)
        window = slice(start, i)
        if np.any(np.abs(highs[window] - highs[i]) <= tolerance):
            eq_high[i] = True
        if np.any(np.abs(lows[window] - lows[i]) <= tolerance):
            eq_low[i] = True

    df["equal_high_pool"] = eq_high
    df["equal_low_pool"]  = eq_low
    return df


def detect_asia_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Asia session range (00:00–08:00 UTC) and detect Judas Swings.

    The Asia range (high/low) is forward-filled so every bar in London/NY
    carries that day's Asia reference levels.

    Judas Swing Bear: outside Asia, high > asia_high AND close < asia_high
    Judas Swing Bull: outside Asia, low  < asia_low  AND close > asia_low

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex and 'high', 'low', 'close'.

    Returns:
        Copy of df with added columns:
            in_asia_session  (bool)
            asia_high        (float, forward-filled)
            asia_low         (float, forward-filled)
            judas_swing_bear (bool)
            judas_swing_bull (bool)

    Raises:
        ValueError: if required columns are missing.
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_asia_session requires {required}. Missing: {missing}"
        )

    df = df.copy()

    # Asia session flag: 00:00 UTC (inclusive) to 08:00 UTC (exclusive)
    df["in_asia_session"] = (df.index.hour >= 0) & (df.index.hour < 8)

    # Daily Asia H/L: compute per calendar day, then join back
    date_key  = df.index.normalize()
    df["_date"] = date_key

    asia_ranges = (
        df[df["in_asia_session"]]
        .groupby("_date")
        .agg(asia_high=("high", "max"), asia_low=("low", "min"))
    )

    df = df.join(asia_ranges, on="_date", how="left")
    df["asia_high"] = df["asia_high"].ffill()
    df["asia_low"]  = df["asia_low"].ffill()
    df = df.drop(columns=["_date"])

    # Judas Swings — only outside Asia session
    outside = ~df["in_asia_session"]
    df["judas_swing_bear"] = (
        outside &
        (df["high"]  > df["asia_high"]) &
        (df["close"] < df["asia_high"])
    ).astype(bool)

    df["judas_swing_bull"] = (
        outside &
        (df["low"]   < df["asia_low"]) &
        (df["close"] > df["asia_low"])
    ).astype(bool)

    return df
