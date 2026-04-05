# src/smc/structure.py
import numpy as np
import pandas as pd


def detect_fractals(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Detect swing highs and lows using an N-bar fractal rule.

    A swing high at bar i: df.high[i] is the maximum over the rolling window
    centered at i with width (2*n + 1). Edges within n bars of start/end
    are forced False (insufficient confirmation bars).

    A swing low at bar i: df.low[i] is the minimum over the same window.

    Args:
        df: OHLCV DataFrame with 'high' and 'low' columns
        n:  half-window size (bars on each side required for confirmation)

    Returns:
        df copy with added bool columns: 'swing_high', 'swing_low'
    """
    df = df.copy()
    n_bars = 2 * n + 1

    rolling_max = df["high"].rolling(n_bars, center=True).max()
    rolling_min = df["low"].rolling(n_bars, center=True).min()

    df["swing_high"] = (df["high"] == rolling_max) & df["high"].notna()
    df["swing_low"]  = (df["low"]  == rolling_min) & df["low"].notna()

    # Mask edges — rolling(center=True) needs n bars on each side
    df.loc[df.index[:n],  "swing_high"] = False
    df.loc[df.index[-n:], "swing_high"] = False
    df.loc[df.index[:n],  "swing_low"]  = False
    df.loc[df.index[-n:], "swing_low"]  = False

    # Ensure bool dtype
    df["swing_high"] = df["swing_high"].astype(bool)
    df["swing_low"]  = df["swing_low"].astype(bool)

    return df


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    Requires detect_fractals() columns: swing_high, swing_low.

    BOS Bullish : close crosses above the last confirmed swing high price
    BOS Bearish : close crosses below the last confirmed swing low price
    CHoCH Bullish: BOS bullish that contradicts prevailing bearish trend (-1)
    CHoCH Bearish: BOS bearish that contradicts prevailing bullish trend (+1)

    Trend state:
        +1  bullish  (last BOS was bullish)
        -1  bearish  (last BOS was bearish)
         0  undefined (no BOS yet)

    Returns:
        df copy with added columns:
            bos_bullish, bos_bearish  (bool)
            choch_bullish, choch_bearish (bool)
            trend (int: -1 / 0 / +1)
    """
    df = df.copy()

    # Price level of the last confirmed swing high / low
    last_sh_price = df["high"].where(df["swing_high"]).ffill()
    last_sl_price = df["low"].where(df["swing_low"]).ffill()

    # BOS fires when close breaks the PREVIOUS bar's swing level
    df["bos_bullish"] = df["close"] > last_sh_price.shift(1)
    df["bos_bearish"] = df["close"] < last_sl_price.shift(1)

    # Build trend state from BOS signals
    raw_trend = np.where(
        df["bos_bullish"], 1,
        np.where(df["bos_bearish"], -1, np.nan)
    )
    trend_series = (
        pd.Series(raw_trend, index=df.index, dtype=float)
        .ffill()
        .fillna(0)
        .astype(int)
    )
    df["trend"] = trend_series

    # CHoCH = BOS that contradicts the prior bar's trend
    prior_trend = df["trend"].shift(1).fillna(0).astype(int)
    df["choch_bullish"] = df["bos_bullish"] & (prior_trend == -1)
    df["choch_bearish"] = df["bos_bearish"] & (prior_trend == 1)

    return df
