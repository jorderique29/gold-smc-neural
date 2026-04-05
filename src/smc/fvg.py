# src/smc/fvg.py
"""
Fair Value Gap (FVG) / Imbalance detection.

A FVG is a 3-candle pattern where price moves so aggressively that the
wicks of candle[i-2] and candle[i] do not overlap — leaving an untraded
price zone (imbalance) that price tends to revisit.

Bullish FVG at bar i : candle[i].low  > candle[i-2].high
                       gap_size = candle[i].low - candle[i-2].high

Bearish FVG at bar i : candle[i].high < candle[i-2].low
                       gap_size = candle[i-2].low - candle[i].high
"""
import numpy as np
import pandas as pd

_REQUIRED = {"high", "low"}


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect bullish and bearish Fair Value Gaps.

    Args:
        df: OHLCV DataFrame with at minimum 'high' and 'low' columns.

    Returns:
        Copy of df with added columns:
            fvg_bull      (bool)  : True if bullish FVG at this bar
            fvg_bull_size (float) : size of the gap in price units (0 if none)
            fvg_bear      (bool)  : True if bearish FVG at this bar
            fvg_bear_size (float) : size of the gap in price units (0 if none)

    Raises:
        ValueError: if 'high' or 'low' columns are missing.
    """
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_fvg requires columns {_REQUIRED}. Missing: {missing}."
        )

    df = df.copy()

    high_2ago = df["high"].shift(2)
    low_2ago  = df["low"].shift(2)

    bull_gap = df["low"] - high_2ago          # positive → bullish FVG
    bear_gap = low_2ago  - df["high"]         # positive → bearish FVG

    df["fvg_bull"]      = (bull_gap > 0).astype(bool)
    df["fvg_bull_size"] = bull_gap.clip(lower=0).fillna(0.0)

    df["fvg_bear"]      = (bear_gap > 0).astype(bool)
    df["fvg_bear_size"] = bear_gap.clip(lower=0).fillna(0.0)

    return df
