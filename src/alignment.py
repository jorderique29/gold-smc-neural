# src/alignment.py
"""
Look-ahead-safe multi-timeframe alignment for XAUUSD SMC pipeline.

Core invariant
--------------
A candlestick bar opening at time T and having frequency F closes at T + F.
That bar's data must only be visible to bars timestamped >= T + F.

Strategy
--------
1. Re-index each higher-TF DataFrame so every bar is stamped at its CLOSE time:
       H1  bar at 10:00  ->  re-indexed to 11:00
       M15 bar at 10:00  ->  re-indexed to 10:15
2. pd.merge_asof(direction="backward") then assigns to each M5 bar the most
   recent higher-TF bar whose close_time <= M5 timestamp.

This eliminates look-ahead bias completely.
"""
import pandas as pd


# Columns to carry forward from H1 into M5
_H1_COLS = [
    "close", "high", "low", "trend", "vol_zscore",
    "bull_ob", "bear_ob", "ob_midpoint",
    "bos_bullish", "bos_bearish",
    "asia_high", "asia_low",
]

# Columns to carry forward from M15 into M5
_M15_COLS = [
    "close", "high", "low", "trend", "vol_zscore",
    "bull_ob", "bear_ob", "ob_midpoint",
    "bos_bullish", "bos_bearish",
    "fvg_bull", "fvg_bear",
]


def _shift_to_close_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Return a copy of df with its DatetimeIndex shifted forward by one period,
    so each bar is stamped at its CLOSE time rather than its OPEN time.

    Args:
        df:   DataFrame with a tz-aware DatetimeIndex.
        freq: Frequency string matching the bar interval (e.g. '1h', '15min').
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    shifted = df.copy()
    shifted.index = shifted.index + offset
    return shifted


def _select_and_prefix(df: pd.DataFrame, cols: list, prefix: str) -> pd.DataFrame:
    """Keep only columns present in df from `cols`, renaming each with `prefix_`."""
    available = [c for c in cols if c in df.columns]
    return df[available].rename(columns={c: f"{prefix}_{c}" for c in available})


def align_timeframes(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge M15 and H1 SMC features into the M5 DataFrame without look-ahead bias.

    Each M5 row at timestamp T receives the most recent M15/H1 bar whose
    CLOSE TIME is <= T.

    Args:
        m5:  M5 DataFrame (base frame -- index is preserved exactly).
        m15: M15 DataFrame with SMC features.
        h1:  H1  DataFrame with SMC features.

    Returns:
        M5 DataFrame extended with columns prefixed 'h1_' and 'm15_'.
        Index is identical to m5.index. No rows are added or dropped.
    """
    # Stamp each higher-TF bar at its close time
    h1_closed  = _shift_to_close_time(h1,  "1h")
    m15_closed = _shift_to_close_time(m15, "15min")

    # Select and prefix columns
    h1_slim  = _select_and_prefix(h1_closed,  _H1_COLS,  "h1")
    m15_slim = _select_and_prefix(m15_closed, _M15_COLS, "m15")

    # merge_asof: for each M5 bar, find the last H1/M15 bar whose
    # close_time <= M5 timestamp (direction="backward" = look back only)
    result = pd.merge_asof(
        m5.sort_index(),
        h1_slim.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    result = pd.merge_asof(
        result,
        m15_slim.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return result
