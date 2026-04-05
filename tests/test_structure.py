# tests/test_structure.py
import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from src.smc.structure import detect_fractals, detect_bos_choch


def test_fractal_columns_exist():
    df = make_ohlcv(100)
    result = detect_fractals(df)
    assert "swing_high" in result.columns
    assert "swing_low" in result.columns
    assert result["swing_high"].dtype == bool
    assert result["swing_low"].dtype == bool


def test_fractal_no_swing_at_edges():
    """First and last N bars cannot be confirmed fractals (need N-bar lookahead)."""
    df = make_ohlcv(100)
    result = detect_fractals(df, n=2)
    assert not result["swing_high"].iloc[:2].any()
    assert not result["swing_high"].iloc[-2:].any()
    assert not result["swing_low"].iloc[:2].any()
    assert not result["swing_low"].iloc[-2:].any()


def test_fractal_swing_high_at_peak():
    """A bar with high strictly above all neighbors in [i-n..i+n] must be swing_high."""
    df = make_ohlcv(20, seed=0)
    # Inject a clear spike at bar 10
    df = df.copy()
    df.iloc[10, df.columns.get_loc("high")] = 9999.0
    result = detect_fractals(df, n=2)
    assert result["swing_high"].iloc[10], "Bar 10 with spike high should be swing_high"


def test_fractal_swing_low_at_trough():
    """A bar with low strictly below all neighbors must be swing_low."""
    df = make_ohlcv(20, seed=0)
    df = df.copy()
    df.iloc[10, df.columns.get_loc("low")] = 0.001
    result = detect_fractals(df, n=2)
    assert result["swing_low"].iloc[10], "Bar 10 with spike low should be swing_low"


def test_bos_columns_exist():
    df = make_ohlcv(150)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    for col in ["bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish", "trend"]:
        assert col in df.columns, f"Missing: {col}"


def test_trend_only_valid_values():
    df = make_ohlcv(200)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    assert df["trend"].isin([-1, 0, 1]).all(), "trend must be -1, 0, or 1"


def test_choch_bullish_requires_prior_bearish_trend():
    """CHoCH bullish must only fire when previous trend was -1 (bearish)."""
    df = make_ohlcv(200, seed=7)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    choch_bull_idx = df.index[df["choch_bullish"]]
    for idx in choch_bull_idx:
        pos = df.index.get_loc(idx)
        if pos > 0:
            assert df["trend"].iloc[pos - 1] == -1, (
                f"CHoCH bullish at position {pos} requires prior trend=-1, "
                f"got {df['trend'].iloc[pos-1]}"
            )


def test_choch_bearish_requires_prior_bullish_trend():
    """CHoCH bearish must only fire when previous trend was +1 (bullish)."""
    df = make_ohlcv(200, seed=7)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    choch_bear_idx = df.index[df["choch_bearish"]]
    for idx in choch_bear_idx:
        pos = df.index.get_loc(idx)
        if pos > 0:
            assert df["trend"].iloc[pos - 1] == 1, (
                f"CHoCH bearish at position {pos} requires prior trend=+1, "
                f"got {df['trend'].iloc[pos-1]}"
            )
