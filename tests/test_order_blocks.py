# tests/test_order_blocks.py
import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from src.smc.structure import detect_fractals, detect_bos_choch
from src.smc.fvg import detect_fvg
from src.smc.order_blocks import detect_order_blocks


def _pipeline(df):
    """Run the full pre-processing pipeline required before OB detection."""
    df = detect_fractals(df, n=2)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    return detect_order_blocks(df)


def test_ob_output_columns_exist():
    df = _pipeline(make_ohlcv(300))
    required = ["bull_ob", "bear_ob", "ob_top", "ob_bottom",
                "ob_midpoint", "ob_type", "vol_zscore"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_ob_type_only_valid_values():
    df = _pipeline(make_ohlcv(300))
    assert set(df["ob_type"].unique()).issubset({-1, 0, 1}), (
        f"ob_type must be in {{-1, 0, 1}}, got {df['ob_type'].unique()}"
    )


def test_ob_bool_dtypes():
    df = _pipeline(make_ohlcv(300))
    assert df["bull_ob"].dtype == bool
    assert df["bear_ob"].dtype == bool


def test_ob_midpoint_between_top_and_bottom():
    df = _pipeline(make_ohlcv(400, seed=5))
    ob_rows = df[df["bull_ob"] | df["bear_ob"]]
    if len(ob_rows) > 0:
        assert (ob_rows["ob_midpoint"] <= ob_rows["ob_top"] + 1e-9).all()
        assert (ob_rows["ob_midpoint"] >= ob_rows["ob_bottom"] - 1e-9).all()


def test_vol_zscore_nan_only_in_warmup():
    df = _pipeline(make_ohlcv(300))
    # After 50-bar warmup window, vol_zscore must be finite
    assert df["vol_zscore"].iloc[55:].notna().all()


def test_bull_ob_is_bearish_candle():
    """Every bull OB bar must have close < open (it's a bearish candle)."""
    df = _pipeline(make_ohlcv(500, seed=3))
    bull_ob_bars = df[df["bull_ob"]]
    if len(bull_ob_bars) > 0:
        assert (bull_ob_bars["close"] < bull_ob_bars["open"]).all(), (
            "All bull OB candles must be bearish (close < open)"
        )


def test_bear_ob_is_bullish_candle():
    """Every bear OB bar must have close > open (it's a bullish candle)."""
    df = _pipeline(make_ohlcv(500, seed=3))
    bear_ob_bars = df[df["bear_ob"]]
    if len(bear_ob_bars) > 0:
        assert (bear_ob_bars["close"] > bear_ob_bars["open"]).all(), (
            "All bear OB candles must be bullish (close > open)"
        )


def test_ob_top_equals_high():
    """OB top must equal the candle's high."""
    df = _pipeline(make_ohlcv(500, seed=7))
    ob_rows = df[df["bull_ob"] | df["bear_ob"]]
    if len(ob_rows) > 0:
        assert (ob_rows["ob_top"] == ob_rows["high"]).all()


def test_ob_bottom_equals_low():
    """OB bottom must equal the candle's low."""
    df = _pipeline(make_ohlcv(500, seed=7))
    ob_rows = df[df["bull_ob"] | df["bear_ob"]]
    if len(ob_rows) > 0:
        assert (ob_rows["ob_bottom"] == ob_rows["low"]).all()


def test_no_ob_without_full_pipeline():
    """detect_order_blocks should raise ValueError if required columns are missing."""
    raw = make_ohlcv(100)
    with pytest.raises(ValueError, match="bos_bullish|swing_high|fvg_bull"):
        detect_order_blocks(raw)


def test_ob_does_not_mutate_input():
    df = make_ohlcv(300)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    original_cols = set(df.columns)
    detect_order_blocks(df)
    assert set(df.columns) == original_cols, "detect_order_blocks must not mutate input"
