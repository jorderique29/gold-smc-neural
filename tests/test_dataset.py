# tests/test_dataset.py
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from tests.conftest import make_ohlcv
from src.features.normalizer import normalize_features
from src.dataset import save_parquet, FEATURE_COLS


def _make_aligned_df(n: int = 200) -> pd.DataFrame:
    """Simulate an aligned DataFrame with all expected feature columns."""
    rng = np.random.default_rng(42)
    df = make_ohlcv(n)
    # Add SMC columns that would be present after full pipeline
    df["trend"]          = rng.choice([-1, 0, 1], n)
    df["ob_type"]        = rng.choice([-1, 0, 1], n)
    df["ob_midpoint"]    = df["close"] + rng.uniform(-1, 1, n)
    df["ob_top"]         = df["close"] + rng.uniform(0, 2, n)
    df["ob_bottom"]      = df["close"] - rng.uniform(0, 2, n)
    df["bull_ob"]        = rng.choice([True, False], n)
    df["bear_ob"]        = rng.choice([True, False], n)
    df["bos_bullish"]    = rng.choice([True, False], n)
    df["bos_bearish"]    = rng.choice([True, False], n)
    df["choch_bullish"]  = rng.choice([True, False], n)
    df["choch_bearish"]  = rng.choice([True, False], n)
    df["fvg_bull"]       = rng.choice([True, False], n)
    df["fvg_bear"]       = rng.choice([True, False], n)
    df["fvg_bull_size"]  = rng.uniform(0, 2, n)
    df["fvg_bear_size"]  = rng.uniform(0, 2, n)
    df["vol_zscore"]     = rng.normal(0, 1, n)
    df["rsi_lagless"]    = rng.uniform(20, 80, n)
    df["atr"]            = rng.uniform(1, 5, n)
    df["ob_freshness"]   = rng.uniform(0, 50, n)
    df["equal_high_pool"] = rng.choice([True, False], n)
    df["equal_low_pool"]  = rng.choice([True, False], n)
    df["in_asia_session"] = rng.choice([True, False], n)
    df["judas_swing_bull"] = rng.choice([True, False], n)
    df["judas_swing_bear"] = rng.choice([True, False], n)
    df["asia_high"]      = df["close"] + rng.uniform(1, 3, n)
    df["asia_low"]       = df["close"] - rng.uniform(1, 3, n)
    # M15 and H1 prefixed columns
    for prefix in ["m15", "h1"]:
        df[f"{prefix}_close"]       = df["close"] + rng.normal(0, 0.5, n)
        df[f"{prefix}_trend"]       = rng.choice([-1, 0, 1], n)
        df[f"{prefix}_ob_midpoint"] = df["close"] + rng.uniform(-1, 1, n)
        df[f"{prefix}_bull_ob"]     = rng.choice([True, False], n)
        df[f"{prefix}_bear_ob"]     = rng.choice([True, False], n)
        df[f"{prefix}_bos_bullish"] = rng.choice([True, False], n)
        df[f"{prefix}_bos_bearish"] = rng.choice([True, False], n)
    df["h1_asia_high"]   = df["close"] + rng.uniform(1, 3, n)
    df["h1_asia_low"]    = df["close"] - rng.uniform(1, 3, n)
    df["m15_fvg_bull"]   = rng.choice([True, False], n)
    df["m15_fvg_bear"]   = rng.choice([True, False], n)
    # Risk columns
    df["entry"]          = df["close"] + rng.uniform(-0.5, 0.5, n)
    df["sl"]             = df["close"] - rng.uniform(0.5, 2, n)
    df["tp"]             = df["close"] + rng.uniform(1, 5, n)
    df["rr_achieved"]    = rng.uniform(1, 4, n)
    df["dd_filter_ok"]   = rng.choice([True, False], n)
    df["risk_price"]     = rng.uniform(0.1, 2, n)
    return df


# ── normalizer tests ──────────────────────────────────────────────────────────

def test_normalize_returns_dataframe():
    df = _make_aligned_df()
    result, scalers = normalize_features(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(df)


def test_normalize_returns_scalers_dict():
    df = _make_aligned_df()
    _, scalers = normalize_features(df)
    assert isinstance(scalers, dict)
    assert "price" in scalers
    assert "indicators" in scalers


def test_normalize_does_not_mutate_input():
    df = _make_aligned_df()
    original = df.copy()
    normalize_features(df)
    pd.testing.assert_frame_equal(df[original.columns], original)


def test_normalize_price_columns_scaled():
    """After normalization, price columns should have ~0 median (RobustScaler)."""
    df = _make_aligned_df(500)
    result, _ = normalize_features(df)
    # RobustScaler centers on median → result median should be near 0
    if "close" in result.columns:
        assert abs(result["close"].median()) < 0.5, \
            "close should be near-zero median after RobustScaler"


def test_normalize_preserves_passthrough_columns():
    """Binary/categorical columns (trend, ob_type, etc.) must not be scaled."""
    df = _make_aligned_df()
    result, _ = normalize_features(df)
    if "trend" in result.columns:
        assert set(result["trend"].unique()).issubset({-1, 0, 1}), \
            "trend column must remain unscaled"
    if "ob_type" in result.columns:
        assert set(result["ob_type"].unique()).issubset({-1, 0, 1}), \
            "ob_type column must remain unscaled"


# ── dataset tests ─────────────────────────────────────────────────────────────

def test_save_parquet_creates_file(tmp_path):
    df = _make_aligned_df()
    result, _ = normalize_features(df)
    out = save_parquet(result, tmp_path / "test.parquet")
    assert out.exists()
    assert out.suffix == ".parquet"


def test_save_parquet_readable(tmp_path):
    df = _make_aligned_df()
    result, _ = normalize_features(df)
    out = save_parquet(result, tmp_path / "test.parquet")
    loaded = pd.read_parquet(out)
    assert len(loaded) == len(result)


def test_save_parquet_only_feature_cols(tmp_path):
    """Saved Parquet must only contain columns from FEATURE_COLS (that exist in df)."""
    df = _make_aligned_df()
    result, _ = normalize_features(df)
    out = save_parquet(result, tmp_path / "test.parquet")
    loaded = pd.read_parquet(out)
    for col in loaded.columns:
        assert col in FEATURE_COLS, f"Unexpected column in Parquet: {col}"


def test_feature_cols_list_nonempty():
    assert len(FEATURE_COLS) > 20, "FEATURE_COLS should contain at least 20 feature columns"
