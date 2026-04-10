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
        assert pos > 0, "CHoCH cannot fire at bar 0"
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
        assert pos > 0, "CHoCH cannot fire at bar 0"
        assert df["trend"].iloc[pos - 1] == 1, (
                f"CHoCH bearish at position {pos} requires prior trend=+1, "
                f"got {df['trend'].iloc[pos-1]}"
            )


def test_bos_bullish_dominates_in_uptrend():
    """In a sustained uptrend, bos_bullish should fire more than bos_bearish."""
    from tests.conftest import make_ohlcv
    import numpy as np
    # Build a clean uptrend: each bar's close is strictly higher than the last
    rng = np.random.default_rng(10)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1900.0 + np.cumsum(rng.uniform(0.1, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.2, 0.8, n)
    open_ = np.roll(close, 1); open_[0] = close[0] - 0.1
    volume = rng.integers(300, 1500, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    n_bull = df["bos_bullish"].sum()
    n_bear = df["bos_bearish"].sum()
    assert n_bull > n_bear, (
        f"In an uptrend, expected more bullish BOS ({n_bull}) than bearish ({n_bear})"
    )


def test_detect_fractals_rejects_n_zero():
    import pytest
    df = make_ohlcv(20)
    with pytest.raises(ValueError, match="n must be >= 1"):
        detect_fractals(df, n=0)


def test_bos_choch_requires_fractal_columns():
    import pytest
    df = make_ohlcv(20)  # raw df, no fractal columns
    with pytest.raises(ValueError, match="detect_fractals"):
        detect_bos_choch(df)
