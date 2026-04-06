# tests/test_liquidity.py
import pandas as pd
import numpy as np
import pytest
from src.smc.liquidity import detect_equal_highs_lows, detect_asia_session


def _make_equal_high_df():
    """10 bars: bars 0 and 4 have nearly identical highs (105.0 vs 105.02)."""
    dates = pd.date_range("2024-01-02 10:00", periods=10, freq="5min", tz="UTC")
    data = {
        "open":   [100.0] * 10,
        "high":   [105.0, 104.9, 102.0, 103.0, 105.02, 104.0, 103.0, 102.0, 101.0, 100.0],
        "low":    [99.0]  * 10,
        "close":  [101.0] * 10,
        "volume": [500.0] * 10,
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="timestamp"))


def _make_full_day_df(seed: int = 0) -> pd.DataFrame:
    """288 bars (full 24h of 5-min bars) for 2024-01-02 UTC."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=288, freq="5min", tz="UTC")
    close = 1900.0 + np.cumsum(rng.normal(0, 0.3, 288))
    return pd.DataFrame({
        "open":   close - rng.uniform(0.1, 0.5, 288),
        "high":   close + rng.uniform(0.5, 2.0, 288),
        "low":    close - rng.uniform(0.5, 2.0, 288),
        "close":  close,
        "volume": rng.integers(100, 1000, 288).astype(float),
    }, index=pd.DatetimeIndex(dates, name="timestamp"))


# ── Equal Highs / Lows tests ──────────────────────────────────────────────────

def test_equal_highs_columns_exist():
    df = detect_equal_highs_lows(_make_equal_high_df())
    assert "equal_high_pool" in df.columns
    assert "equal_low_pool" in df.columns


def test_equal_high_detected_within_tolerance():
    """Bars 0 and 4 have highs 105.0 and 105.02 → within 5-pip (0.50 price) tolerance."""
    df = detect_equal_highs_lows(_make_equal_high_df(), tolerance_pips=5.0)
    assert df["equal_high_pool"].iloc[4], "Bar 4 (105.02) should match bar 0 (105.0) at 5-pip tol"


def test_equal_high_not_detected_outside_tolerance():
    """With 0.1-pip tolerance (0.01 price), 105.02 vs 105.0 differ by 0.02 → NOT equal."""
    df = detect_equal_highs_lows(_make_equal_high_df(), tolerance_pips=0.1)
    assert not df["equal_high_pool"].iloc[4], "Should NOT match at tight 0.1-pip tolerance"


def test_equal_pool_bool_dtype():
    df = detect_equal_highs_lows(_make_equal_high_df())
    assert df["equal_high_pool"].dtype == bool
    assert df["equal_low_pool"].dtype == bool


def test_equal_hl_no_mutation():
    df = _make_equal_high_df()
    original = df.copy()
    detect_equal_highs_lows(df)
    pd.testing.assert_frame_equal(df, original)


def test_equal_hl_requires_columns():
    with pytest.raises(ValueError, match="high|low"):
        detect_equal_highs_lows(pd.DataFrame({"x": [1, 2, 3]}))


# ── Asia Session tests ────────────────────────────────────────────────────────

def test_asia_session_columns_exist():
    df = detect_asia_session(_make_full_day_df())
    for col in ["asia_high", "asia_low", "in_asia_session",
                "judas_swing_bull", "judas_swing_bear"]:
        assert col in df.columns, f"Missing: {col}"


def test_in_asia_session_correct_hours():
    df = detect_asia_session(_make_full_day_df())
    assert df.between_time("00:00", "07:55")["in_asia_session"].all()
    assert not df.between_time("08:00", "23:55")["in_asia_session"].any()


def test_asia_high_forward_filled():
    df = detect_asia_session(_make_full_day_df())
    assert df.between_time("08:00", "23:55")["asia_high"].notna().all()
    assert df.between_time("08:00", "23:55")["asia_low"].notna().all()


def test_asia_high_equals_session_max():
    df = detect_asia_session(_make_full_day_df())
    expected = df.between_time("00:00", "07:55")["high"].max()
    assert df.iloc[150]["asia_high"] == pytest.approx(expected)


def test_judas_swing_bear_conditions():
    df = detect_asia_session(_make_full_day_df())
    bear = df[df["judas_swing_bear"]]
    if len(bear) > 0:
        assert (bear["high"] > bear["asia_high"]).all()
        assert (bear["close"] < bear["asia_high"]).all()
        assert not bear["in_asia_session"].any()


def test_judas_swing_bull_conditions():
    df = detect_asia_session(_make_full_day_df())
    bull = df[df["judas_swing_bull"]]
    if len(bull) > 0:
        assert (bull["low"] < bull["asia_low"]).all()
        assert (bull["close"] > bull["asia_low"]).all()
        assert not bull["in_asia_session"].any()


def test_asia_session_no_mutation():
    df = _make_full_day_df()
    original = df.copy()
    detect_asia_session(df)
    pd.testing.assert_frame_equal(df, original)


def test_asia_session_requires_columns():
    with pytest.raises(ValueError, match="high|low|close"):
        detect_asia_session(pd.DataFrame({"x": [1, 2, 3]}))
