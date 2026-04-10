# tests/test_fvg.py
import pandas as pd
import numpy as np
import pytest
from src.smc.fvg import detect_fvg


def _make_bullish_fvg_df():
    """5-bar DataFrame with a clear bullish FVG at bar 2.
    bar 0: high=100.0  → bar 2's low must be > 100.0
    bar 2: low=102.0   → gap = 102.0 - 100.0 = 2.0
    """
    data = {
        "open":   [99.0,  100.0, 103.0, 104.0, 105.0],
        "high":   [100.0, 105.0, 106.0, 107.0, 108.0],
        "low":    [98.0,   99.0, 102.0, 103.0, 104.0],
        "close":  [99.5,  104.0, 105.5, 106.5, 107.5],
        "volume": [500.0,  800.0, 400.0, 300.0, 350.0],
    }
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def _make_bearish_fvg_df():
    """5-bar DataFrame with a clear bearish FVG at bar 2.
    bar 0: low=104.0   → bar 2's high must be < 104.0
    bar 2: high=102.0  → gap = 104.0 - 102.0 = 2.0
    """
    data = {
        "open":   [105.0, 104.0, 101.0, 100.0,  99.0],
        "high":   [106.0, 105.0, 102.0, 101.0, 100.0],
        "low":    [104.0, 100.0,  98.0,  97.0,  96.0],
        "close":  [104.5, 101.0,  99.0,  98.0,  97.0],
        "volume": [500.0, 800.0, 400.0, 300.0, 350.0],
    }
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def test_fvg_columns_exist():
    from tests.conftest import make_ohlcv
    result = detect_fvg(make_ohlcv(50))
    for col in ["fvg_bull", "fvg_bull_size", "fvg_bear", "fvg_bear_size"]:
        assert col in result.columns, f"Missing: {col}"


def test_bullish_fvg_detected_at_correct_bar():
    df = detect_fvg(_make_bullish_fvg_df())
    assert df["fvg_bull"].iloc[2], "Bar 2 should be bullish FVG"
    assert not df["fvg_bull"].iloc[0], "Bar 0 cannot be FVG (no i-2)"
    assert not df["fvg_bull"].iloc[1], "Bar 1 cannot be FVG (no i-2 high above bar 1 low)"


def test_bullish_fvg_size_correct():
    df = detect_fvg(_make_bullish_fvg_df())
    # gap = bar2.low - bar0.high = 102.0 - 100.0 = 2.0
    assert df["fvg_bull_size"].iloc[2] == pytest.approx(2.0)


def test_bearish_fvg_detected_at_correct_bar():
    df = detect_fvg(_make_bearish_fvg_df())
    assert df["fvg_bear"].iloc[2], "Bar 2 should be bearish FVG"
    assert not df["fvg_bear"].iloc[0]
    assert not df["fvg_bear"].iloc[1]


def test_bearish_fvg_size_correct():
    df = detect_fvg(_make_bearish_fvg_df())
    # gap = bar0.low - bar2.high = 104.0 - 102.0 = 2.0
    assert df["fvg_bear_size"].iloc[2] == pytest.approx(2.0)


def test_no_fvg_when_gap_closed():
    """If bar2.low <= bar0.high, no bullish FVG."""
    df = _make_bullish_fvg_df().copy()
    df.loc[df.index[2], "low"] = 99.0   # overlap: 99 < 100 = no gap
    result = detect_fvg(df)
    assert not result["fvg_bull"].iloc[2]
    assert result["fvg_bull_size"].iloc[2] == pytest.approx(0.0)


def test_fvg_sizes_non_negative():
    from tests.conftest import make_ohlcv
    df = detect_fvg(make_ohlcv(200))
    assert (df["fvg_bull_size"] >= 0).all()
    assert (df["fvg_bear_size"] >= 0).all()


def test_fvg_bool_dtype():
    from tests.conftest import make_ohlcv
    df = detect_fvg(make_ohlcv(100))
    assert df["fvg_bull"].dtype == bool
    assert df["fvg_bear"].dtype == bool


def test_fvg_does_not_mutate_input():
    from tests.conftest import make_ohlcv
    original = make_ohlcv(50)
    original_copy = original.copy()
    detect_fvg(original)
    pd.testing.assert_frame_equal(original, original_copy)


def test_fvg_requires_ohlcv_columns():
    import pytest
    bad_df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="high.*low|low.*high"):
        detect_fvg(bad_df)
