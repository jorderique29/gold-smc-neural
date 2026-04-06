# tests/test_alignment.py
import pandas as pd
import numpy as np
import pytest
from src.alignment import align_timeframes


def _make_tf(freq: str, n: int, base_price: float = 1900.0, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame at a given frequency."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq=freq, tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open":      close - 0.1,
        "high":      close + 1.0,
        "low":       close - 1.0,
        "close":     close,
        "volume":    rng.uniform(100, 1000, n),
        "trend":     rng.choice([-1, 0, 1], n),
        "vol_zscore": rng.normal(0, 1, n),
        "bull_ob":   rng.choice([True, False], n),
        "bear_ob":   rng.choice([True, False], n),
        "ob_midpoint": close + rng.uniform(-1, 1, n),
        "bos_bullish": rng.choice([True, False], n),
        "bos_bearish": rng.choice([True, False], n),
        "fvg_bull":  rng.choice([True, False], n),
        "fvg_bear":  rng.choice([True, False], n),
        "asia_high": close + rng.uniform(0, 2, n),
        "asia_low":  close - rng.uniform(0, 2, n),
    }, index=pd.DatetimeIndex(dates, name="timestamp"))


def test_align_returns_m5_length():
    """Output must have the same number of rows as M5 input."""
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)
    result = align_timeframes(m5, m15, h1)
    assert len(result) == len(m5)


def test_align_preserves_m5_index():
    """Output index must be identical to M5 index."""
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)
    result = align_timeframes(m5, m15, h1)
    pd.testing.assert_index_equal(result.index, m5.index)


def test_align_adds_h1_columns():
    """Result must contain h1_close and h1_trend columns."""
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)
    result = align_timeframes(m5, m15, h1)
    assert "h1_close"  in result.columns
    assert "h1_trend"  in result.columns


def test_align_adds_m15_columns():
    """Result must contain m15_close and m15_trend columns."""
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)
    result = align_timeframes(m5, m15, h1)
    assert "m15_close" in result.columns
    assert "m15_trend" in result.columns


def test_no_lookahead_h1_at_hour_boundary():
    """
    Critical look-ahead test:
    The H1 bar that OPENS at 00:00 UTC CLOSES at 01:00 UTC.
    M5 bars at 00:05, 00:10, ... 00:55 must NOT see that H1 bar's data.
    The M5 bar at 01:00 is the FIRST that may see the 00:00 H1 bar.

    We inject a sentinel value (close=9999.0) into the H1 bar at 00:00.
    Any M5 bar timestamped < 01:00 that shows h1_close=9999 has look-ahead bias.
    """
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)

    # Inject sentinel into H1 bar at 00:00 UTC
    sentinel_ts = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    h1.loc[sentinel_ts, "close"] = 9999.0

    result = align_timeframes(m5, m15, h1)

    # M5 bars BEFORE 01:00 must NOT have h1_close=9999
    before_1am = result[result.index < pd.Timestamp("2024-01-02 01:00:00", tz="UTC")]
    assert not (before_1am["h1_close"] == 9999.0).any(), (
        "LOOK-AHEAD BIAS: M5 bars before 01:00 UTC see H1 bar that closes at 01:00 UTC"
    )

    # M5 bar AT 01:00 must have h1_close=9999
    at_1am = result.loc[result.index == pd.Timestamp("2024-01-02 01:00:00", tz="UTC")]
    if len(at_1am) > 0:
        assert (at_1am["h1_close"] == 9999.0).all(), (
            "M5 bar at 01:00 UTC should see the H1 bar that closed at 01:00 UTC"
        )


def test_no_lookahead_m15_at_15min_boundary():
    """
    The M15 bar opening at 00:00 UTC closes at 00:15 UTC.
    M5 bars at 00:05 and 00:10 must NOT see it.
    M5 bar at 00:15 is the first that may see it.
    """
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)

    sentinel_ts = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")
    m15.loc[sentinel_ts, "close"] = 8888.0

    result = align_timeframes(m5, m15, h1)

    before_15 = result[result.index < pd.Timestamp("2024-01-02 00:15:00", tz="UTC")]
    assert not (before_15["m15_close"] == 8888.0).any(), (
        "LOOK-AHEAD BIAS: M5 bars before 00:15 see M15 bar that closes at 00:15"
    )

    at_15 = result.loc[result.index == pd.Timestamp("2024-01-02 00:15:00", tz="UTC")]
    if len(at_15) > 0:
        assert (at_15["m15_close"] == 8888.0).all(), (
            "M5 bar at 00:15 should see M15 bar that closed at 00:15"
        )


def test_align_no_mutation():
    """align_timeframes must not modify any input DataFrame."""
    m5  = _make_tf("5min",  288)
    m15 = _make_tf("15min",  96)
    h1  = _make_tf("1h",     24)
    m5_orig  = m5.copy()
    m15_orig = m15.copy()
    h1_orig  = h1.copy()
    align_timeframes(m5, m15, h1)
    pd.testing.assert_frame_equal(m5,  m5_orig)
    pd.testing.assert_frame_equal(m15, m15_orig)
    pd.testing.assert_frame_equal(h1,  h1_orig)


def test_align_no_extra_rows():
    """Result must not have more rows than M5 input (merge_asof never adds rows)."""
    m5  = _make_tf("5min",  500)
    m15 = _make_tf("15min", 167)
    h1  = _make_tf("1h",    21)
    result = align_timeframes(m5, m15, h1)
    assert len(result) == len(m5)
