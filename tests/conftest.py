# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with realistic XAUUSD-like price action."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1900.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    open_ = close - rng.normal(0, 0.3, n)
    volume = rng.integers(100, 2000, n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )
    return df


@pytest.fixture
def synthetic_ohlcv():
    return make_ohlcv()


@pytest.fixture
def trending_up_ohlcv():
    """200-bar sustained uptrend."""
    rng = np.random.default_rng(10)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1900.0 + np.cumsum(rng.uniform(0.1, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.2, 0.8, n)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.1
    volume = rng.integers(300, 1500, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )
