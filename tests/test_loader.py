# tests/test_loader.py
import pytest
from src.loader import load_mt5_csv

def test_load_m5_returns_typed_dataframe():
    df = load_mt5_csv("xauusd_m5_history.csv")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert df.index.dtype == "datetime64[ns, UTC]"
    assert df.dtypes["close"] == float
    assert len(df) > 300_000

def test_load_m5_no_nulls():
    df = load_mt5_csv("xauusd_m5_history.csv")
    assert df.isnull().sum().sum() == 0

def test_load_m5_monotonic_index():
    df = load_mt5_csv("xauusd_m5_history.csv")
    assert df.index.is_monotonic_increasing
