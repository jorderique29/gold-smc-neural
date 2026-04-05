# tests/test_loader.py
import pytest
from src.loader import load_mt5_csv

@pytest.fixture(scope="module")
def m5_df():
    return load_mt5_csv("xauusd_m5_history.csv")

def test_load_m5_returns_typed_dataframe(m5_df):
    df = m5_df
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert df.index.dtype == "datetime64[ns, UTC]"
    assert df.dtypes["close"] == float
    assert len(df) > 300_000

def test_load_m5_no_nulls(m5_df):
    assert m5_df.isnull().sum().sum() == 0

def test_load_m5_monotonic_index(m5_df):
    assert m5_df.index.is_monotonic_increasing
