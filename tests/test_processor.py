# tests/test_processor.py
import pytest
from pathlib import Path
from src.processor import GoldQuantProcessor


@pytest.fixture(scope="module")
def processor():
    """Shared processor instance — loaded once for all tests in this module."""
    return GoldQuantProcessor(
        m5_file="xauusd_m5_history.csv",
        m15_file="xauusd_m15_history.csv",
        h1_file="xauusd_h1_history.csv",
    )


# ── detect_smc ────────────────────────────────────────────────────────────────

def test_detect_smc_runs(processor):
    """detect_smc() must not raise and must populate self.m5, self.m15, self.h1."""
    processor.detect_smc()
    assert processor.m5  is not None
    assert processor.m15 is not None
    assert processor.h1  is not None


def test_detect_smc_m5_has_ob_columns(processor):
    """M5 must contain bull_ob, bear_ob, trend after detect_smc()."""
    processor.detect_smc()
    for col in ["bull_ob", "bear_ob", "trend", "bos_bullish", "fvg_bull",
                "rsi_lagless", "atr", "ob_freshness"]:
        assert col in processor.m5.columns, f"M5 missing: {col}"


def test_detect_smc_h1_has_prev_day_levels(processor):
    """H1 must contain prev_day_high and prev_day_low after detect_smc()."""
    processor.detect_smc()
    assert "prev_day_high" in processor.h1.columns
    assert "prev_day_low"  in processor.h1.columns


def test_detect_smc_preserves_utc_index(processor):
    """All three DataFrames must have UTC-aware DatetimeIndex."""
    processor.detect_smc()
    import pandas as pd
    assert processor.m5.index.tz  is not None
    assert processor.m15.index.tz is not None
    assert processor.h1.index.tz  is not None


# ── align_timeframes ──────────────────────────────────────────────────────────

def test_align_timeframes_runs(processor):
    """align_timeframes() must not raise and must populate self.aligned."""
    processor.detect_smc()
    processor.align_timeframes()
    assert processor.aligned is not None


def test_aligned_has_h1_columns(processor):
    """Aligned DataFrame must contain h1_trend and h1_close."""
    processor.detect_smc()
    processor.align_timeframes()
    assert "h1_trend" in processor.aligned.columns
    assert "h1_close" in processor.aligned.columns


def test_aligned_has_m15_columns(processor):
    """Aligned DataFrame must contain m15_trend and m15_close."""
    processor.detect_smc()
    processor.align_timeframes()
    assert "m15_trend" in processor.aligned.columns
    assert "m15_close" in processor.aligned.columns


def test_aligned_length_equals_m5(processor):
    """Aligned DataFrame must have same row count as M5."""
    processor.detect_smc()
    processor.align_timeframes()
    assert len(processor.aligned) == len(processor.m5)


def test_align_requires_detect_smc_first(processor):
    """align_timeframes() without detect_smc() first should raise RuntimeError."""
    fresh = GoldQuantProcessor(
        m5_file="xauusd_m5_history.csv",
        m15_file="xauusd_m15_history.csv",
        h1_file="xauusd_h1_history.csv",
    )
    with pytest.raises(RuntimeError, match="detect_smc"):
        fresh.align_timeframes()


# ── generate_training_tensors ─────────────────────────────────────────────────

def test_generate_training_tensors_creates_parquet(processor, tmp_path):
    """generate_training_tensors() must write at least one .parquet file."""
    processor.detect_smc()
    processor.align_timeframes()
    out = processor.generate_training_tensors(output_dir=tmp_path)
    assert out.exists(), f"Parquet file not found at {out}"
    assert out.suffix == ".parquet"


def test_generate_requires_align_first():
    """generate_training_tensors() without align_timeframes() should raise RuntimeError."""
    fresh = GoldQuantProcessor(
        m5_file="xauusd_m5_history.csv",
        m15_file="xauusd_m15_history.csv",
        h1_file="xauusd_h1_history.csv",
    )
    fresh.detect_smc()
    with pytest.raises(RuntimeError, match="align_timeframes"):
        fresh.generate_training_tensors()


def test_parquet_readable(processor, tmp_path):
    """The written Parquet file must be readable with pandas."""
    import pandas as pd
    processor.detect_smc()
    processor.align_timeframes()
    out = processor.generate_training_tensors(output_dir=tmp_path)
    df = pd.read_parquet(out)
    assert len(df) > 0
