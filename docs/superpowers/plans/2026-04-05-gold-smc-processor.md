# Gold SMC Processor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `GoldQuantProcessor` — a vectorized Python pipeline that extracts high-probability SMC features from XAUUSD CSVs (M5/M15/H1) and outputs a look-ahead-bias-free Parquet dataset for Transformer training.

**Architecture:** Each timeframe is processed independently (structure → OBs → FVG → liquidity), then aligned to M5 timestamps using backward `merge_asof` so no future candle bleeds into a past row. Feature engineering and risk calculations are applied last, before serialization to Parquet + TFRecord.

**Tech Stack:** Python 3.10+, Pandas 2.x, NumPy, scikit-learn (scalers), TensorFlow 2.x (TFRecord), Pyarrow (Parquet), Matplotlib + mplfinance (validation charts).

---

## File Map

| File | Responsibility |
|---|---|
| `src/loader.py` | Parse MT5 TSV → typed DataFrames |
| `src/smc/structure.py` | Fractal swings, BOS, CHoCH, trend state |
| `src/smc/fvg.py` | Fair Value Gap detection + size |
| `src/smc/order_blocks.py` | 4-filter high-probability OB detection |
| `src/smc/liquidity.py` | Equal H/L pools, Asia session range, Judas swing flag |
| `src/features/technical.py` | RSI-Lagless, ATR |
| `src/features/normalizer.py` | RobustScaler fit/transform per feature group |
| `src/alignment.py` | look-ahead-safe M5←M15←H1 merge |
| `src/risk.py` | Entry price, SL, TP, DD filter flag |
| `src/processor.py` | `GoldQuantProcessor` orchestrator class |
| `src/dataset.py` | Parquet + TFRecord serialization |
| `models/transformer.py` | Transformer encoder architecture (schema) |
| `validation/visualizer.py` | 2-day candlestick chart with OB/BOS/FVG overlays |
| `tests/conftest.py` | Shared pytest fixtures (synthetic OHLCV) |
| `tests/test_structure.py` | Fractal/BOS/CHoCH unit tests |
| `tests/test_order_blocks.py` | OB 4-filter unit tests |
| `tests/test_fvg.py` | FVG detection unit tests |
| `tests/test_alignment.py` | Look-ahead bias regression tests |
| `tests/test_risk.py` | SL/TP/DD filter unit tests |
| `tests/test_processor.py` | End-to-end smoke test |
| `requirements.txt` | Pinned dependencies |

---

## Task 1: Project Setup & Data Loader

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/smc/__init__.py`, `src/features/__init__.py`
- Create: `src/loader.py`
- Create: `tests/conftest.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write requirements.txt**

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
pyarrow>=12.0.0
matplotlib>=3.7.0
mplfinance>=0.12.10b0
pytest>=7.4.0
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run to verify failure**

```bash
cd "C:/Users/Juan/GOLD SMC NEURAL"
python -m pytest tests/test_loader.py -v
```
Expected: `ImportError: No module named 'src'`

- [ ] **Step 4: Create `src/__init__.py`, `src/smc/__init__.py`, `src/features/__init__.py`**

All three are empty files.

- [ ] **Step 5: Implement `src/loader.py`**

```python
# src/loader.py
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROJECT_DIR = Path(__file__).parent.parent


def load_mt5_csv(filename: str, base_dir: Path | None = None) -> pd.DataFrame:
    """
    Parse MT5 tab-separated export into a timezone-aware DataFrame.

    Columns returned: open, high, low, close, volume
    Index: 'timestamp' (UTC, datetime64[ns, UTC])
    Volume: uses TICKVOL column (VOL is always 0 from MT5 export).
    """
    path = (base_dir or PROJECT_DIR) / filename
    if not path.exists():
        # fallback: check data/raw
        path = DATA_DIR / filename

    df = pd.read_csv(
        path,
        sep="\t",
        parse_dates={"timestamp": ["<DATE>", "<TIME>"]},
        date_format="%Y.%m.%d %H:%M:%S",
        index_col="timestamp",
    )

    df.index = df.index.tz_localize("UTC")
    df.index.name = "timestamp"

    df = df.rename(columns={
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
    })[["open", "high", "low", "close", "volume"]]

    df = df.astype(float)
    df = df.sort_index()
    return df
```

- [ ] **Step 6: Run tests to verify pass**

```bash
python -m pytest tests/test_loader.py -v
```
Expected: 3 PASSED

- [ ] **Step 7: Create `tests/conftest.py` with shared fixtures**

```python
# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with realistic XAUUSD-like price action."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 1900.0 + np.cumsum(rng.normal(0, 0.5, n))
    spread = rng.uniform(0.3, 1.5, n)
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
```

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt src/ tests/
git commit -m "feat: project scaffold + MT5 CSV loader"
```

---

## Task 2: Fractal Swing Structure

**Files:**
- Create: `src/smc/structure.py`
- Test: `tests/test_structure.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structure.py
import pandas as pd
import numpy as np
import pytest
from tests.conftest import make_ohlcv
from src.smc.structure import detect_fractals, detect_bos_choch


def test_fractal_swing_high_detected():
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


def test_bos_bullish_after_uptrend():
    """After a sustained uptrend, bos_bullish should fire."""
    from tests.conftest import make_ohlcv
    df = make_ohlcv(150, seed=10)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    assert "bos_bullish" in df.columns
    assert "bos_bearish" in df.columns
    assert "choch_bullish" in df.columns
    assert "trend" in df.columns
    assert df["trend"].isin([-1, 0, 1]).all()


def test_choch_flips_trend():
    """CHoCH must only fire when trend was already established in opposite direction."""
    df = make_ohlcv(200, seed=7)
    df = detect_fractals(df)
    df = detect_bos_choch(df)
    # choch_bullish can only fire where trend was -1 previously
    choch_bull_idx = df.index[df["choch_bullish"]]
    for idx in choch_bull_idx:
        pos = df.index.get_loc(idx)
        if pos > 0:
            assert df["trend"].iloc[pos - 1] == -1, "CHoCH bullish requires prior bearish trend"
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_structure.py -v
```
Expected: `ImportError: cannot import name 'detect_fractals'`

- [ ] **Step 3: Implement `src/smc/structure.py`**

```python
# src/smc/structure.py
import numpy as np
import pandas as pd


def detect_fractals(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    Detect swing highs and lows using an N-bar fractal rule.

    A swing high at bar i: df.high[i] is the maximum over [i-n .. i+n].
    A swing low  at bar i: df.low[i]  is the minimum over [i-n .. i+n].

    Bars within n of the edges are set to False (insufficient confirmation).
    """
    df = df.copy()
    n_bars = 2 * n + 1

    rolling_max = df["high"].rolling(n_bars, center=True).max()
    rolling_min = df["low"].rolling(n_bars, center=True).min()

    df["swing_high"] = (df["high"] == rolling_max) & df["high"].notna()
    df["swing_low"] = (df["low"] == rolling_min) & df["low"].notna()

    # Edge masking: rolling(center=True) with window=2n+1 needs n bars on each side
    df.loc[df.index[:n], "swing_high"] = False
    df.loc[df.index[-n:], "swing_high"] = False
    df.loc[df.index[:n], "swing_low"] = False
    df.loc[df.index[-n:], "swing_low"] = False

    return df


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS Bullish : close > last confirmed swing_high (structure continuation up)
    BOS Bearish : close < last confirmed swing_low  (structure continuation down)
    CHoCH Bullish: BOS bullish while prevailing trend was bearish (-1)
    CHoCH Bearish: BOS bearish while prevailing trend was bullish (+1)

    Requires detect_fractals() to have been run first.
    """
    df = df.copy()

    # Forward-fill last swing high/low price (the level, not a bool)
    last_sh_price = df["high"].where(df["swing_high"]).ffill()
    last_sl_price = df["low"].where(df["swing_low"]).ffill()

    # BOS: current close breaks the PREVIOUS confirmed swing level
    df["bos_bullish"] = df["close"] > last_sh_price.shift(1)
    df["bos_bearish"] = df["close"] < last_sl_price.shift(1)

    # Derive trend state: +1 (bullish), -1 (bearish), 0 (undefined)
    trend_signal = np.where(df["bos_bullish"], 1, np.where(df["bos_bearish"], -1, np.nan))
    trend_series = pd.Series(trend_signal, index=df.index, dtype=float)
    trend_series = trend_series.ffill().fillna(0).astype(int)
    df["trend"] = trend_series

    # CHoCH: BOS that contradicts the previous bar's trend
    prior_trend = df["trend"].shift(1).fillna(0).astype(int)
    df["choch_bullish"] = df["bos_bullish"] & (prior_trend == -1)
    df["choch_bearish"] = df["bos_bearish"] & (prior_trend == 1)

    return df
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_structure.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/smc/structure.py tests/test_structure.py
git commit -m "feat: fractal swing detection + BOS/CHoCH"
```

---

## Task 3: Fair Value Gap (FVG)

**Files:**
- Create: `src/smc/fvg.py`
- Test: `tests/test_fvg.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fvg.py
import pandas as pd
import numpy as np
import pytest
from src.smc.fvg import detect_fvg


def _make_bullish_fvg_df():
    """Manually craft a 5-bar dataframe with one bullish FVG at bar 2."""
    # Bar 0: high=100, Bar 1: big bull candle, Bar 2: low=102 > bar0.high=100 → FVG
    data = {
        "open":  [99.0,  100.0, 103.0, 104.0, 105.0],
        "high":  [100.0, 105.0, 106.0, 107.0, 108.0],
        "low":   [98.0,   99.0, 102.0, 103.0, 104.0],  # bar2.low=102 > bar0.high=100
        "close": [99.5,  104.0, 105.5, 106.5, 107.5],
        "volume":[500.0,  800.0, 400.0, 300.0, 350.0],
    }
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def _make_bearish_fvg_df():
    """Manually craft a 5-bar dataframe with one bearish FVG at bar 2."""
    data = {
        "open":  [105.0, 104.0, 101.0, 100.0,  99.0],
        "high":  [106.0, 105.0, 102.0, 101.0, 100.0],
        "low":   [104.0, 100.0,  98.0,  97.0,  96.0],  # bar2.high=102 < bar0.low=104
        "close": [104.5, 101.0,  99.0,  98.0,  97.0],
        "volume":[500.0, 800.0, 400.0, 300.0, 350.0],
    }
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def test_bullish_fvg_detected():
    df = detect_fvg(_make_bullish_fvg_df())
    assert df["fvg_bull"].iloc[2], "Bar 2 should be bullish FVG"
    assert df["fvg_bull_size"].iloc[2] == pytest.approx(2.0)  # 102 - 100
    assert not df["fvg_bull"].iloc[0]
    assert not df["fvg_bull"].iloc[1]


def test_bearish_fvg_detected():
    df = detect_fvg(_make_bearish_fvg_df())
    assert df["fvg_bear"].iloc[2], "Bar 2 should be bearish FVG"
    assert df["fvg_bear_size"].iloc[2] == pytest.approx(2.0)  # 104 - 102
    assert not df["fvg_bear"].iloc[0]


def test_fvg_size_zero_when_no_gap():
    df = _make_bullish_fvg_df()
    df.loc[df.index[2], "low"] = 99.0  # overlap removes the gap
    result = detect_fvg(df)
    assert result["fvg_bull_size"].iloc[2] == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_fvg.py -v
```
Expected: `ImportError: cannot import name 'detect_fvg'`

- [ ] **Step 3: Implement `src/smc/fvg.py`**

```python
# src/smc/fvg.py
import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (3-candle imbalances).

    Bullish FVG at bar i: df.low[i] > df.high[i-2]  → gap between candle[i-2].high and candle[i].low
    Bearish FVG at bar i: df.high[i] < df.low[i-2]  → gap between candle[i].high and candle[i-2].low

    Adds columns:
        fvg_bull      (bool)   : True if bullish FVG at this bar
        fvg_bull_size (float)  : size of bullish gap in price units (0 if none)
        fvg_bear      (bool)   : True if bearish FVG at this bar
        fvg_bear_size (float)  : size of bearish gap in price units (0 if none)
    """
    df = df.copy()

    high_2ago = df["high"].shift(2)
    low_2ago = df["low"].shift(2)

    bull_gap = df["low"] - high_2ago
    bear_gap = low_2ago - df["high"]

    df["fvg_bull"] = bull_gap > 0
    df["fvg_bull_size"] = bull_gap.clip(lower=0).fillna(0.0)

    df["fvg_bear"] = bear_gap > 0
    df["fvg_bear_size"] = bear_gap.clip(lower=0).fillna(0.0)

    return df
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_fvg.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/smc/fvg.py tests/test_fvg.py
git commit -m "feat: FVG detection (bullish + bearish)"
```

---

## Task 4: High-Probability Order Block Detection (4 Filters)

**Files:**
- Create: `src/smc/order_blocks.py`
- Test: `tests/test_order_blocks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_order_blocks.py
import pandas as pd
import numpy as np
import pytest
from src.smc.structure import detect_fractals, detect_bos_choch
from src.smc.fvg import detect_fvg
from src.smc.order_blocks import detect_order_blocks


def _build_pipeline(df):
    df = detect_fractals(df, n=2)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    return detect_order_blocks(df)


def test_ob_columns_exist():
    from tests.conftest import make_ohlcv
    df = _build_pipeline(make_ohlcv(300))
    for col in ["bull_ob", "bear_ob", "ob_top", "ob_bottom",
                "ob_midpoint", "ob_type", "vol_zscore"]:
        assert col in df.columns, f"Missing column: {col}"


def test_ob_type_values():
    from tests.conftest import make_ohlcv
    df = _build_pipeline(make_ohlcv(300))
    valid = {0, 1, -1}
    assert set(df["ob_type"].unique()).issubset(valid)


def test_vol_zscore_no_nan_after_warmup():
    from tests.conftest import make_ohlcv
    df = _build_pipeline(make_ohlcv(300))
    # After 50-bar warmup, vol_zscore must be finite
    assert df["vol_zscore"].iloc[60:].notna().all()


def test_ob_midpoint_between_top_bottom():
    from tests.conftest import make_ohlcv
    df = _build_pipeline(make_ohlcv(300))
    ob_rows = df[df["bull_ob"] | df["bear_ob"]]
    if len(ob_rows) > 0:
        assert (ob_rows["ob_midpoint"] <= ob_rows["ob_top"]).all()
        assert (ob_rows["ob_midpoint"] >= ob_rows["ob_bottom"]).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_order_blocks.py -v
```

- [ ] **Step 3: Implement `src/smc/order_blocks.py`**

```python
# src/smc/order_blocks.py
"""
High-Probability Order Block Detection — 4 Filters

Filter 1: The OB candle must have caused a BOS or CHoCH within the next LOOKAHEAD bars.
Filter 2: A FVG must exist in the bar immediately following the OB (consecutive imbalance).
Filter 3: The OB candle must have swept the prior bar's High (bearish OB) or Low (bullish OB).
Filter 4: The break candle (the BOS/CHoCH bar) must have vol_zscore > VOL_ZSCORE_MIN.

Bullish OB: last bearish candle (close < open) before a bullish BOS/CHoCH.
Bearish OB: last bullish candle (close > open) before a bearish BOS/CHoCH.
"""
import numpy as np
import pandas as pd

LOOKAHEAD = 20        # bars to look forward for a confirming BOS/CHoCH
VOL_ZSCORE_MIN = 1.5  # minimum volume z-score on the break candle
VOL_WINDOW = 50       # rolling window for volume z-score


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns from detect_fractals + detect_bos_choch + detect_fvg.

    Adds:
        vol_zscore  (float): rolling 50-bar volume z-score
        bull_ob     (bool) : True if bar is a confirmed bullish OB
        bear_ob     (bool) : True if bar is a confirmed bearish OB
        ob_top      (float): OB high (NaN if not OB)
        ob_bottom   (float): OB low  (NaN if not OB)
        ob_midpoint (float): (ob_top + ob_bottom) / 2
        ob_type     (int)  : 1=bull, -1=bear, 0=none
    """
    df = df.copy()

    # --- Volume Z-Score ---
    vol_mean = df["volume"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    vol_std  = df["volume"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().replace(0, np.nan)
    df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

    n = len(df)
    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)

    bos_bull_arr  = df["bos_bullish"].to_numpy()
    bos_bear_arr  = df["bos_bearish"].to_numpy()
    choch_bull    = df["choch_bullish"].to_numpy()
    choch_bear    = df["choch_bearish"].to_numpy()
    fvg_bull_arr  = df["fvg_bull"].to_numpy()
    fvg_bear_arr  = df["fvg_bear"].to_numpy()
    vol_z         = df["vol_zscore"].to_numpy()
    open_arr      = df["open"].to_numpy()
    close_arr     = df["close"].to_numpy()
    high_arr      = df["high"].to_numpy()
    low_arr       = df["low"].to_numpy()

    for i in range(1, n - LOOKAHEAD):
        # --- Bullish OB candidate: bearish candle (close < open) ---
        if close_arr[i] < open_arr[i]:
            # Filter 3: swept prior bar's low
            swept_low = low_arr[i] < low_arr[i - 1]
            if not swept_low:
                continue
            # Filter 2: FVG on the very next bar
            if i + 1 >= n or not fvg_bull_arr[i + 1]:
                continue
            # Filter 1+4: within LOOKAHEAD bars, find a bullish BOS/CHoCH with vol z > threshold
            window = slice(i + 1, min(i + 1 + LOOKAHEAD, n))
            bos_in_window  = bos_bull_arr[window] | choch_bull[window]
            vol_in_window  = vol_z[window]
            confirmed = any(
                bos_in_window[k] and vol_in_window[k] > VOL_ZSCORE_MIN
                for k in range(len(bos_in_window))
            )
            if confirmed:
                bull_ob[i] = True

        # --- Bearish OB candidate: bullish candle (close > open) ---
        elif close_arr[i] > open_arr[i]:
            # Filter 3: swept prior bar's high
            swept_high = high_arr[i] > high_arr[i - 1]
            if not swept_high:
                continue
            # Filter 2: FVG on the very next bar
            if i + 1 >= n or not fvg_bear_arr[i + 1]:
                continue
            # Filter 1+4
            window = slice(i + 1, min(i + 1 + LOOKAHEAD, n))
            bos_in_window  = bos_bear_arr[window] | choch_bear[window]
            vol_in_window  = vol_z[window]
            confirmed = any(
                bos_in_window[k] and vol_in_window[k] > VOL_ZSCORE_MIN
                for k in range(len(bos_in_window))
            )
            if confirmed:
                bear_ob[i] = True

    df["bull_ob"] = bull_ob
    df["bear_ob"] = bear_ob

    df["ob_top"] = np.where(
        bull_ob, df["high"], np.where(bear_ob, df["high"], np.nan)
    )
    df["ob_bottom"] = np.where(
        bull_ob, df["low"], np.where(bear_ob, df["low"], np.nan)
    )
    df["ob_midpoint"] = (df["ob_top"] + df["ob_bottom"]) / 2.0
    df["ob_type"] = np.where(bull_ob, 1, np.where(bear_ob, -1, 0)).astype(int)

    return df
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_order_blocks.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/smc/order_blocks.py tests/test_order_blocks.py
git commit -m "feat: 4-filter high-probability OB detection"
```

---

## Task 5: Liquidity Detection

**Files:**
- Create: `src/smc/liquidity.py`
- Test: `tests/test_liquidity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_liquidity.py
import pandas as pd
import numpy as np
from src.smc.liquidity import detect_equal_highs_lows, detect_asia_session


def _make_equal_high_df():
    """Two bars with nearly identical highs (within tolerance) → equal highs pool."""
    dates = pd.date_range("2024-01-02 10:00", periods=10, freq="5min", tz="UTC")
    data = {
        "open":   [100.0]*10,
        "high":   [105.0, 104.9, 102.0, 103.0, 105.02, 104.0, 103.0, 102.0, 101.0, 100.0],
        "low":    [99.0]*10,
        "close":  [101.0]*10,
        "volume": [500.0]*10,
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="timestamp"))


def test_equal_highs_detected():
    df = detect_equal_highs_lows(_make_equal_high_df(), tolerance_pips=5.0)
    # Bars 0 and 4 have highs 105.0 and 105.02 → within 5 pip (0.50 price) tolerance
    assert "equal_high_pool" in df.columns
    assert df["equal_high_pool"].any(), "Should detect at least one equal high pool"


def test_asia_session_range():
    dates = pd.date_range("2024-01-02 00:00", periods=288, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    data = {
        "open": rng.uniform(1900, 1910, 288),
        "high": rng.uniform(1910, 1920, 288),
        "low":  rng.uniform(1890, 1900, 288),
        "close":rng.uniform(1900, 1910, 288),
        "volume": rng.uniform(100, 1000, 288),
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="timestamp"))
    result = detect_asia_session(df)
    assert "asia_high" in result.columns
    assert "asia_low" in result.columns
    assert "judas_swing_bull" in result.columns
    assert "judas_swing_bear" in result.columns
    # Non-Asia hours should have asia_high/low filled forward
    post_asia = result.between_time("08:01", "23:59")
    assert post_asia["asia_high"].notna().all()
```

- [ ] **Step 2: Implement `src/smc/liquidity.py`**

```python
# src/smc/liquidity.py
"""
Liquidity pool detection:
  - Equal Highs / Equal Lows (within pip tolerance)
  - Asia session range (00:00–08:00 UTC)
  - Judas Swing: price temporarily breaks Asia range then reverses
"""
import numpy as np
import pandas as pd

PIP_SIZE = 0.10  # 1 pip = $0.10 for XAUUSD


def detect_equal_highs_lows(
    df: pd.DataFrame,
    tolerance_pips: float = 5.0,
    lookback: int = 50,
) -> pd.DataFrame:
    """
    Mark bars where the high (or low) matches a prior high (or low) within
    tolerance_pips. These form liquidity pools (stop clusters).

    Adds:
        equal_high_pool (bool): True if this bar's high ≈ a prior high in lookback
        equal_low_pool  (bool): True if this bar's low  ≈ a prior low  in lookback
    """
    df = df.copy()
    tolerance = tolerance_pips * PIP_SIZE

    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    n = len(df)

    eq_high = np.zeros(n, dtype=bool)
    eq_low  = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        window_highs = highs[max(0, i - lookback): i]
        window_lows  = lows[max(0, i - lookback): i]
        if np.any(np.abs(window_highs - highs[i]) <= tolerance):
            eq_high[i] = True
        if np.any(np.abs(window_lows - lows[i]) <= tolerance):
            eq_low[i] = True

    df["equal_high_pool"] = eq_high
    df["equal_low_pool"]  = eq_low
    return df


def detect_asia_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Asia session range (00:00–08:00 UTC) for each trading day.
    Then detect Judas Swings: a move that sweeps the Asia H/L then reverses.

    Adds:
        asia_high        (float): Asia session high for that day, forward-filled
        asia_low         (float): Asia session low for that day, forward-filled
        in_asia_session  (bool) : True during 00:00–08:00 UTC
        judas_swing_bull (bool) : price swept Asia low then recovered above it
        judas_swing_bear (bool) : price swept Asia high then dropped below it
    """
    df = df.copy()

    df["_date"] = df.index.normalize()
    df["in_asia_session"] = (df.index.hour >= 0) & (df.index.hour < 8)

    asia = df[df["in_asia_session"]].groupby("_date").agg(
        asia_high=("high", "max"),
        asia_low=("low", "min"),
    )

    df = df.join(asia, on="_date", how="left")
    # Forward-fill within each day so non-Asia hours carry the day's Asia range
    df["asia_high"] = df["asia_high"].ffill()
    df["asia_low"]  = df["asia_low"].ffill()

    # Judas Swing Bearish: swept Asia high (high > asia_high) then close drops back
    df["judas_swing_bear"] = (
        (df["high"] > df["asia_high"]) &
        (df["close"] < df["asia_high"]) &
        (~df["in_asia_session"])
    )

    # Judas Swing Bullish: swept Asia low (low < asia_low) then close recovers
    df["judas_swing_bull"] = (
        (df["low"] < df["asia_low"]) &
        (df["close"] > df["asia_low"]) &
        (~df["in_asia_session"])
    )

    df = df.drop(columns=["_date"])
    return df
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_liquidity.py -v
```
Expected: 2 PASSED

- [ ] **Step 4: Commit**

```bash
git add src/smc/liquidity.py tests/test_liquidity.py
git commit -m "feat: equal H/L pools + Asia session + Judas swing detection"
```

---

## Task 6: Technical Features (RSI-Lagless + ATR)

**Files:**
- Create: `src/features/technical.py`
- Test: `tests/test_technical.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_technical.py
import pytest
from tests.conftest import make_ohlcv
from src.features.technical import add_rsi_lagless, add_atr


def test_rsi_lagless_bounded():
    df = add_rsi_lagless(make_ohlcv(200))
    assert "rsi_lagless" in df.columns
    valid = df["rsi_lagless"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_lagless_no_future_data():
    """RSI at bar i must only use bars [0..i]."""
    df = make_ohlcv(200)
    result = add_rsi_lagless(df)
    # If we drop the last 10 bars and recompute, values up to bar N-10 must be identical
    result_short = add_rsi_lagless(df.iloc[:-10])
    common_idx = result_short.index
    pd.testing.assert_series_equal(
        result.loc[common_idx, "rsi_lagless"],
        result_short["rsi_lagless"],
        check_names=False,
    )


def test_atr_positive():
    df = add_atr(make_ohlcv(200))
    assert "atr" in df.columns
    assert (df["atr"].dropna() > 0).all()
```

- [ ] **Step 2: Implement `src/features/technical.py`**

```python
# src/features/technical.py
"""
Lag-reduced technical indicators for Transformer feature input.

RSI-Lagless: Ehlers' smoothed RSI using a 2-pole Super Smoother filter
             (replaces EMA's phase lag with minimal-lag IIR filter).
ATR        : 14-bar Average True Range (Wilder smoothing).
"""
import numpy as np
import pandas as pd


def add_rsi_lagless(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Lagless RSI using John Ehlers' Super Smoother pre-filter on price changes.

    The Super Smoother (2-pole) drastically cuts lag vs standard EMA-based RSI
    while preserving the [0,100] bounded output.

    Reference: Ehlers, J. (2013) Cybernetic Analysis for Stocks and Futures, ch. 3.
    """
    df = df.copy()
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    # Super Smoother coefficients (cutoff = period bars)
    import math
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # Apply Super Smoother to price
    ss = np.zeros(n)
    for i in range(2, n):
        ss[i] = c1 * (close[i] + close[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]

    # Compute RSI on smoothed price
    delta = np.diff(ss, prepend=ss[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    # Wilder smoothing (EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().to_numpy()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan

    df["rsi_lagless"] = rsi
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """14-bar ATR using Wilder smoothing."""
    df = df.copy()
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return df
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_technical.py -v
```
Expected: 3 PASSED

- [ ] **Step 4: Commit**

```bash
git add src/features/technical.py tests/test_technical.py
git commit -m "feat: RSI-Lagless (Ehlers Super Smoother) + ATR"
```

---

## Task 7: Look-Ahead-Safe Timeframe Alignment

**Files:**
- Create: `src/alignment.py`
- Test: `tests/test_alignment.py`

**Design rule:** For each M5 bar at time T, we join M15/H1 features using the LAST COMPLETED bar whose close time ≤ T. An H1 bar opening at 10:00 and closing at 11:00 must NOT be available to M5 bars before 11:00.

- [ ] **Step 1: Write failing tests — focus on bias regression**

```python
# tests/test_alignment.py
import pandas as pd
import numpy as np
import pytest
from src.alignment import align_timeframes


def _make_tf(freq: str, n: int, base_price: float = 1900.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02 00:00", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close - 0.1, "high": close + 1.0,
        "low": close - 1.0, "close": close, "volume": rng.uniform(100, 1000, n),
        "trend": rng.choice([-1, 0, 1], n),
    }, index=pd.DatetimeIndex(dates, name="timestamp"))


def test_alignment_no_lookahead_h1():
    """
    The H1 feature at an M5 timestamp T must come from an H1 bar
    whose open_time + 1h <= T  (i.e., bar closed before or at T).
    """
    m5  = _make_tf("5min", 288)    # 1 day
    m15 = _make_tf("15min", 96)
    h1  = _make_tf("1h", 24)

    result = align_timeframes(m5, m15, h1)

    # For each M5 row, the h1_close used must be <= the M5 timestamp
    # The H1 bar at time T opens at T and closes at T+1h, so it's only
    # available AFTER T+1h.  align_timeframes shifts h1 by 1 period.
    assert "h1_close" in result.columns
    assert "h1_trend" in result.columns
    assert "m15_close" in result.columns

    # Check: at M5 bar at 01:00 UTC, h1_close must be from the 00:00 H1 bar
    # (which closed at 01:00, so the value is the 00:00 bar's close)
    m5_at_01 = result.at_time("01:00")
    if len(m5_at_01) > 0:
        h1_00_close = h1.iloc[0]["close"]
        assert m5_at_01["h1_close"].iloc[0] == pytest.approx(h1_00_close, rel=1e-6)


def test_alignment_m5_index_preserved():
    m5  = _make_tf("5min", 288)
    m15 = _make_tf("15min", 96)
    h1  = _make_tf("1h", 24)
    result = align_timeframes(m5, m15, h1)
    pd.testing.assert_index_equal(result.index, m5.index)


def test_alignment_no_nulls_after_warmup():
    m5  = _make_tf("5min", 500)
    m15 = _make_tf("15min", 167)
    h1  = _make_tf("1h", 21)
    result = align_timeframes(m5, m15, h1)
    # After 1 H1 bar worth of warmup (12 M5 bars), h1 columns must be filled
    late = result.iloc[15:]
    assert late[["h1_close", "m15_close"]].notna().all().all()
```

- [ ] **Step 2: Implement `src/alignment.py`**

```python
# src/alignment.py
"""
Look-ahead-safe multi-timeframe alignment.

Strategy:
  - H1 bar opens at T and CLOSES at T + 1h.
    We shift H1 index by +1 period so that bar T is indexed at T+1h
    (its close time). Then merge_asof backward ensures M5 bar at time X
    only gets the last H1 bar whose close_time <= X.
  - Same logic applies to M15 (shift +1 period = +15min).

This guarantees strict causal ordering: no M5 bar ever sees data from
a higher-timeframe bar that hasn't closed yet.
"""
import pandas as pd


_H1_COLS  = ["close", "high", "low", "trend", "vol_zscore",
             "asia_high", "asia_low", "bull_ob", "bear_ob",
             "ob_midpoint", "bos_bullish", "bos_bearish"]

_M15_COLS = ["close", "high", "low", "trend", "vol_zscore",
             "bull_ob", "bear_ob", "ob_midpoint",
             "bos_bullish", "bos_bearish", "fvg_bull", "fvg_bear"]


def _shift_to_close_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Re-index a DataFrame so each bar is stamped at its CLOSE time."""
    offset = pd.tseries.frequencies.to_offset(freq)
    df = df.copy()
    df.index = df.index + offset
    return df


def _prefix_cols(df: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
    """Keep only 'cols' that exist in df and rename with prefix."""
    available = [c for c in cols if c in df.columns]
    return df[available].rename(columns={c: f"{prefix}_{c}" for c in available})


def align_timeframes(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    h1: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge M15 and H1 features into the M5 DataFrame without look-ahead bias.

    Each M5 row receives the most recent M15/H1 bar that has fully closed
    before or at that M5 timestamp.

    Returns:
        m5 DataFrame with additional columns prefixed 'h1_' and 'm15_'.
    """
    # Stamp each higher-TF bar at its close time
    h1_close_stamped  = _shift_to_close_time(h1,  "1h")
    m15_close_stamped = _shift_to_close_time(m15, "15min")

    h1_slim  = _prefix_cols(h1_close_stamped,  "h1",  _H1_COLS)
    m15_slim = _prefix_cols(m15_close_stamped, "m15", _M15_COLS)

    result = pd.merge_asof(
        m5.sort_index(),
        h1_slim.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    result = pd.merge_asof(
        result,
        m15_slim.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return result
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_alignment.py -v
```
Expected: 3 PASSED

- [ ] **Step 4: Commit**

```bash
git add src/alignment.py tests/test_alignment.py
git commit -m "feat: look-ahead-safe M5←M15←H1 alignment via close-time shift"
```

---

## Task 8: Risk Management (Entry, SL, TP, DD Filter)

**Files:**
- Create: `src/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_risk.py
import pandas as pd
import numpy as np
import pytest
from src.risk import compute_risk_params, BALANCE


def _ob_row(ob_top=1910.0, ob_bottom=1908.0, atr=2.0,
            next_unmitigated_ob=1920.0, ob_type=1):
    return {
        "ob_top": ob_top, "ob_bottom": ob_bottom,
        "ob_midpoint": (ob_top + ob_bottom) / 2,
        "atr": atr, "ob_type": ob_type,
        "next_unmit_ob_h1": next_unmitigated_ob,
    }


def test_bullish_ob_entry_at_midpoint():
    row = _ob_row()
    r = compute_risk_params(row)
    assert r["entry"] == pytest.approx(1909.0)  # midpoint


def test_sl_below_ob_low_plus_atr_buffer():
    row = _ob_row(ob_bottom=1908.0, atr=2.0)
    r = compute_risk_params(row)
    # SL = ob_bottom - 2 pips - (atr * ATR_MULT)
    # 2 pips = 0.20, ATR_MULT = 0.25 → 1908 - 0.20 - 0.50 = 1907.30
    assert r["sl"] == pytest.approx(1907.30, abs=0.01)


def test_tp_minimum_rr():
    row = _ob_row(ob_top=1910.0, ob_bottom=1908.0, atr=2.0,
                  next_unmitigated_ob=1909.5)  # too close → use RR 2.5
    r = compute_risk_params(row)
    risk = r["entry"] - r["sl"]
    assert r["tp"] >= r["entry"] + 2.5 * risk


def test_dd_filter_rejects_large_sl():
    """If SL risk > 0.5% of balance, flag should be set."""
    # Make a huge ATR so SL is very large
    row = _ob_row(ob_bottom=1900.0, atr=100.0)
    r = compute_risk_params(row)
    # Entry 1905, SL ~1899.3 → risk 5.7 → for 1 lot on XAUUSD
    # Just check the flag exists and is bool
    assert isinstance(r["dd_filter_ok"], bool)
    # With huge ATR the filter should fail
    assert not r["dd_filter_ok"]


def test_dd_filter_passes_normal_setup():
    row = _ob_row(ob_bottom=1908.0, atr=1.5)
    r = compute_risk_params(row)
    assert r["dd_filter_ok"]
```

- [ ] **Step 2: Implement `src/risk.py`**

```python
# src/risk.py
"""
Risk parameter computation for each OB-based setup.

XAUUSD specs (ADN broker ADNX10):
  - 1 pip = $0.10 price movement
  - Standard lot = 100 oz, so 1 pip on 1 lot = $10 USD
  - DD filter: projected $ risk on 1 lot must be < 0.5% of BALANCE
"""

PIP_SIZE   = 0.10   # XAUUSD: $0.10 per pip
PIPS_SL    = 2.0    # fixed pip buffer below/above OB
ATR_MULT   = 0.25   # ATR contribution to SL buffer
MIN_RR     = 2.5    # minimum reward:risk ratio
LOT_SIZE   = 100.0  # oz per standard lot (XAUUSD)
BALANCE    = 10_000.0  # assumed account balance (USD)
DD_LIMIT   = 0.005  # 0.5% of balance max risk per trade


def compute_risk_params(row: dict) -> dict:
    """
    Compute entry, SL, TP and DD filter flag for one OB setup.

    Args:
        row: dict with keys: ob_top, ob_bottom, ob_midpoint, atr,
             ob_type (1=bull, -1=bear), next_unmit_ob_h1

    Returns:
        dict with: entry, sl, tp, risk_pips, rr_achieved, dd_filter_ok
    """
    ob_type   = int(row["ob_type"])
    ob_top    = float(row["ob_top"])
    ob_bottom = float(row["ob_bottom"])
    midpoint  = float(row["ob_midpoint"])
    atr       = float(row["atr"])
    next_ob   = float(row.get("next_unmit_ob_h1", 0) or 0)

    sl_buffer = PIPS_SL * PIP_SIZE + atr * ATR_MULT

    if ob_type == 1:  # Bullish OB — long trade
        entry = midpoint
        sl    = ob_bottom - sl_buffer
        risk  = entry - sl
        if risk <= 0:
            risk = sl_buffer  # safety floor

        # TP: use next H1 unmitigated OB if it gives ≥ MIN_RR, else RR-based
        tp_target = next_ob if next_ob > entry + MIN_RR * risk else entry + MIN_RR * risk
        tp = max(tp_target, entry + MIN_RR * risk)

    else:  # Bearish OB — short trade
        entry = midpoint
        sl    = ob_top + sl_buffer
        risk  = sl - entry
        if risk <= 0:
            risk = sl_buffer

        tp_target = next_ob if (0 < next_ob < entry - MIN_RR * risk) else entry - MIN_RR * risk
        tp = min(tp_target, entry - MIN_RR * risk)

    rr_achieved = abs(tp - entry) / risk if risk > 0 else 0.0

    # DD filter: $ risk on 1 standard lot
    dollar_risk = risk * LOT_SIZE
    dd_filter_ok = dollar_risk < DD_LIMIT * BALANCE

    return {
        "entry":        round(entry, 2),
        "sl":           round(sl,    2),
        "tp":           round(tp,    2),
        "risk_price":   round(risk,  4),
        "rr_achieved":  round(rr_achieved, 2),
        "dd_filter_ok": dd_filter_ok,
    }
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_risk.py -v
```
Expected: 5 PASSED

- [ ] **Step 4: Commit**

```bash
git add src/risk.py tests/test_risk.py
git commit -m "feat: OB-based entry/SL/TP + 0.5% DD filter"
```

---

## Task 9: GoldQuantProcessor Orchestrator

**Files:**
- Create: `src/processor.py`
- Test: `tests/test_processor.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_processor.py
import pytest
from pathlib import Path
from src.processor import GoldQuantProcessor


@pytest.fixture(scope="module")
def processor():
    return GoldQuantProcessor(
        m5_file="xauusd_m5_history.csv",
        m15_file="xauusd_m15_history.csv",
        h1_file="xauusd_h1_history.csv",
    )


def test_detect_smc_runs(processor):
    processor.detect_smc()
    assert processor.m5 is not None
    assert "bull_ob" in processor.m5.columns
    assert "trend" in processor.m5.columns


def test_align_timeframes_runs(processor):
    processor.detect_smc()
    processor.align_timeframes()
    assert processor.aligned is not None
    assert "h1_trend" in processor.aligned.columns
    assert "m15_trend" in processor.aligned.columns


def test_generate_training_tensors_produces_parquet(processor, tmp_path):
    processor.detect_smc()
    processor.align_timeframes()
    out = processor.generate_training_tensors(output_dir=tmp_path)
    parquet_files = list(tmp_path.glob("*.parquet"))
    assert len(parquet_files) >= 1, "Expected at least one .parquet output"
```

- [ ] **Step 2: Implement `src/processor.py`**

```python
# src/processor.py
"""
GoldQuantProcessor — main orchestrator for the SMC feature pipeline.

Usage:
    proc = GoldQuantProcessor("xauusd_m5_history.csv",
                              "xauusd_m15_history.csv",
                              "xauusd_h1_history.csv")
    proc.detect_smc()
    proc.align_timeframes()
    proc.generate_training_tensors(output_dir="data/processed")
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.loader import load_mt5_csv
from src.smc.structure import detect_fractals, detect_bos_choch
from src.smc.fvg import detect_fvg
from src.smc.order_blocks import detect_order_blocks
from src.smc.liquidity import detect_equal_highs_lows, detect_asia_session
from src.features.technical import add_rsi_lagless, add_atr
from src.alignment import align_timeframes as _align
from src.risk import compute_risk_params
from src.dataset import save_parquet, save_tfrecord


class GoldQuantProcessor:
    """
    Three-stage pipeline:
      1. detect_smc()              — run full SMC analysis on each timeframe
      2. align_timeframes()        — merge M15/H1 features into M5 (no look-ahead)
      3. generate_training_tensors() — normalize and serialize to Parquet/TFRecord
    """

    def __init__(
        self,
        m5_file: str,
        m15_file: str,
        h1_file: str,
        fractal_n: int = 2,
    ):
        self.m5_file   = m5_file
        self.m15_file  = m15_file
        self.h1_file   = h1_file
        self.fractal_n = fractal_n

        self.m5:      Optional[pd.DataFrame] = None
        self.m15:     Optional[pd.DataFrame] = None
        self.h1:      Optional[pd.DataFrame] = None
        self.aligned: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def detect_smc(self) -> None:
        """Load CSVs and run the full SMC detection pipeline on each TF."""
        print("[GoldQuantProcessor] Loading data...")
        m5  = load_mt5_csv(self.m5_file)
        m15 = load_mt5_csv(self.m15_file)
        h1  = load_mt5_csv(self.h1_file)

        print(f"  M5 rows: {len(m5):,}  M15: {len(m15):,}  H1: {len(h1):,}")

        self.m5  = self._run_smc_pipeline(m5,  self.fractal_n)
        self.m15 = self._run_smc_pipeline(m15, self.fractal_n)
        self.h1  = self._run_smc_pipeline_h1(h1, self.fractal_n)

        print("[GoldQuantProcessor] SMC detection complete.")

    # ------------------------------------------------------------------
    def align_timeframes(self) -> None:
        """Merge M15 and H1 SMC features into the M5 DataFrame (no look-ahead)."""
        if self.m5 is None or self.m15 is None or self.h1 is None:
            raise RuntimeError("Call detect_smc() before align_timeframes().")

        print("[GoldQuantProcessor] Aligning timeframes...")
        self.aligned = _align(self.m5, self.m15, self.h1)
        print(f"  Aligned shape: {self.aligned.shape}")

    # ------------------------------------------------------------------
    def generate_training_tensors(
        self,
        output_dir: str | Path = "data/processed",
        save_tf: bool = False,
    ) -> Path:
        """
        Normalize the aligned DataFrame and serialize to Parquet (+ optionally TFRecord).

        Returns path to the saved Parquet file.
        """
        if self.aligned is None:
            raise RuntimeError("Call align_timeframes() before generate_training_tensors().")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.aligned.copy()

        # Compute risk params row-by-row for OB rows
        df = self._add_risk_columns(df)

        # Normalize
        from src.features.normalizer import normalize_features
        df_norm, scaler = normalize_features(df)

        # Serialize
        parquet_path = save_parquet(df_norm, output_dir / "gold_smc_dataset.parquet")
        print(f"[GoldQuantProcessor] Parquet saved → {parquet_path}")

        if save_tf:
            tf_path = save_tfrecord(df_norm, output_dir / "gold_smc_dataset.tfrecord")
            print(f"[GoldQuantProcessor] TFRecord saved → {tf_path}")

        return parquet_path

    # ------------------------------------------------------------------
    @staticmethod
    def _run_smc_pipeline(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """M5 / M15 pipeline: structure → FVG → OB → liquidity → technicals."""
        df = detect_fractals(df, n=n)
        df = detect_bos_choch(df)
        df = detect_fvg(df)
        df = detect_order_blocks(df)
        df = detect_equal_highs_lows(df)
        df = detect_asia_session(df)
        df = add_rsi_lagless(df)
        df = add_atr(df)
        # OB freshness: bars since this OB was formed (0 on OB bar, +1 each bar)
        df = _add_ob_freshness(df)
        return df

    @staticmethod
    def _run_smc_pipeline_h1(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """H1 pipeline — same as M5/M15 plus previous-day H/L."""
        df = GoldQuantProcessor._run_smc_pipeline(df, n)
        df["prev_day_high"] = df["high"].resample("D").transform("max").shift(1, freq="D")
        df["prev_day_low"]  = df["low"].resample("D").transform("min").shift(1, freq="D")
        return df

    @staticmethod
    def _add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add entry/sl/tp/dd_filter_ok columns using compute_risk_params."""
        ob_mask = df["ob_type"] != 0
        risk_cols = ["entry", "sl", "tp", "risk_price", "rr_achieved", "dd_filter_ok"]
        for col in risk_cols:
            df[col] = np.nan if col != "dd_filter_ok" else False

        for idx in df.index[ob_mask]:
            row = df.loc[idx]
            # Pass next H1 unmitigated OB if available
            row_dict = row.to_dict()
            row_dict.setdefault("next_unmit_ob_h1", 0)
            params = compute_risk_params(row_dict)
            for col, val in params.items():
                df.at[idx, col] = val
        return df


def _add_ob_freshness(df: pd.DataFrame) -> pd.DataFrame:
    """Track how many bars ago the most recent OB was formed."""
    df = df.copy()
    ob_any = (df["bull_ob"] | df["bear_ob"]).to_numpy()
    freshness = np.full(len(df), np.nan)
    last_ob = np.nan
    for i in range(len(df)):
        if ob_any[i]:
            last_ob = 0
        if not np.isnan(last_ob):
            freshness[i] = last_ob
            last_ob += 1
    df["ob_freshness"] = freshness
    return df
```

- [ ] **Step 3: Run smoke test**

```bash
python -m pytest tests/test_processor.py -v -s
```
Expected: 3 PASSED (will take ~2–3 min for full M5 dataset)

- [ ] **Step 4: Commit**

```bash
git add src/processor.py tests/test_processor.py
git commit -m "feat: GoldQuantProcessor orchestrator (detect_smc, align_timeframes, generate_training_tensors)"
```

---

## Task 10: Feature Normalization + Dataset Serialization

**Files:**
- Create: `src/features/normalizer.py`
- Create: `src/dataset.py`

- [ ] **Step 1: Implement `src/features/normalizer.py`**

```python
# src/features/normalizer.py
"""
Normalize feature groups using RobustScaler (preferred for financial data
with outliers) before feeding into the Transformer attention layers.

Feature groups:
  PRICE_FEATURES   → absolute price columns (open/high/low/close, OB levels)
  INDICATOR_FEATURES → bounded [0..100] RSI, ATR (ratio to price)
  STRUCTURAL_FEATURES → binary/categorical flags (trend, ob_type, etc.)
"""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

PRICE_FEATURES = [
    "open", "high", "low", "close",
    "ob_top", "ob_bottom", "ob_midpoint",
    "asia_high", "asia_low",
    "h1_close", "h1_high", "h1_low", "h1_ob_midpoint",
    "m15_close", "m15_high", "m15_low", "m15_ob_midpoint",
    "entry", "sl", "tp",
]

INDICATOR_FEATURES = [
    "rsi_lagless", "atr", "vol_zscore",
    "fvg_bull_size", "fvg_bear_size",
    "ob_freshness", "risk_price", "rr_achieved",
]

PASSTHROUGH_FEATURES = [
    # Binary/categorical — no scaling
    "trend", "ob_type", "bull_ob", "bear_ob",
    "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
    "fvg_bull", "fvg_bear", "equal_high_pool", "equal_low_pool",
    "in_asia_session", "judas_swing_bull", "judas_swing_bear",
    "dd_filter_ok",
    "h1_trend", "h1_bull_ob", "h1_bear_ob",
    "m15_trend", "m15_bull_ob", "m15_bear_ob",
    "h1_bos_bullish", "h1_bos_bearish",
    "m15_bos_bullish", "m15_bos_bearish",
]


def normalize_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize PRICE_FEATURES and INDICATOR_FEATURES using RobustScaler.
    PASSTHROUGH_FEATURES are kept as-is.

    Returns (normalized_df, scalers_dict) where scalers_dict maps
    group name → fitted RobustScaler (for inverse transform at inference).
    """
    df = df.copy()
    scalers = {}

    for group_name, cols in [
        ("price", PRICE_FEATURES),
        ("indicators", INDICATOR_FEATURES),
    ]:
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        scaler = RobustScaler()
        data = df[available].to_numpy(dtype=float)
        # Fill NaN with column median before scaling
        col_medians = np.nanmedian(data, axis=0)
        for j in range(data.shape[1]):
            mask = np.isnan(data[:, j])
            data[mask, j] = col_medians[j]
        df[available] = scaler.fit_transform(data)
        scalers[group_name] = scaler

    return df, scalers
```

- [ ] **Step 2: Implement `src/dataset.py`**

```python
# src/dataset.py
"""
Serialize the processed DataFrame to Parquet and TFRecord formats.
"""
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd


FEATURE_COLS = [
    # M5 features
    "open", "high", "low", "close", "volume",
    "rsi_lagless", "atr", "vol_zscore",
    "fvg_bull_size", "fvg_bear_size",
    "ob_freshness", "ob_type", "ob_midpoint",
    "trend", "bos_bullish", "bos_bearish",
    "choch_bullish", "choch_bearish",
    "equal_high_pool", "equal_low_pool",
    "in_asia_session", "judas_swing_bull", "judas_swing_bear",
    "asia_high", "asia_low",
    # M15 features
    "m15_close", "m15_trend", "m15_ob_midpoint",
    "m15_bull_ob", "m15_bear_ob",
    "m15_bos_bullish", "m15_bos_bearish",
    "m15_fvg_bull", "m15_fvg_bear",
    # H1 features
    "h1_close", "h1_trend", "h1_ob_midpoint",
    "h1_bull_ob", "h1_bear_ob",
    "h1_bos_bullish", "h1_bos_bearish",
    "h1_asia_high", "h1_asia_low",
    # Risk / labels
    "entry", "sl", "tp", "rr_achieved", "dd_filter_ok",
]


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save to Parquet with snappy compression. Only FEATURE_COLS are kept."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    out = df[available].copy()
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
    return path


def save_tfrecord(df: pd.DataFrame, path: Path) -> Path:
    """Save to TFRecord (one example per row)."""
    import tensorflow as tf

    available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available].fillna(0).astype("float32")

    def _float_feature(values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    with tf.io.TFRecordWriter(str(path)) as writer:
        for _, row in data.iterrows():
            feature = {col: _float_feature([row[col]]) for col in available}
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())

    return path
```

- [ ] **Step 3: Run full pipeline end-to-end**

```bash
python -c "
from src.processor import GoldQuantProcessor
proc = GoldQuantProcessor('xauusd_m5_history.csv', 'xauusd_m15_history.csv', 'xauusd_h1_history.csv')
proc.detect_smc()
proc.align_timeframes()
out = proc.generate_training_tensors('data/processed')
print('Output:', out)
"
```
Expected: Parquet written to `data/processed/gold_smc_dataset.parquet`, no errors.

- [ ] **Step 4: Commit**

```bash
git add src/features/normalizer.py src/dataset.py
git commit -m "feat: RobustScaler normalization + Parquet/TFRecord serialization"
```

---

## Task 11: Validation / Visualization Script

**Files:**
- Create: `validation/visualizer.py`

- [ ] **Step 1: Implement `validation/visualizer.py`**

```python
# validation/visualizer.py
"""
Run:  python validation/visualizer.py --date 2024-03-01 --days 2
Plots a 2-day M5 candlestick chart with OB zones, BOS arrows, FVG fills.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf

from src.loader import load_mt5_csv
from src.smc.structure import detect_fractals, detect_bos_choch
from src.smc.fvg import detect_fvg
from src.smc.order_blocks import detect_order_blocks
from src.smc.liquidity import detect_asia_session


def run_pipeline(m5_path: str) -> pd.DataFrame:
    df = load_mt5_csv(m5_path)
    df = detect_fractals(df, n=2)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = detect_asia_session(df)
    return df


def plot_validation(df: pd.DataFrame, start: str, days: int = 2, title: str = "SMC Validation"):
    end = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=days)
    window = df.loc[start:end].copy()

    if len(window) == 0:
        print(f"No data found for {start} + {days} days.")
        return

    # mplfinance expects specific column names
    ohlcv = window[["open", "high", "low", "close", "volume"]].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlcv.index.name = "Date"

    # --- Build addplot overlays ---
    ap_list = []

    # Swing highs / lows as scatter markers
    sh_prices = np.where(window["swing_high"], window["high"] + 0.5, np.nan)
    sl_prices = np.where(window["swing_low"],  window["low"]  - 0.5, np.nan)
    ap_list.append(mpf.make_addplot(sh_prices, type="scatter", markersize=60,
                                    marker="v", color="red"))
    ap_list.append(mpf.make_addplot(sl_prices, type="scatter", markersize=60,
                                    marker="^", color="lime"))

    # BOS bullish: green dot on close
    bos_bull = np.where(window["bos_bullish"], window["close"], np.nan)
    bos_bear = np.where(window["bos_bearish"], window["close"], np.nan)
    if not np.all(np.isnan(bos_bull)):
        ap_list.append(mpf.make_addplot(bos_bull, type="scatter", markersize=80,
                                        marker="*", color="cyan"))
    if not np.all(np.isnan(bos_bear)):
        ap_list.append(mpf.make_addplot(bos_bear, type="scatter", markersize=80,
                                        marker="*", color="orange"))

    # Asia session range as horizontal lines
    asia_high_line = window["asia_high"].copy()
    asia_low_line  = window["asia_low"].copy()
    if asia_high_line.notna().any():
        ap_list.append(mpf.make_addplot(asia_high_line, color="purple",
                                        linestyle="--", width=0.8))
        ap_list.append(mpf.make_addplot(asia_low_line,  color="purple",
                                        linestyle="--", width=0.8))

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        style="charles",
        addplot=ap_list,
        title=title,
        volume=True,
        figsize=(18, 9),
        returnfig=True,
        warn_too_much_data=10000,
    )

    ax = axes[0]

    # --- Draw OB rectangles ---
    ob_mask = window["bull_ob"] | window["bear_ob"]
    ob_bars  = window[ob_mask]
    x_ticks  = {ts: i for i, ts in enumerate(window.index)}

    for ts, row in ob_bars.iterrows():
        x = x_ticks[ts]
        color = "#00cc44" if row["bull_ob"] else "#cc2200"
        rect = mpatches.FancyBboxPatch(
            (x - 0.4, row["ob_bottom"]),
            width=0.8,
            height=row["ob_top"] - row["ob_bottom"],
            boxstyle="square,pad=0",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.25,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(x, row["ob_midpoint"], "OB", fontsize=6, color=color,
                ha="center", va="center")

    # --- Draw FVG fills ---
    for i, (ts, row) in enumerate(window.iterrows()):
        if row["fvg_bull"] and row["fvg_bull_size"] > 0:
            x = x_ticks[ts]
            fvg_low  = window["high"].shift(2).loc[ts]
            fvg_high = row["low"]
            ax.axhspan(fvg_low, fvg_high, xmin=(x-1)/(len(window)),
                       xmax=(x+1)/(len(window)), alpha=0.12, color="blue")
        if row["fvg_bear"] and row["fvg_bear_size"] > 0:
            x = x_ticks[ts]
            fvg_high = window["low"].shift(2).loc[ts]
            fvg_low  = row["high"]
            ax.axhspan(fvg_low, fvg_high, xmin=(x-1)/(len(window)),
                       xmax=(x+1)/(len(window)), alpha=0.12, color="red")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#00cc44", alpha=0.5, label="Bullish OB"),
        mpatches.Patch(color="#cc2200", alpha=0.5, label="Bearish OB"),
        mpatches.Patch(color="cyan",   alpha=0.8, label="BOS Bullish (*)"),
        mpatches.Patch(color="orange", alpha=0.8, label="BOS Bearish (*)"),
        mpatches.Patch(color="purple", alpha=0.8, label="Asia Range"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = Path("validation") / f"smc_validation_{start[:10]}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",  default="2024-03-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--days",  type=int, default=2,  help="Number of days to plot")
    parser.add_argument("--file",  default="xauusd_m5_history.csv")
    args = parser.parse_args()

    df = run_pipeline(args.file)
    plot_validation(df, args.date, args.days,
                    title=f"XAUUSD M5 SMC — {args.date} ({args.days}d)")
```

- [ ] **Step 2: Run the validation script**

```bash
python validation/visualizer.py --date 2024-01-02 --days 2 --file xauusd_m5_history.csv
```
Expected: PNG saved to `validation/smc_validation_2024-01-02.png` and chart displayed.

- [ ] **Step 3: Commit**

```bash
git add validation/visualizer.py
git commit -m "feat: SMC validation chart (OBs, BOS, FVG, Asia range)"
```

---

## Task 12: Transformer Architecture Schema

**Files:**
- Create: `models/transformer.py`

- [ ] **Step 1: Implement `models/transformer.py`**

```python
# models/transformer.py
"""
XAUUSD Multi-Timeframe Transformer — GPT-style Decoder-only architecture.

Input:  Sequence of T=96 M5 bars, each with F=~40 features (normalized).
        Features already encode M15 and H1 context via alignment pipeline.
Output: 3-class softmax → [Long_OB, Short_OB, No_Trade]
        + regression head → predicted RR (reward:risk ratio)

Architecture choices:
  - Decoder-only (causal self-attention) → no data leakage across time
  - Rotary Position Embeddings (RoPE) → better extrapolation vs learned
  - 3 Transformer blocks → empirically sufficient for ~40 features / 96-bar window
  - Separate classification + regression heads sharing the same backbone
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEQ_LEN   = 96    # 8 hours of M5 bars (context window)
N_FEATURES = 42   # number of normalized input features (adjust after pipeline)
D_MODEL    = 128  # embedding dimension
N_HEADS    = 4    # attention heads (D_MODEL must be divisible by N_HEADS)
D_FF       = 256  # feed-forward hidden size
N_LAYERS   = 3    # Transformer blocks
DROPOUT    = 0.1
N_CLASSES  = 3    # Long, Short, No-Trade


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) — applied inside scaled dot-product attn
# ---------------------------------------------------------------------------

def _rope_freqs(seq_len: int, d_head: int) -> tf.Tensor:
    """Precompute RoPE rotation frequencies."""
    theta = 1.0 / (10000.0 ** (tf.cast(tf.range(0, d_head, 2), tf.float32) / d_head))
    positions = tf.cast(tf.range(seq_len), tf.float32)
    freqs = tf.einsum("i,j->ij", positions, theta)   # [seq_len, d_head//2]
    return tf.concat([freqs, freqs], axis=-1)          # [seq_len, d_head]


def _apply_rope(x: tf.Tensor, freqs: tf.Tensor) -> tf.Tensor:
    """Apply rotary embedding to query/key tensors [batch, heads, seq, d_head]."""
    cos = tf.cos(freqs)[tf.newaxis, tf.newaxis, :, :]  # [1, 1, seq, d_head]
    sin = tf.sin(freqs)[tf.newaxis, tf.newaxis, :, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = tf.concat([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# Causal Multi-Head Attention with RoPE
# ---------------------------------------------------------------------------

class CausalMHAWithRoPE(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.q_proj = layers.Dense(d_model, use_bias=False)
        self.k_proj = layers.Dense(d_model, use_bias=False)
        self.v_proj = layers.Dense(d_model, use_bias=False)
        self.out    = layers.Dense(d_model, use_bias=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        B, T, _ = tf.unstack(tf.shape(x))
        scale    = tf.math.sqrt(tf.cast(self.d_head, tf.float32))

        def _split(z):
            z = tf.reshape(z, [B, T, self.n_heads, self.d_head])
            return tf.transpose(z, [0, 2, 1, 3])  # [B, H, T, d_head]

        Q = _apply_rope(_split(self.q_proj(x)), _rope_freqs(T, self.d_head))
        K = _apply_rope(_split(self.k_proj(x)), _rope_freqs(T, self.d_head))
        V = _split(self.v_proj(x))

        # Causal mask
        mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.float32), -1, 0)
        mask = (1.0 - mask) * -1e9

        attn = tf.matmul(Q, K, transpose_b=True) / scale + mask
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, V)                          # [B, H, T, d_head]
        out = tf.transpose(out, [0, 2, 1, 3])             # [B, T, H, d_head]
        out = tf.reshape(out, [B, T, self.d_model])
        return self.out(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.attn    = CausalMHAWithRoPE(d_model, n_heads)
        self.ffn     = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        self.norm2   = layers.LayerNormalization(epsilon=1e-6)
        self.drop1   = layers.Dropout(dropout)
        self.drop2   = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Pre-norm (more stable than post-norm for financial data)
        x = x + self.drop1(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop2(self.ffn(self.norm2(x)),                      training=training)
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

def build_gold_transformer(
    seq_len:    int = SEQ_LEN,
    n_features: int = N_FEATURES,
    d_model:    int = D_MODEL,
    n_heads:    int = N_HEADS,
    d_ff:       int = D_FF,
    n_layers:   int = N_LAYERS,
    dropout:    float = DROPOUT,
    n_classes:  int = N_CLASSES,
) -> keras.Model:
    """
    Returns a compiled Keras model for XAUUSD directional classification.

    Input shape:  (batch, seq_len, n_features)
    Outputs:
        direction_logits  : (batch, n_classes)   — Long / Short / No-Trade
        rr_prediction     : (batch, 1)            — predicted reward:risk
    """
    inp = keras.Input(shape=(seq_len, n_features), name="m5_sequence")

    # Feature projection into model dimension
    x = layers.Dense(d_model, name="feature_proj")(inp)
    x = layers.LayerNormalization(epsilon=1e-6, name="input_norm")(x)

    # Stack Transformer blocks
    for i in range(n_layers):
        x = TransformerBlock(d_model, n_heads, d_ff, dropout, name=f"block_{i}")(x)

    # Use only the LAST token representation for prediction (GPT-style)
    last = x[:, -1, :]  # [batch, d_model]

    # Classification head
    direction = layers.Dense(d_model // 2, activation="gelu", name="cls_hidden")(last)
    direction = layers.Dropout(dropout)(direction)
    direction = layers.Dense(n_classes, name="direction_logits")(direction)
    direction = layers.Softmax(name="direction_probs")(direction)

    # Regression head (RR prediction)
    rr = layers.Dense(d_model // 4, activation="gelu", name="rr_hidden")(last)
    rr = layers.Dense(1, activation="relu", name="rr_prediction")(rr)

    model = keras.Model(inputs=inp, outputs={"direction": direction, "rr": rr},
                        name="GoldTransformer")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
        loss={
            "direction": keras.losses.SparseCategoricalCrossentropy(),
            "rr":        keras.losses.Huber(delta=0.5),
        },
        loss_weights={"direction": 1.0, "rr": 0.3},
        metrics={"direction": ["accuracy"], "rr": ["mae"]},
    )

    return model


if __name__ == "__main__":
    model = build_gold_transformer()
    model.summary()
    print("\nInput:  (batch, 96, 42)  →  96 M5 bars × 42 features")
    print("Output: direction probs (3-class) + RR regression (1 value)")
```

- [ ] **Step 2: Print model summary**

```bash
python models/transformer.py
```
Expected: Model summary printed showing ~1.5M parameters, no errors.

- [ ] **Step 3: Commit**

```bash
git add models/transformer.py
git commit -m "feat: GPT-style Transformer (causal MHA + RoPE) for XAUUSD direction + RR prediction"
```

---

## Self-Review vs Spec

| Spec Requirement | Covered In |
|---|---|
| OB Filter 1: caused BOS/CHoCH | Task 4 — `detect_order_blocks` Filter 1+4 window check |
| OB Filter 2: consecutive FVG | Task 4 — `fvg_bull_arr[i+1]` check |
| OB Filter 3: liquidity sweep of prior H/L | Task 4 — `swept_low` / `swept_high` check |
| OB Filter 4: vol Z-Score > 1.5 on break bar | Task 4 — `vol_z[k] > VOL_ZSCORE_MIN` |
| Equal Highs/Lows | Task 5 — `detect_equal_highs_lows` |
| Asia session + Judas Swings | Task 5 — `detect_asia_session` |
| Fractal swing structure, trend | Task 2 — `detect_fractals`, `detect_bos_choch` |
| Entry at OB midpoint or open | Task 8 — `compute_risk_params` → `entry = midpoint` |
| SL: 2 pips + ATR buffer below OB | Task 8 — `sl_buffer = PIPS_SL*PIP_SIZE + atr*ATR_MULT` |
| TP: next unmitigated H1 OB or RR 1:2.5 | Task 8 — `tp_target` logic |
| DD filter: SL < 0.5% balance | Task 8 — `dd_filter_ok` |
| M5/M15/H1 alignment, no look-ahead | Task 7 — `_shift_to_close_time` + `merge_asof` |
| M5 features: OHLCV, RSI-Lagless, OB_Freshness, FVG_Gap | Tasks 6, 9 |
| M15 features: Trend_ID, supply/demand zones | Tasks 2–4, 10 via alignment |
| H1 features: PD Arrays, Sessions | Tasks 5, 9 (`prev_day_high/low`) |
| RobustScaler normalization | Task 10 — `normalizer.py` |
| Output: .parquet (primary) + .tfrecord | Task 10 — `dataset.py` |
| GoldQuantProcessor class | Task 9 — `processor.py` |
| Validation chart | Task 11 — `visualizer.py` |
| Transformer schema | Task 12 — `models/transformer.py` |

No gaps found.
