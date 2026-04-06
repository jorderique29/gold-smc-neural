# src/dataset.py
"""
Dataset serialization for the Gold SMC pipeline.

Saves the normalized feature DataFrame to:
  - Parquet (primary, snappy compressed) — used for Transformer training
  - TFRecord (optional) — for TensorFlow Dataset API pipelines
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# ── Feature columns saved to disk ─────────────────────────────────────────────
# Only these columns are written — raw price columns and intermediate
# computation columns are excluded to keep the dataset lean.

FEATURE_COLS = [
    # ── M5 OHLCV + core SMC ──────────────────────────────────────────────────
    "open", "high", "low", "close", "volume",
    "rsi_lagless", "atr", "vol_zscore",
    "fvg_bull_size", "fvg_bear_size",
    "ob_freshness", "ob_type", "ob_midpoint",
    "trend", "bos_bullish", "bos_bearish",
    "choch_bullish", "choch_bearish",
    "fvg_bull", "fvg_bear",
    "bull_ob", "bear_ob",
    "equal_high_pool", "equal_low_pool",
    "in_asia_session", "judas_swing_bull", "judas_swing_bear",
    "asia_high", "asia_low",
    # ── M15 features ─────────────────────────────────────────────────────────
    "m15_close", "m15_trend", "m15_ob_midpoint",
    "m15_bull_ob", "m15_bear_ob",
    "m15_bos_bullish", "m15_bos_bearish",
    "m15_fvg_bull", "m15_fvg_bear",
    # ── H1 features ──────────────────────────────────────────────────────────
    "h1_close", "h1_trend", "h1_ob_midpoint",
    "h1_bull_ob", "h1_bear_ob",
    "h1_bos_bullish", "h1_bos_bearish",
    "h1_asia_high", "h1_asia_low",
    # ── Risk / labels ─────────────────────────────────────────────────────────
    "entry", "sl", "tp", "rr_achieved", "dd_filter_ok", "risk_price",
]


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """
    Save the normalized DataFrame to Parquet.
    Only columns present in both df and FEATURE_COLS are written.

    Args:
        df:   Normalized DataFrame.
        path: Destination file path (parent dirs must exist or be created).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    out = df[available].copy()
    out.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
    return path


def save_tfrecord(df: pd.DataFrame, path: Path) -> Path:
    """
    Save the normalized DataFrame to TFRecord format (one example per row).

    Args:
        df:   Normalized DataFrame.
        path: Destination .tfrecord file path.

    Returns:
        Path to the written file.
    """
    import tensorflow as tf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available].fillna(0).astype("float32")

    def _float_feature(value: float) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    with tf.io.TFRecordWriter(str(path)) as writer:
        for _, row in data.iterrows():
            feature = {col: _float_feature(row[col]) for col in available}
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example.SerializeToString())

    return path
