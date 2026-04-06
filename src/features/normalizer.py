# src/features/normalizer.py
"""
Feature normalization for Transformer input.

Uses RobustScaler (median + IQR) for continuous features — preferred over
StandardScaler for financial data with outliers (fat tails, volatility spikes).

Feature groups:
  PRICE_FEATURES      → absolute price columns — scaled together
  INDICATOR_FEATURES  → RSI, ATR, z-scores — scaled together
  PASSTHROUGH_FEATURES → binary / categorical — NOT scaled
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ── Feature group definitions ─────────────────────────────────────────────────

PRICE_FEATURES = [
    "open", "high", "low", "close",
    "ob_top", "ob_bottom", "ob_midpoint",
    "asia_high", "asia_low",
    "h1_close", "h1_high", "h1_low", "h1_ob_midpoint",
    "h1_asia_high", "h1_asia_low",
    "m15_close", "m15_high", "m15_low", "m15_ob_midpoint",
    "entry", "sl", "tp",
]

INDICATOR_FEATURES = [
    "volume", "rsi_lagless", "atr", "vol_zscore",
    "fvg_bull_size", "fvg_bear_size",
    "ob_freshness", "risk_price", "rr_achieved",
]

PASSTHROUGH_FEATURES = [
    # Trend / structure flags
    "trend", "ob_type",
    "bull_ob", "bear_ob",
    "bos_bullish", "bos_bearish",
    "choch_bullish", "choch_bearish",
    "fvg_bull", "fvg_bear",
    "equal_high_pool", "equal_low_pool",
    "in_asia_session", "judas_swing_bull", "judas_swing_bear",
    "dd_filter_ok",
    # M15 / H1 structure flags
    "h1_trend", "h1_bull_ob", "h1_bear_ob",
    "h1_bos_bullish", "h1_bos_bearish",
    "m15_trend", "m15_bull_ob", "m15_bear_ob",
    "m15_bos_bullish", "m15_bos_bearish",
    "m15_fvg_bull", "m15_fvg_bear",
]


def normalize_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize continuous features using RobustScaler.
    Passthrough (binary / categorical) columns are left unchanged.

    NaN values in continuous columns are imputed with the column median
    before scaling (prevents scaler failure; NaN pattern is preserved
    in passthrough columns).

    Args:
        df: Aligned DataFrame from GoldQuantProcessor.align_timeframes()
            with risk columns added.

    Returns:
        (normalized_df, scalers) where:
            normalized_df : copy of df with PRICE and INDICATOR columns scaled
            scalers       : dict {"price": RobustScaler, "indicators": RobustScaler}
    """
    df = df.copy()
    scalers: dict = {}

    for group_name, cols in [
        ("price",      PRICE_FEATURES),
        ("indicators", INDICATOR_FEATURES),
    ]:
        available = [c for c in cols if c in df.columns]
        if not available:
            continue

        data = df[available].to_numpy(dtype=float)

        # Impute NaN with column median (no look-ahead: median of full dataset is OK
        # for training data; inference should use the fitted scaler's transform)
        col_medians = np.nanmedian(data, axis=0)
        for j in range(data.shape[1]):
            nan_mask = np.isnan(data[:, j])
            if nan_mask.any():
                data[nan_mask, j] = col_medians[j]

        scaler = RobustScaler()
        df[available] = scaler.fit_transform(data)
        scalers[group_name] = scaler

    return df, scalers
