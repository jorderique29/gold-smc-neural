# src/smc/order_blocks.py
"""
High-Probability Order Block Detection — 4 Filters.

An Order Block (OB) is the last directional candle before a strong BOS/CHoCH.

Bullish OB : last bearish candle (close < open) before a bullish BOS/CHoCH
Bearish OB : last bullish candle (close > open) before a bearish BOS/CHoCH

All 4 filters must be satisfied for an OB to be flagged:
  F1 (BOS/CHoCH)     : within LOOKAHEAD bars after the OB, a confirming
                       BOS or CHoCH must occur.
  F2 (FVG)           : bar immediately after the OB (i+1) must have
                       fvg_bull=True (bull OB) or fvg_bear=True (bear OB).
  F3 (Liquidity Sweep): the OB candle must have taken the prior bar's
                       low (bull OB) or high (bear OB).
  F4 (Volume)        : the confirming BOS/CHoCH bar must have
                       vol_zscore > VOL_ZSCORE_MIN.
"""
import numpy as np
import pandas as pd

LOOKAHEAD      = 20    # bars to look forward for a confirming BOS/CHoCH
VOL_ZSCORE_MIN = 1.5   # minimum volume z-score on the confirming break bar
VOL_WINDOW     = 50    # rolling window for volume z-score calculation

_REQUIRED = {"bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
             "fvg_bull", "fvg_bear", "volume"}


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect high-probability Order Blocks using all 4 SMC filters.

    Requires (from prior pipeline steps):
        detect_fractals()  → swing_high, swing_low
        detect_bos_choch() → bos_bullish, bos_bearish, choch_bullish, choch_bearish
        detect_fvg()       → fvg_bull, fvg_bear

    Adds columns:
        vol_zscore  (float) : rolling 50-bar volume z-score
        bull_ob     (bool)  : bar is a confirmed bullish OB
        bear_ob     (bool)  : bar is a confirmed bearish OB
        ob_top      (float) : OB high (NaN if not an OB)
        ob_bottom   (float) : OB low  (NaN if not an OB)
        ob_midpoint (float) : (ob_top + ob_bottom) / 2  (NaN if not an OB)
        ob_type     (int)   : 1 = bull OB, -1 = bear OB, 0 = none
    """
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"detect_order_blocks requires columns {_REQUIRED}. "
            f"Missing: {missing}. Run detect_fractals, detect_bos_choch, "
            f"detect_fvg first."
        )

    df = df.copy()

    # --- Volume Z-Score (rolling, no lookahead) ---
    vol_mean = df["volume"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
    vol_std  = (
        df["volume"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
        .replace(0, np.nan)
    )
    df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std

    # --- Extract numpy arrays for fast iteration ---
    n             = len(df)
    open_arr      = df["open"].to_numpy()
    close_arr     = df["close"].to_numpy()
    high_arr      = df["high"].to_numpy()
    low_arr       = df["low"].to_numpy()
    bos_bull      = df["bos_bullish"].to_numpy()
    bos_bear      = df["bos_bearish"].to_numpy()
    choch_bull    = df["choch_bullish"].to_numpy()
    choch_bear    = df["choch_bearish"].to_numpy()
    fvg_bull_arr  = df["fvg_bull"].to_numpy()
    fvg_bear_arr  = df["fvg_bear"].to_numpy()
    vol_z         = df["vol_zscore"].to_numpy()

    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)

    for i in range(1, n - LOOKAHEAD):

        # ── Bullish OB candidate: bearish candle ──────────────────────────
        if close_arr[i] < open_arr[i]:

            # F3: swept prior bar's low
            if low_arr[i] >= low_arr[i - 1]:
                continue

            # F2: FVG on the very next bar
            if i + 1 >= n or not fvg_bull_arr[i + 1]:
                continue

            # F1 + F4: confirming BOS/CHoCH with high volume in lookahead window
            end = min(i + 1 + LOOKAHEAD, n)
            confirmed = False
            for k in range(i + 1, end):
                if (bos_bull[k] or choch_bull[k]) and vol_z[k] > VOL_ZSCORE_MIN:
                    confirmed = True
                    break
            if confirmed:
                bull_ob[i] = True

        # ── Bearish OB candidate: bullish candle ──────────────────────────
        elif close_arr[i] > open_arr[i]:

            # F3: swept prior bar's high
            if high_arr[i] <= high_arr[i - 1]:
                continue

            # F2: FVG on the very next bar
            if i + 1 >= n or not fvg_bear_arr[i + 1]:
                continue

            # F1 + F4: confirming BOS/CHoCH with high volume in lookahead window
            end = min(i + 1 + LOOKAHEAD, n)
            confirmed = False
            for k in range(i + 1, end):
                if (bos_bear[k] or choch_bear[k]) and vol_z[k] > VOL_ZSCORE_MIN:
                    confirmed = True
                    break
            if confirmed:
                bear_ob[i] = True

    # --- Build output columns ---
    df["bull_ob"] = bull_ob
    df["bear_ob"] = bear_ob

    ob_high = np.where(bull_ob | bear_ob, df["high"].to_numpy(), np.nan)
    ob_low  = np.where(bull_ob | bear_ob, df["low"].to_numpy(),  np.nan)

    df["ob_top"]      = ob_high
    df["ob_bottom"]   = ob_low
    df["ob_midpoint"] = (ob_high + ob_low) / 2.0
    df["ob_type"]     = np.where(bull_ob, 1, np.where(bear_ob, -1, 0)).astype(int)

    return df
