# src/processor.py
"""
GoldQuantProcessor — main orchestrator for the XAUUSD SMC feature pipeline.

Three-stage usage:
    proc = GoldQuantProcessor("xauusd_m5_history.csv",
                              "xauusd_m15_history.csv",
                              "xauusd_h1_history.csv")
    proc.detect_smc()
    proc.align_timeframes()
    proc.generate_training_tensors(output_dir="data/processed")
"""
from __future__ import annotations

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
from src.alignment import align_timeframes as _align_tf
from src.risk import compute_risk_params


class GoldQuantProcessor:
    """
    Orchestrates the full SMC feature extraction pipeline for XAUUSD.

    Attributes:
        m5, m15, h1  : processed DataFrames per timeframe (set by detect_smc)
        aligned       : M5 DataFrame enriched with M15/H1 features (set by align_timeframes)
    """

    def __init__(
        self,
        m5_file:   str,
        m15_file:  str,
        h1_file:   str,
        fractal_n: int = 2,
    ) -> None:
        self.m5_file   = m5_file
        self.m15_file  = m15_file
        self.h1_file   = h1_file
        self.fractal_n = fractal_n

        self.m5:      Optional[pd.DataFrame] = None
        self.m15:     Optional[pd.DataFrame] = None
        self.h1:      Optional[pd.DataFrame] = None
        self.aligned: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    def detect_smc(self) -> None:
        """
        Load the three CSV files and run the full SMC detection pipeline
        on each timeframe independently.

        Populates: self.m5, self.m15, self.h1
        """
        print("[GoldQuantProcessor] Loading CSVs ...")
        raw_m5  = load_mt5_csv(self.m5_file)
        raw_m15 = load_mt5_csv(self.m15_file)
        raw_h1  = load_mt5_csv(self.h1_file)
        print(f"  M5: {len(raw_m5):,}  M15: {len(raw_m15):,}  H1: {len(raw_h1):,}")

        print("[GoldQuantProcessor] Running SMC pipeline ...")
        self.m5  = self._smc_pipeline(raw_m5,  self.fractal_n)
        self.m15 = self._smc_pipeline(raw_m15, self.fractal_n)
        self.h1  = self._smc_pipeline_h1(raw_h1, self.fractal_n)
        print("[GoldQuantProcessor] detect_smc() complete.")

    # ──────────────────────────────────────────────────────────────────────────
    def align_timeframes(self) -> None:
        """
        Merge M15 and H1 SMC features into the M5 DataFrame without look-ahead bias.

        Requires: detect_smc() called first.
        Populates: self.aligned
        """
        if self.m5 is None or self.m15 is None or self.h1 is None:
            raise RuntimeError(
                "Call detect_smc() before align_timeframes()."
            )

        print("[GoldQuantProcessor] Aligning timeframes ...")
        self.aligned = _align_tf(self.m5, self.m15, self.h1)
        print(f"  Aligned shape: {self.aligned.shape}")

    # ──────────────────────────────────────────────────────────────────────────
    def generate_training_tensors(
        self,
        output_dir: str | Path = "data/processed",
    ) -> Path:
        """
        Add risk columns to the aligned DataFrame and serialize to Parquet.

        Requires: align_timeframes() called first.

        Args:
            output_dir: directory where the .parquet file will be written.

        Returns:
            Path to the written Parquet file.
        """
        if self.aligned is None:
            raise RuntimeError(
                "Call align_timeframes() before generate_training_tensors()."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.aligned.copy()

        # Add risk parameters for OB rows
        df = self._add_risk_columns(df)

        # TODO (Task 10): apply RobustScaler normalization via src.features.normalizer
        # TODO (Task 10): use src.dataset.save_parquet for column selection + compression

        # Simplified serialization for now — write full DataFrame to Parquet
        out_path = output_dir / "gold_smc_dataset.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=True)
        print(f"[GoldQuantProcessor] Parquet saved -> {out_path}")

        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _smc_pipeline(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Run the full SMC feature pipeline for M5 or M15 data.

        Order of operations:
            fractals -> BOS/CHoCH -> FVG -> OB -> equal H/L -> Asia session
            -> RSI-Lagless -> ATR -> OB freshness
        """
        df = detect_fractals(df, n=n)
        df = detect_bos_choch(df)
        df = detect_fvg(df)
        df = detect_order_blocks(df)
        df = detect_equal_highs_lows(df)
        df = detect_asia_session(df)
        df = add_rsi_lagless(df)
        df = add_atr(df)
        df = _add_ob_freshness(df)
        return df

    @staticmethod
    def _smc_pipeline_h1(df: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        H1 pipeline: same as M5/M15 plus previous-day high/low (PD Arrays).
        """
        df = GoldQuantProcessor._smc_pipeline(df, n)
        # Previous-day High/Low: resample daily, shift 1 day forward
        daily_high = df["high"].resample("D").max()
        daily_low  = df["low"].resample("D").min()
        df["prev_day_high"] = daily_high.shift(1).reindex(df.index, method="ffill")
        df["prev_day_low"]  = daily_low.shift(1).reindex(df.index,  method="ffill")
        return df

    @staticmethod
    def _add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry/sl/tp/dd_filter_ok for each OB row using compute_risk_params().
        Non-OB rows get NaN for float columns and False for dd_filter_ok.
        """
        risk_float_cols = ["entry", "sl", "tp", "risk_price", "rr_achieved"]
        for col in risk_float_cols:
            df[col] = np.nan
        df["dd_filter_ok"] = False

        ob_mask = df["ob_type"] != 0
        for idx in df.index[ob_mask]:
            row_dict = df.loc[idx].to_dict()
            row_dict.setdefault("next_unmit_ob_h1", 0)
            try:
                params = compute_risk_params(row_dict)
                for col, val in params.items():
                    df.at[idx, col] = val
            except Exception:
                pass  # skip malformed rows silently

        return df


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helper (used by _smc_pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def _add_ob_freshness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'ob_freshness' column: number of bars since the most recent OB was formed.
    - 0 on the OB bar itself
    - Increments by 1 each subsequent bar
    - NaN until the first OB appears
    """
    df = df.copy()
    ob_any    = (df["bull_ob"] | df["bear_ob"]).to_numpy()
    freshness = np.full(len(df), np.nan)
    counter   = np.nan

    for i in range(len(df)):
        if ob_any[i]:
            counter = 0.0
        if not np.isnan(counter):
            freshness[i] = counter
            counter += 1.0

    df["ob_freshness"] = freshness
    return df
