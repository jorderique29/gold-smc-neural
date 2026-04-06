# validation/visualizer.py
"""
SMC Validation Chart for XAUUSD M5 data.

Usage:
    python validation/visualizer.py --date 2024-01-02 --days 2
    python validation/visualizer.py --date 2023-06-15 --days 3 --file xauusd_m5_history.csv

Generates a candlestick chart with SMC overlays:
    - Order Block rectangles (green = bullish, red = bearish)
    - BOS / CHoCH markers (cyan/orange stars)
    - Swing High/Low triangles
    - FVG shading (blue = bullish, red = bearish)
    - Asia session range (purple dashed lines)
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False

from src.loader import load_mt5_csv
from src.smc.structure import detect_fractals, detect_bos_choch
from src.smc.fvg import detect_fvg
from src.smc.order_blocks import detect_order_blocks
from src.smc.liquidity import detect_asia_session


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(csv_file: str) -> pd.DataFrame:
    """Run the SMC feature pipeline on M5 data."""
    print(f"[visualizer] Loading {csv_file}...")
    df = load_mt5_csv(csv_file)
    print(f"[visualizer] Running SMC pipeline on {len(df):,} bars...")
    df = detect_fractals(df, n=2)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = detect_asia_session(df)
    print("[visualizer] Pipeline complete.")
    return df


# ── Chart ─────────────────────────────────────────────────────────────────────

def plot_validation(
    df: pd.DataFrame,
    start: str,
    days: int = 2,
    title: str = "XAUUSD M5 — SMC Validation",
    save_dir: Path = Path("validation"),
) -> Path:
    """
    Plot a candlestick chart with SMC overlays for the given date window.

    Returns path to the saved PNG.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = start_ts + pd.Timedelta(days=days)
    window   = df.loc[start_ts:end_ts].copy()

    if len(window) == 0:
        print(f"[visualizer] No data found for {start} + {days} days.")
        print(f"  Available range: {df.index[0]} to {df.index[-1]}")
        return None

    print(f"[visualizer] Plotting {len(window)} bars from {window.index[0]} to {window.index[-1]}")

    # ── Build figure with mplfinance ──────────────────────────────────────────
    ohlcv = window[["open", "high", "low", "close", "volume"]].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
    ohlcv.index.name = "Date"

    addplots = []

    # Swing High markers (red downward triangle above high)
    sh = np.where(window["swing_high"], window["high"] + 0.8, np.nan)
    sl = np.where(window["swing_low"],  window["low"]  - 0.8, np.nan)
    if not np.all(np.isnan(sh)):
        addplots.append(mpf.make_addplot(sh, type="scatter", markersize=50,
                                          marker="v", color="red"))
    if not np.all(np.isnan(sl)):
        addplots.append(mpf.make_addplot(sl, type="scatter", markersize=50,
                                          marker="^", color="lime"))

    # BOS / CHoCH markers
    bos_bull = np.where(window["bos_bullish"],  window["close"], np.nan)
    bos_bear = np.where(window["bos_bearish"],  window["close"], np.nan)
    if not np.all(np.isnan(bos_bull)):
        addplots.append(mpf.make_addplot(bos_bull, type="scatter", markersize=100,
                                          marker="*", color="cyan"))
    if not np.all(np.isnan(bos_bear)):
        addplots.append(mpf.make_addplot(bos_bear, type="scatter", markersize=100,
                                          marker="*", color="orange"))

    # Asia range lines
    ah = window["asia_high"].copy()
    al = window["asia_low"].copy()
    if ah.notna().any():
        addplots.append(mpf.make_addplot(ah, color="purple", linestyle="--",
                                          width=0.8, alpha=0.7))
        addplots.append(mpf.make_addplot(al, color="purple", linestyle="--",
                                          width=0.8, alpha=0.7))

    fig, axes = mpf.plot(
        ohlcv,
        type="candle",
        style="charles",
        addplot=addplots if addplots else None,
        title=f"\n{title}",
        volume=True,
        figsize=(20, 10),
        returnfig=True,
        warn_too_much_data=5000,
    )

    ax = axes[0]

    # ── OB rectangles ─────────────────────────────────────────────────────────
    x_map = {ts: i for i, ts in enumerate(window.index)}
    ob_mask = window["bull_ob"] | window["bear_ob"]

    for ts, row in window[ob_mask].iterrows():
        x   = x_map[ts]
        col = "#00cc44" if row["bull_ob"] else "#cc2200"
        rect = mpatches.Rectangle(
            (x - 0.4, row["ob_bottom"]),
            width=0.8,
            height=row["ob_top"] - row["ob_bottom"],
            linewidth=1.5,
            edgecolor=col,
            facecolor=col,
            alpha=0.30,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(x, row["ob_midpoint"], "OB", fontsize=6,
                color=col, ha="center", va="center", fontweight="bold")

    # ── FVG shading ───────────────────────────────────────────────────────────
    high_s2 = window["high"].shift(2)
    low_s2  = window["low"].shift(2)
    n_bars  = len(window)

    for i, (ts, row) in enumerate(window.iterrows()):
        x = x_map[ts]
        xmin = max(0, (x - 1) / n_bars)
        xmax = min(1, (x + 1) / n_bars)

        if row["fvg_bull"] and row["fvg_bull_size"] > 0:
            fvg_bot = high_s2.loc[ts] if not pd.isna(high_s2.loc[ts]) else row["low"]
            fvg_top = row["low"]
            if fvg_top > fvg_bot:
                ax.axhspan(fvg_bot, fvg_top, xmin=xmin, xmax=xmax,
                           alpha=0.12, color="dodgerblue")

        if row["fvg_bear"] and row["fvg_bear_size"] > 0:
            fvg_top = low_s2.loc[ts] if not pd.isna(low_s2.loc[ts]) else row["high"]
            fvg_bot = row["high"]
            if fvg_top > fvg_bot:
                ax.axhspan(fvg_bot, fvg_top, xmin=xmin, xmax=xmax,
                           alpha=0.12, color="tomato")

    # ── Statistics panel ──────────────────────────────────────────────────────
    n_bull_ob  = window["bull_ob"].sum()
    n_bear_ob  = window["bear_ob"].sum()
    n_bos_bull = window["bos_bullish"].sum()
    n_bos_bear = window["bos_bearish"].sum()
    n_fvg_bull = window["fvg_bull"].sum()
    n_fvg_bear = window["fvg_bear"].sum()
    stats_text = (
        f"Bull OB: {n_bull_ob}  |  Bear OB: {n_bear_ob}  |  "
        f"BOS Bull: {n_bos_bull}  |  BOS Bear: {n_bos_bear}  |  "
        f"FVG Bull: {n_fvg_bull}  |  FVG Bear: {n_fvg_bear}"
    )
    ax.set_xlabel(stats_text, fontsize=8, color="gray")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#00cc44", alpha=0.5, label="Bullish OB"),
        mpatches.Patch(color="#cc2200", alpha=0.5, label="Bearish OB"),
        mpatches.Patch(color="cyan",    alpha=0.9, label="BOS Bull (*)"),
        mpatches.Patch(color="orange",  alpha=0.9, label="BOS Bear (*)"),
        mpatches.Patch(color="dodgerblue", alpha=0.4, label="FVG Bull"),
        mpatches.Patch(color="tomato",  alpha=0.4, label="FVG Bear"),
        mpatches.Patch(color="purple",  alpha=0.7, label="Asia Range"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, framealpha=0.8)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"smc_validation_{start[:10]}_{days}d.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] Chart saved -> {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot SMC validation chart for XAUUSD M5 data"
    )
    parser.add_argument("--date",  default="2024-01-02",
                        help="Start date YYYY-MM-DD (default: 2024-01-02)")
    parser.add_argument("--days",  type=int, default=2,
                        help="Number of days to plot (default: 2)")
    parser.add_argument("--file",  default="xauusd_m5_history.csv",
                        help="M5 CSV file (default: xauusd_m5_history.csv)")
    parser.add_argument("--out",   default="validation",
                        help="Output directory for PNG (default: validation/)")
    args = parser.parse_args()

    df = run_pipeline(args.file)
    out = plot_validation(
        df,
        start=args.date,
        days=args.days,
        title=f"XAUUSD M5 SMC Validation — {args.date} ({args.days}d)",
        save_dir=Path(args.out),
    )
    if out:
        print(f"[visualizer] Done! Open: {out.resolve()}")


if __name__ == "__main__":
    main()
