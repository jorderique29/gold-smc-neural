# src/risk.py
"""
Risk parameter computation for Order Block-based XAUUSD setups.

XAUUSD specs (ADN Broker ADNX10):
  1 pip  = $0.10 price movement
  1 lot  = 100 oz  →  1 pip/lot = $10 USD profit/loss
  DD limit per trade: SL dollar_risk < 0.5% of account balance
"""

import math

# ── Constants ─────────────────────────────────────────────────────────────────
PIP_SIZE  = 0.10     # price distance of 1 pip on XAUUSD
PIPS_SL   = 2.0      # fixed pip buffer beyond OB edge for SL
ATR_MULT  = 0.25     # fraction of ATR added to SL buffer
MIN_RR    = 2.5      # minimum reward:risk ratio for TP
LOT_SIZE  = 100.0    # oz per standard lot (XAUUSD)
BALANCE   = 10_000.0 # assumed account balance (USD)
DD_LIMIT  = 0.005    # 0.5% of balance = max dollar risk per trade


def compute_risk_params(row: dict) -> dict:
    """
    Compute entry, SL, TP and DD filter flag for one OB-based setup.

    Args:
        row: dict with keys:
            ob_type         (int)   : 1 = bullish OB, -1 = bearish OB
            ob_top          (float) : OB high price
            ob_bottom       (float) : OB low price
            ob_midpoint     (float) : (ob_top + ob_bottom) / 2
            atr             (float) : current ATR value
            next_unmit_ob_h1(float) : price of next unmitigated H1 OB (0 or NaN if none)

    Returns:
        dict with keys:
            entry       (float) : trade entry price
            sl          (float) : stop-loss price
            tp          (float) : take-profit price
            risk_price  (float) : |entry - sl| in price units
            rr_achieved (float) : |tp - entry| / risk_price
            dd_filter_ok(bool)  : True if dollar_risk < 0.5% of balance
    """
    ob_type   = int(row["ob_type"])
    ob_top    = float(row["ob_top"])
    ob_bottom = float(row["ob_bottom"])
    midpoint  = float(row["ob_midpoint"])
    atr       = float(row["atr"])

    # Safely handle NaN / 0 for next unmitigated OB
    next_ob_raw = row.get("next_unmit_ob_h1", 0) or 0
    try:
        next_ob = float(next_ob_raw)
    except (TypeError, ValueError):
        next_ob = 0.0
    if math.isnan(next_ob):
        next_ob = 0.0

    sl_buffer = PIPS_SL * PIP_SIZE + atr * ATR_MULT  # e.g. 0.20 + atr*0.25

    if ob_type == 1:  # ── Bullish OB → Long trade ─────────────────────────
        entry = midpoint
        sl    = ob_bottom - sl_buffer
        risk  = entry - sl
        if risk <= 0:
            risk = sl_buffer  # safety floor (should not happen in practice)

        rr_tp  = entry + MIN_RR * risk
        # Use next H1 OB if it's further than the RR-based TP
        tp = next_ob if (next_ob > rr_tp) else rr_tp

    else:             # ── Bearish OB → Short trade ────────────────────────
        entry = midpoint
        sl    = ob_top + sl_buffer
        risk  = sl - entry
        if risk <= 0:
            risk = sl_buffer

        rr_tp = entry - MIN_RR * risk
        # Use next H1 OB if it's further (lower) than the RR-based TP
        tp = next_ob if (0 < next_ob < rr_tp) else rr_tp

    rr_achieved  = abs(tp - entry) / risk if risk > 0 else 0.0
    dollar_risk  = risk * LOT_SIZE
    dd_filter_ok = bool(dollar_risk < DD_LIMIT * BALANCE)

    return {
        "entry":        round(entry, 2),
        "sl":           round(sl,    2),
        "tp":           round(tp,    2),
        "risk_price":   round(risk,  4),
        "rr_achieved":  round(rr_achieved, 4),
        "dd_filter_ok": dd_filter_ok,
    }
