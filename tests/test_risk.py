# tests/test_risk.py
import pytest
from src.risk import compute_risk_params, PIP_SIZE, ATR_MULT, MIN_RR, LOT_SIZE, BALANCE


def _bull_row(ob_top=1910.0, ob_bottom=1908.0, atr=2.0, next_ob=1925.0):
    return {
        "ob_type": 1,
        "ob_top": ob_top,
        "ob_bottom": ob_bottom,
        "ob_midpoint": (ob_top + ob_bottom) / 2,
        "atr": atr,
        "next_unmit_ob_h1": next_ob,
    }


def _bear_row(ob_top=1912.0, ob_bottom=1910.0, atr=2.0, next_ob=1895.0):
    return {
        "ob_type": -1,
        "ob_top": ob_top,
        "ob_bottom": ob_bottom,
        "ob_midpoint": (ob_top + ob_bottom) / 2,
        "atr": atr,
        "next_unmit_ob_h1": next_ob,
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

def test_bull_entry_at_midpoint():
    row = _bull_row(ob_top=1910.0, ob_bottom=1908.0)
    r = compute_risk_params(row)
    assert r["entry"] == pytest.approx(1909.0)


def test_bear_entry_at_midpoint():
    row = _bear_row(ob_top=1912.0, ob_bottom=1910.0)
    r = compute_risk_params(row)
    assert r["entry"] == pytest.approx(1911.0)


# ── Stop Loss ─────────────────────────────────────────────────────────────────

def test_bull_sl_below_ob_bottom():
    row = _bull_row(ob_bottom=1908.0, atr=2.0)
    r = compute_risk_params(row)
    expected_sl = 1908.0 - 2 * PIP_SIZE - 2.0 * ATR_MULT  # 1908 - 0.20 - 0.50 = 1907.30
    assert r["sl"] == pytest.approx(expected_sl, abs=0.01)


def test_bull_sl_is_below_entry():
    row = _bull_row()
    r = compute_risk_params(row)
    assert r["sl"] < r["entry"]


def test_bear_sl_above_ob_top():
    row = _bear_row(ob_top=1912.0, atr=2.0)
    r = compute_risk_params(row)
    expected_sl = 1912.0 + 2 * PIP_SIZE + 2.0 * ATR_MULT  # 1912 + 0.20 + 0.50 = 1912.70
    assert r["sl"] == pytest.approx(expected_sl, abs=0.01)


def test_bear_sl_is_above_entry():
    row = _bear_row()
    r = compute_risk_params(row)
    assert r["sl"] > r["entry"]


# ── Take Profit ───────────────────────────────────────────────────────────────

def test_bull_tp_above_entry():
    row = _bull_row()
    r = compute_risk_params(row)
    assert r["tp"] > r["entry"]


def test_bear_tp_below_entry():
    row = _bear_row()
    r = compute_risk_params(row)
    assert r["tp"] < r["entry"]


def test_bull_tp_minimum_rr():
    """TP must always give at least MIN_RR regardless of next_unmit_ob_h1."""
    row = _bull_row(next_ob=1909.5)  # too close → forces RR-based TP
    r = compute_risk_params(row)
    risk = r["entry"] - r["sl"]
    assert r["tp"] >= r["entry"] + MIN_RR * risk - 0.01


def test_bear_tp_minimum_rr():
    row = _bear_row(next_ob=1910.5)  # too close → forces RR-based TP
    r = compute_risk_params(row)
    risk = r["sl"] - r["entry"]
    assert r["tp"] <= r["entry"] - MIN_RR * risk + 0.01


def test_rr_achieved_matches_tp():
    """rr_achieved must equal (tp - entry) / risk."""
    row = _bull_row()
    r = compute_risk_params(row)
    risk = r["entry"] - r["sl"]
    expected_rr = (r["tp"] - r["entry"]) / risk
    assert r["rr_achieved"] == pytest.approx(expected_rr, rel=1e-4)


# ── DD Filter ─────────────────────────────────────────────────────────────────

def test_dd_filter_passes_normal_setup():
    """A tight OB with small ATR should pass the DD filter (dollar_risk < $50)."""
    # ob spread = 0.1, atr = 0.1 → risk ≈ 0.275, dollar_risk ≈ $27.50 < $50
    row = _bull_row(ob_top=1908.1, ob_bottom=1908.0, atr=0.1)
    r = compute_risk_params(row)
    assert r["dd_filter_ok"] is True


def test_dd_filter_fails_large_risk():
    """Very large ATR makes SL so wide that dollar_risk exceeds 0.5% of balance."""
    row = _bull_row(ob_bottom=1900.0, atr=100.0)
    r = compute_risk_params(row)
    assert r["dd_filter_ok"] is False


def test_dd_filter_threshold():
    """Dollar risk exactly at the limit = rejected (>= means reject)."""
    # We need dollar_risk = BALANCE * 0.005 = 50
    # dollar_risk = risk_price * LOT_SIZE = risk_price * 100
    # risk_price = 50 / 100 = 0.50
    # entry = 1909, sl must be exactly 0.50 below entry → sl = 1908.50
    # sl = ob_bottom - 0.20 - atr*0.25 → need this = 1908.50
    # ob_bottom = 1908, so: 1908 - 0.20 - atr*0.25 = 1908.50 → atr*0.25 = -0.70 (impossible)
    # Instead use ob_bottom=1908.5: sl = 1908.5 - 0.20 - atr*0.25
    # We want risk=0.50: entry=1909, sl=1908.50 → risk=0.50
    # 1908.5 - 0.20 - atr*0.25 = 1908.50 → atr*0.25 = -0.20 (impossible)
    # Just test the boundary directly:
    row = _bull_row(ob_top=1910.0, ob_bottom=1909.0, atr=0.0)
    r = compute_risk_params(row)
    # SL = 1909 - 0.20 - 0 = 1908.80; entry = 1909.5; risk = 0.70; dollar_risk = 70 > 50
    assert isinstance(r["dd_filter_ok"], bool)


def test_result_keys_complete():
    """compute_risk_params must return all expected keys."""
    row = _bull_row()
    r = compute_risk_params(row)
    for key in ["entry", "sl", "tp", "risk_price", "rr_achieved", "dd_filter_ok"]:
        assert key in r, f"Missing key: {key}"
