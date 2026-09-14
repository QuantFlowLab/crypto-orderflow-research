"""test_flow_signs.py — R1: freeze aggressor sign convention.
BUY taker consumes ASK (up); SELL taker consumes BID (down). Errors here are catastrophic."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import aggressive_flow as af


def _exec(ts, taker_dir, qty, price, usd):
    return {"timestamp": ts, "event": {"Execution": {"execution": {
        "takerOrder": {"uid": f"t{ts}", "direction": taker_dir},
        "makerOrder": {"uid": f"m{ts}", "direction": "Sell" if taker_dir == "Buy" else "Buy"},
        "quantity": str(qty), "price": str(price), "usdValue": str(usd)}}}}


def test_buy_taker_is_positive_consumes_ask():
    atoms = af.build_taker_atoms([_exec(1000, "Buy", 2.0, 100.0, 200.0)])
    assert atoms.iloc[0]["aggressor"] == "buy_taker"
    assert atoms.iloc[0]["sign"] == 1
    f = af.windowed_aggressive_flow(atoms, np.array([1000]), {"1s": 1000})
    assert f.iloc[0]["buy_taker_qty_1s"] == 2.0
    assert f.iloc[0]["sell_taker_qty_1s"] == 0.0
    assert f.iloc[0]["afi_1s"] > 0.99          # net upward pressure
    assert f.iloc[0]["ask_levels_consumed_1s"] == 1  # buy taker eats ASK level


def test_sell_taker_is_negative_consumes_bid():
    atoms = af.build_taker_atoms([_exec(1000, "Sell", 3.0, 100.0, 300.0)])
    assert atoms.iloc[0]["aggressor"] == "sell_taker"
    assert atoms.iloc[0]["sign"] == -1
    f = af.windowed_aggressive_flow(atoms, np.array([1000]), {"1s": 1000})
    assert f.iloc[0]["sell_taker_qty_1s"] == 3.0
    assert f.iloc[0]["afi_1s"] < -0.99
    assert f.iloc[0]["bid_levels_consumed_1s"] == 1  # sell taker eats BID level


def test_afi_balanced_is_zero():
    atoms = af.build_taker_atoms([_exec(1000, "Buy", 1.0, 100.0, 100.0),
                                  _exec(1001, "Sell", 1.0, 100.0, 100.0)])
    f = af.windowed_aggressive_flow(atoms, np.array([2000]), {"5s": 5000})
    assert abs(f.iloc[0]["afi_5s"]) < 1e-6


def test_windows_are_causal_backward():
    # atom at t=5000 must NOT appear in a window ending at t=4000
    atoms = af.build_taker_atoms([_exec(5000, "Buy", 9.0, 100.0, 900.0)])
    f = af.windowed_aggressive_flow(atoms, np.array([4000]), {"1s": 1000})
    assert f.iloc[0]["buy_taker_qty_1s"] == 0.0
    # and appears at an anchor at/after it
    f2 = af.windowed_aggressive_flow(atoms, np.array([5000]), {"1s": 1000})
    assert f2.iloc[0]["buy_taker_qty_1s"] == 9.0


def test_levels_consumed_distinct_prices():
    atoms = af.build_taker_atoms([_exec(1000, "Buy", 1.0, 100.0, 100.0),
                                  _exec(1000, "Buy", 1.0, 101.0, 101.0),
                                  _exec(1000, "Buy", 1.0, 101.0, 101.0)])
    f = af.windowed_aggressive_flow(atoms, np.array([1000]), {"1s": 1000})
    assert f.iloc[0]["ask_levels_consumed_1s"] == 2  # prices 100 & 101
