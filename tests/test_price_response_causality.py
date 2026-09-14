"""test_price_response_causality.py — R1 Part D: price response must use ONLY the same
historical interval (t-W, t] — never a forward outcome."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import price_response as pr
import aggressive_flow as af


def _book(ts_mid):
    return pd.DataFrame({"ts": [t for t, _ in ts_mid], "mid": [m for _, m in ts_mid]})


def _exec(ts, direction, qty, price):
    return {"timestamp": ts, "event": {"Execution": {"execution": {
        "takerOrder": {"uid": f"t{ts}", "direction": direction},
        "makerOrder": {"uid": f"m{ts}", "direction": "Sell" if direction == "Buy" else "Buy"},
        "quantity": str(qty), "price": str(price), "usdValue": str(qty * price)}}}}


def test_signed_mid_change_is_backward():
    # mid: 100 @1000, 105 @2000. Window 1s at anchor 2000 -> change = mid(2000)-mid(1000)=+5
    book = _book([(1000, 100.0), (2000, 105.0)])
    f = pr.build_price_response(book, af.build_taker_atoms([]), np.array([2000]),
                               tick=1.0, windows={"1s": 1000})
    assert f.iloc[0]["signed_mid_change_1s"] == 5.0
    assert f.iloc[0]["ticks_moved_1s"] == 5.0
    assert f.iloc[0]["levels_crossed_1s"] == 5.0


def test_future_mid_not_used():
    # a mid observation AFTER the anchor must not affect the response at the anchor
    book = _book([(1000, 100.0), (2000, 105.0), (3000, 999.0)])
    f = pr.build_price_response(book, af.build_taker_atoms([]), np.array([2000]),
                               tick=1.0, windows={"1s": 1000})
    assert f.iloc[0]["signed_mid_change_1s"] == 5.0  # 3000/999 ignored


def test_response_per_qty_sign_matches_flow():
    # buy takers present while mid rose -> price_response_per_buy_qty positive
    book = _book([(1000, 100.0), (2000, 102.0)])
    atoms = af.build_taker_atoms([_exec(1500, "Buy", 4.0, 101.0)])
    f = pr.build_price_response(book, atoms, np.array([2000]), tick=1.0, windows={"1s": 1000})
    assert f.iloc[0]["signed_mid_change_1s"] == 2.0
    assert f.iloc[0]["price_response_per_buy_qty_1s"] > 0
    assert abs(f.iloc[0]["price_response_per_buy_qty_1s"] - 2.0 / 4.0) < 1e-6


def test_missing_mid_is_nan():
    book = _book([(5000, 100.0)])  # no obs at/before anchor-W or anchor
    f = pr.build_price_response(book, af.build_taker_atoms([]), np.array([2000]),
                               tick=1.0, windows={"1s": 1000})
    assert np.isnan(f.iloc[0]["signed_mid_change_1s"])


def test_eth_tick_scaling():
    book = _book([(1000, 100.0), (2000, 100.5)])  # +0.5 with ETH tick 0.1 -> 5 ticks
    f = pr.build_price_response(book, af.build_taker_atoms([]), np.array([2000]),
                               tick=0.1, windows={"1s": 1000})
    assert abs(f.iloc[0]["ticks_moved_1s"] - 5.0) < 1e-9
    assert f.iloc[0]["levels_crossed_1s"] == 5.0
