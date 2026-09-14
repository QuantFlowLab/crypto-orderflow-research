"""test_flow_accounting.py — R1: passive-flow side/action correctness & add/cancel conservation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import passive_flow as pf


def _placed(ts, direction, qty, price, otype="Limit", filled=0):
    return {"timestamp": ts, "event": {"OrderPlaced": {"order": {
        "uid": f"o{ts}", "direction": direction, "quantity": str(qty),
        "limitPrice": str(price), "orderType": otype, "filled": str(filled)}}}}


def _cancelled(ts, direction, qty, price, filled=0, uid=None):
    return {"timestamp": ts, "event": {"OrderCancelled": {"order": {
        "uid": uid or f"o{ts}", "direction": direction, "quantity": str(qty),
        "limitPrice": str(price), "orderType": "Limit", "filled": str(filled)}}}}


def _updated(ts, direction, old_qty, new_qty, price):
    return {"timestamp": ts, "event": {"OrderUpdated": {
        "oldOrder": {"uid": f"u{ts}", "direction": direction, "quantity": str(old_qty), "limitPrice": str(price)},
        "newOrder": {"uid": f"u{ts}", "direction": direction, "quantity": str(new_qty), "limitPrice": str(price),
                     "orderType": "Limit"}}}}


def test_buy_placed_is_bid_add():
    a = pf.build_passive_atoms([_placed(1000, "Buy", 5.0, 100.0)])
    assert a.iloc[0]["book_side"] == "bid"
    assert a.iloc[0]["action"] == "add"
    f = pf.windowed_passive_flow(a, np.array([1000]), {"1s": 1000})
    assert f.iloc[0]["bid_add_qty_1s"] == 5.0
    assert f.iloc[0]["ask_add_qty_1s"] == 0.0


def test_sell_cancel_is_ask_cancel():
    a = pf.build_passive_atoms([_cancelled(1000, "Sell", 4.0, 100.0)])
    assert a.iloc[0]["book_side"] == "ask"
    assert a.iloc[0]["action"] == "cancel"
    f = pf.windowed_passive_flow(a, np.array([1000]), {"1s": 1000})
    assert f.iloc[0]["ask_cancel_qty_1s"] == 4.0


def test_non_resting_placed_excluded():
    # IoC/Market orders never rest -> excluded from passive add
    a = pf.build_passive_atoms([_placed(1000, "Buy", 5.0, 100.0, otype="IoC")])
    assert len(a) == 0


def test_update_increase_is_add_decrease_is_cancel():
    a = pf.build_passive_atoms([_updated(1000, "Buy", 2.0, 5.0, 100.0),
                                _updated(1001, "Sell", 5.0, 1.0, 100.0)])
    add_row = a[a.action == "add"].iloc[0]
    cxl_row = a[a.action == "cancel"].iloc[0]
    assert add_row["book_side"] == "bid" and add_row["qty"] == 3.0
    assert cxl_row["book_side"] == "ask" and cxl_row["qty"] == 4.0


def test_place_then_cancel_conserves():
    # full place then full cancel on same side -> add_qty == cancel_qty over enclosing window
    a = pf.build_passive_atoms([_placed(1000, "Buy", 7.0, 100.0),
                                _cancelled(1500, "Buy", 7.0, 100.0)])
    f = pf.windowed_passive_flow(a, np.array([2000]), {"5s": 5000})
    assert f.iloc[0]["bid_add_qty_5s"] == 7.0
    assert f.iloc[0]["bid_cancel_qty_5s"] == 7.0
    assert f.iloc[0]["bid_net_passive_5s"] == 0.0


def test_passive_windows_causal():
    a = pf.build_passive_atoms([_placed(5000, "Buy", 9.0, 100.0)])
    f = pf.windowed_passive_flow(a, np.array([4000]), {"1s": 1000})
    assert f.iloc[0]["bid_add_qty_1s"] == 0.0
