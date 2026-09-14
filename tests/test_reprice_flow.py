"""test_reprice_flow.py — R2.2: semantic tests for REPRICE_OUT/IN atoms."""
import sys, json, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from passive_flow import build_reprice_atoms, windowed_reprice_flow

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_r2 import _stream_reprice_atoms


def _upd(ts, uid, direction, old_price, old_qty, new_price, new_qty):
    return {"timestamp": ts, "uid": f"u{uid}", "event": {"OrderUpdated": {
        "oldOrder": {"uid": uid, "direction": direction,
                     "limitPrice": str(old_price), "quantity": str(old_qty)},
        "newOrder": {"uid": uid, "direction": direction,
                     "limitPrice": str(new_price), "quantity": str(new_qty),
                     "orderType": "Limit"}}}}


def _run_both(events):
    canonical = build_reprice_atoms(events)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     encoding="utf-8", delete=False) as f:
        for e in events: f.write(json.dumps(e) + "\n")
        tmp = Path(f.name)
    streaming = _stream_reprice_atoms(tmp)
    tmp.unlink(missing_ok=True)
    return canonical, streaming


def test_price_only_emits_two_atoms():
    events = [_upd(1000, "A", "Buy", 100, 5, 99, 5)]  # bid reprices away
    c, s = _run_both(events)
    assert len(c) == 2 and len(s) == 2
    out_c = c[c.action == "reprice_out"].iloc[0]
    in_c  = c[c.action == "reprice_in"].iloc[0]
    assert abs(float(out_c.qty) - 5.0) < 1e-9   # old qty
    assert abs(float(in_c.qty)  - 5.0) < 1e-9   # new qty (same in price-only)
    assert out_c.direction == "away" == in_c.direction
    assert out_c.book_side == "bid"


def test_bid_price_decrease_is_away():
    events = [_upd(1000, "A", "Buy", 100, 5, 99, 5)]
    c, _ = _run_both(events)
    assert (c.direction == "away").all()


def test_bid_price_increase_is_toward():
    events = [_upd(1000, "A", "Buy", 99, 5, 100, 5)]
    c, _ = _run_both(events)
    assert (c.direction == "toward").all()


def test_ask_price_increase_is_away():
    events = [_upd(1000, "A", "Sell", 100, 5, 101, 5)]
    c, _ = _run_both(events)
    assert (c.direction == "away").all()


def test_ask_price_decrease_is_toward():
    events = [_upd(1000, "A", "Sell", 101, 5, 100, 5)]
    c, _ = _run_both(events)
    assert (c.direction == "toward").all()


def test_same_price_update_excluded():
    """Price-only in qty domain (qty changes, price same) must NOT appear in reprice atoms."""
    events = [_upd(1000, "A", "Buy", 100, 5, 100, 7)]  # qty increase, same price
    c, s = _run_both(events)
    assert len(c) == 0 and len(s) == 0


def test_price_and_qty_change_out_uses_old_qty_in_uses_new():
    events = [_upd(1000, "A", "Buy", 100, 10, 99, 8)]  # both changed
    c, _ = _run_both(events)
    out = c[c.action == "reprice_out"].iloc[0]
    inn = c[c.action == "reprice_in"].iloc[0]
    assert abs(float(out.qty) - 10.0) < 1e-9   # old qty leaves old level
    assert abs(float(inn.qty) - 8.0)  < 1e-9   # new qty arrives at new level


def test_canonical_streaming_parity_reprice():
    events = [
        _upd(1000, "A", "Buy",  100, 5,  99, 5),   # bid away
        _upd(1001, "A", "Buy",   99, 5, 100, 5),   # bid toward
        _upd(1002, "B", "Sell", 101, 3, 102, 3),   # ask away
        _upd(1003, "B", "Sell", 102, 3, 101, 3),   # ask toward
        _upd(1004, "C", "Buy",  100, 10, 98, 8),   # bid away + qty change
    ]
    c, s = _run_both(events)
    assert len(c) == len(s) == 10  # 5 events × 2 atoms each
    key = ["ts", "order_uid", "action", "book_side", "direction"]
    cc = c.sort_values(key).reset_index(drop=True)
    ss = s.sort_values(key).reset_index(drop=True)
    assert list(cc["direction"]) == list(ss["direction"])
    assert np.allclose(cc["qty"].astype(float), ss["qty"].astype(float))


def test_windowed_reprice_flow_causal():
    events = [_upd(5000, "A", "Buy", 100, 5, 99, 5)]  # reprice at t=5000
    atoms = build_reprice_atoms(events)
    # Anchor before event: no atoms
    f_before = windowed_reprice_flow(atoms, np.array([4000]), {"1s": 1000})
    assert f_before.iloc[0]["bid_reprice_away_qty_1s"] == 0.0
    # Anchor at event: one away event counted
    f_at = windowed_reprice_flow(atoms, np.array([5000]), {"1s": 1000})
    assert f_at.iloc[0]["bid_reprice_away_qty_1s"] == 5.0
    assert f_at.iloc[0]["bid_reprice_toward_qty_1s"] == 0.0


def test_windowed_reprice_net():
    # Out 10, in 8 (price+qty change bid away)
    events = [_upd(1000, "A", "Buy", 100, 10, 99, 8)]
    atoms = build_reprice_atoms(events)
    f = windowed_reprice_flow(atoms, np.array([2000]), {"5s": 5000})
    out_q = f.iloc[0]["bid_reprice_out_qty_5s"]
    in_q  = f.iloc[0]["bid_reprice_in_qty_5s"]
    net   = f.iloc[0]["bid_reprice_net_5s"]
    assert abs(out_q - 10.0) < 1e-9
    assert abs(in_q  - 8.0)  < 1e-9
    assert abs(net - (8.0 - 10.0)) < 1e-9  # net = in - out = -2
