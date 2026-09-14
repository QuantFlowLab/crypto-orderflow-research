"""test_streaming_parity.py — _stream_passive_atoms parity with build_passive_atoms.

Verifies that the streaming implementation (used for large sessions)
produces identical atoms to the canonical implementation for all
OrderUpdated patterns including the price+qty edge case (r2.2 regression).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from passive_flow import build_passive_atoms
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_r2 import _stream_passive_atoms


def _placed(ts, uid, direction, price, qty, otype="Limit"):
    return {"timestamp": ts, "uid": f"e{uid}", "event": {"OrderPlaced": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": otype, "filled": "0"}}}}


def _cancelled(ts, uid, direction, price, qty):
    return {"timestamp": ts, "uid": f"c{uid}", "event": {"OrderCancelled": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": "Limit", "filled": "0"}}}}


def _updated(ts, uid, direction, old_price, old_qty, new_price, new_qty):
    return {"timestamp": ts, "uid": f"u{uid}", "event": {"OrderUpdated": {
        "oldOrder": {"uid": uid, "direction": direction, "limitPrice": str(old_price), "quantity": str(old_qty)},
        "newOrder": {"uid": uid, "direction": direction, "limitPrice": str(new_price),
                     "quantity": str(new_qty), "orderType": "Limit", "filled": "0"}}}}


def _run_parity(events):
    canonical = build_passive_atoms(events)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     encoding="utf-8", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        tmp_path = Path(f.name)
    streaming = _stream_passive_atoms(tmp_path)
    tmp_path.unlink(missing_ok=True)

    if len(canonical) == 0 and len(streaming) == 0:
        return canonical, streaming

    key = ["ts", "order_uid", "action", "book_side"]
    c = canonical.sort_values(key).reset_index(drop=True)[["ts", "order_uid", "book_side", "action", "qty", "price"]]
    s = streaming.sort_values(key).reset_index(drop=True)[["ts", "order_uid", "book_side", "action", "qty", "price"]]
    return c, s


def test_placed_resting_add():
    c, s = _run_parity([_placed(1000, "A", "Buy", 100, 5)])
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "add" == s.iloc[0]["action"]
    assert c.iloc[0]["book_side"] == "bid" == s.iloc[0]["book_side"]


def test_placed_non_resting_excluded():
    c, s = _run_parity([_placed(1000, "A", "Buy", 100, 5, otype="IoC")])
    assert len(c) == 0 and len(s) == 0


def test_cancelled_cancel_atom():
    c, s = _run_parity([_cancelled(1000, "A", "Sell", 100, 4)])
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "cancel" == s.iloc[0]["action"]


def test_update_qty_increase_same_price():
    c, s = _run_parity([_updated(1000, "A", "Buy", 100, 5, 100, 7)])
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "add" == s.iloc[0]["action"]
    assert abs(float(c.iloc[0]["qty"]) - 2.0) < 1e-9


def test_update_price_only_no_atom():
    """97-98.5% of real updates are price-only. Both must skip silently."""
    c, s = _run_parity([_updated(1000, "A", "Sell", 100, 5, 101, 5)])
    assert len(c) == 0 and len(s) == 0


def test_update_price_and_qty_change_no_size_atom():
    """r2.2 regression: price+qty update → 0 passive atoms (REPRICE handles this)."""
    c, s = _run_parity([_updated(1000, "A", "Buy", 100, 10, 99, 12)])
    assert len(c) == 0, f"canonical emitted {len(c)} atoms for price+qty update"
    assert len(s) == 0, f"streaming emitted {len(s)} atoms for price+qty update"


def test_update_price_and_qty_decrease_no_size_atom():
    """r2.2 regression: price+qty decrease → 0 passive atoms."""
    c, s = _run_parity([_updated(1000, "A", "Sell", 101, 10, 102, 7)])
    assert len(c) == 0 and len(s) == 0


def test_mixed_sequence_full_parity():
    events = [
        _placed(1000, "A", "Buy", 100, 5),
        _updated(1100, "A", "Buy", 100, 5, 101, 5),   # price reprice only → excluded
        _updated(1200, "A", "Buy", 101, 5, 101, 3),   # same price, qty decrease
        _cancelled(1300, "A", "Buy", 101, 3),
        _placed(1100, "B", "Sell", 102, 8),
        _updated(1400, "B", "Sell", 102, 8, 103, 8),  # price reprice only → excluded
        _updated(1500, "B", "Sell", 103, 8, 103, 5),  # same price, qty decrease
        # r2.2: price+qty → excluded
        _updated(1600, "C", "Buy", 99, 7, 98, 9),
    ]
    _placed_C = _placed(1050, "C", "Buy", 99, 7)
    c, s = _run_parity([_placed_C] + events)
    assert len(c) == len(s), f"row count differs: canonical={len(c)} streaming={len(s)}"
    for col in ["ts", "book_side", "action"]:
        assert list(c[col]) == list(s[col]), f"mismatch in column {col}"
    assert np.allclose(c["qty"].astype(float), s["qty"].astype(float), atol=1e-9)
