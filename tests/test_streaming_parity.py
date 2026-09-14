"""test_streaming_parity.py — Verify _stream_passive_atoms == build_passive_atoms.

Hard gate: streaming and canonical implementations must produce identical atom sequences
for all OrderUpdated patterns:
  - qty-only change (increase / decrease)
  - price-only change, qty unchanged -> both must skip (no atom emitted)
  - qty + price change -> emit qty delta
  - OrderPlaced (resting / non-resting)
  - OrderCancelled
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from passive_flow import build_passive_atoms
# Import streaming helper from run_r2 at repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_r2 import _stream_passive_atoms


def _make_event(ts, et, uid, direction, price, qty, old_price=None, old_qty=None,
                otype="Limit", filled=0):
    if et == "OrderPlaced":
        return {"timestamp": ts, "uid": f"e{uid}", "event": {"OrderPlaced": {"order": {
            "uid": uid, "direction": direction, "limitPrice": str(price),
            "quantity": str(qty), "orderType": otype, "filled": str(filled)}}}}
    elif et == "OrderCancelled":
        return {"timestamp": ts, "uid": f"c{uid}", "event": {"OrderCancelled": {"order": {
            "uid": uid, "direction": direction, "limitPrice": str(price),
            "quantity": str(qty), "orderType": otype, "filled": str(filled)}}}}
    elif et == "OrderUpdated":
        return {"timestamp": ts, "uid": f"u{uid}", "event": {"OrderUpdated": {
            "oldOrder": {"uid": uid, "direction": direction,
                         "limitPrice": str(old_price or price), "quantity": str(old_qty or qty)},
            "newOrder": {"uid": uid, "direction": direction, "limitPrice": str(price),
                         "quantity": str(qty), "orderType": otype, "filled": str(filled)}}}}


def _run_parity(events):
    """Run both implementations on the same events, return (canonical, streaming) DataFrames."""
    canonical = build_passive_atoms(events)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     encoding="utf-8", delete=False) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        tmp_path = Path(f.name)

    streaming = _stream_passive_atoms(tmp_path)
    tmp_path.unlink(missing_ok=True)
    # Handle empty DataFrames (no atoms emitted)
    if len(canonical) == 0 and len(streaming) == 0:
        return canonical, streaming
    # Normalize: sort by (ts, order_uid, action, book_side) for comparison
    key = ["ts", "order_uid", "action", "book_side"]
    c = canonical.sort_values(key).reset_index(drop=True)[["ts", "order_uid", "book_side", "action", "qty", "price"]]
    s = streaming.sort_values(key).reset_index(drop=True)[["ts", "order_uid", "book_side", "action", "qty", "price"]]
    return c, s


def test_placed_resting_add():
    events = [_make_event(1000, "OrderPlaced", "A", "Buy", 100, 5, otype="Limit")]
    c, s = _run_parity(events)
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "add" == s.iloc[0]["action"]
    assert c.iloc[0]["book_side"] == "bid" == s.iloc[0]["book_side"]
    assert abs(float(c.iloc[0]["qty"]) - 5.0) < 1e-9
    assert abs(float(s.iloc[0]["qty"]) - 5.0) < 1e-9


def test_placed_non_resting_excluded():
    events = [_make_event(1000, "OrderPlaced", "A", "Buy", 100, 5, otype="IoC")]
    c, s = _run_parity(events)
    assert len(c) == 0 and len(s) == 0


def test_cancelled_cancel_atom():
    events = [_make_event(1000, "OrderCancelled", "A", "Sell", 100, 4)]
    c, s = _run_parity(events)
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "cancel" == s.iloc[0]["action"]
    assert c.iloc[0]["book_side"] == "ask" == s.iloc[0]["book_side"]


def test_update_qty_increase():
    events = [_make_event(1000, "OrderUpdated", "A", "Buy", 100, 7, old_price=100, old_qty=5)]
    c, s = _run_parity(events)
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "add" == s.iloc[0]["action"]
    assert abs(float(c.iloc[0]["qty"]) - 2.0) < 1e-9
    assert abs(float(s.iloc[0]["qty"]) - 2.0) < 1e-9


def test_update_qty_decrease():
    events = [_make_event(1000, "OrderUpdated", "A", "Buy", 100, 3, old_price=100, old_qty=5)]
    c, s = _run_parity(events)
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "cancel" == s.iloc[0]["action"]
    assert abs(float(c.iloc[0]["qty"]) - 2.0) < 1e-9
    assert abs(float(s.iloc[0]["qty"]) - 2.0) < 1e-9


def test_update_price_only_no_atom():
    """97-98.5% of real updates are price-only (qty unchanged). Both must skip silently."""
    events = [_make_event(1000, "OrderUpdated", "A", "Sell", 101, 5, old_price=100, old_qty=5)]
    c, s = _run_parity(events)
    assert len(c) == 0, f"canonical emitted {len(c)} atoms for price-only update"
    assert len(s) == 0, f"streaming emitted {len(s)} atoms for price-only update"


def test_update_price_and_qty_change():
    events = [_make_event(1000, "OrderUpdated", "A", "Buy", 101, 7, old_price=100, old_qty=5)]
    c, s = _run_parity(events)
    assert len(c) == len(s) == 1
    assert c.iloc[0]["action"] == "add" == s.iloc[0]["action"]
    assert abs(float(c.iloc[0]["qty"]) - 2.0) < 1e-9
    assert abs(float(s.iloc[0]["qty"]) - 2.0) < 1e-9


def test_mixed_sequence_full_parity():
    """Real-world mix: place, reprice (qty unchanged), cancel, place, partial fill update."""
    events = [
        _make_event(1000, "OrderPlaced",   "A", "Buy",  100, 5),
        _make_event(1100, "OrderUpdated",  "A", "Buy",  101, 5, old_price=100, old_qty=5),  # reprice only
        _make_event(1200, "OrderUpdated",  "A", "Buy",  101, 3, old_price=101, old_qty=5),  # partial fill
        _make_event(1300, "OrderCancelled","A", "Buy",  101, 3),
        _make_event(1100, "OrderPlaced",   "B", "Sell", 102, 8, otype="Post"),
        _make_event(1400, "OrderUpdated",  "B", "Sell", 103, 8, old_price=102, old_qty=8),  # reprice only
    ]
    c, s = _run_parity(events)
    assert len(c) == len(s), f"row count differs: canonical={len(c)} streaming={len(s)}"
    # Compare all fields
    for col in ["ts", "book_side", "action"]:
        assert list(c[col]) == list(s[col]), f"mismatch in column {col}"
    import numpy as np
    assert np.allclose(c["qty"].astype(float), s["qty"].astype(float), atol=1e-9)
