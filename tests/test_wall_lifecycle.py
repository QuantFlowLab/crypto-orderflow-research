"""test_wall_lifecycle.py — R1 Part E:
  - wall_state_causality: future events don't affect past snapshot
  - level_episode_reset: full disappearance + reappearance = new episode_id
  - price_replace_accounting: price-changing Update = cancel@old + add@new, no double count
  - partial_fill_exec_reduce: execution reduces level qty correctly
  - unknown_cancel_skipped: left-censored cancel produces no atom, no error
  - update_same_price_qty_delta: same-price update emits qty delta, not full qty
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from orderflow import build_event_table
import wall_state as ws

_COLS = ["timestamp_ms", "event_type", "event_uid", "order_uid", "side", "price",
         "qty", "old_qty", "filled_qty", "maker_uid", "taker_uid",
         "execution_price", "execution_qty", "reason", "n_events_same_ms", "ordering_ambiguous"]


def _ev(rows):
    return build_event_table(rows, [])


def _placed(ts, uid, direction, price, qty, otype="Limit"):
    return {"timestamp": ts, "uid": f"e{uid}", "event": {"OrderPlaced": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": otype, "filled": "0"}}}}


def _cancelled(ts, uid, direction, price, qty, filled=0):
    return {"timestamp": ts, "uid": f"c{uid}", "event": {"OrderCancelled": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": "Limit", "filled": str(filled)}}}}


def _updated(ts, uid, direction, old_price, old_qty, new_price, new_qty):
    return {"timestamp": ts, "uid": f"u{uid}", "event": {"OrderUpdated": {
        "oldOrder": {"uid": uid, "direction": direction, "limitPrice": str(old_price), "quantity": str(old_qty)},
        "newOrder": {"uid": uid, "direction": direction, "limitPrice": str(new_price),
                     "quantity": str(new_qty), "orderType": "Limit", "filled": "0"}}}}


def _exec(ts, maker, taker, m_dir, t_dir, price, qty):
    return {"timestamp": ts, "uid": f"x{ts}", "event": {"Execution": {"execution": {
        "makerOrder": {"uid": maker, "direction": m_dir},
        "takerOrder": {"uid": taker, "direction": t_dir},
        "quantity": str(qty), "price": str(price), "usdValue": str(float(price) * float(qty))}}}}


def _level_qty(atoms, side, price, anchor):
    """Causal running qty at (side, price) up to anchor."""
    a = ws.level_snapshot(atoms, anchor)
    if len(a) == 0 or "side" not in a.columns:
        return 0.0
    match = a[(a.side == side) & (a.price == price)]
    return float(match.level_qty.iloc[0]) if len(match) else 0.0


def test_wall_state_causality():
    # Place at T=1000; snapshot at T=1000 → qty=5
    # Add cancel at T=2000; snapshot at T=1000 must still be 5 (future excluded)
    ev1 = _ev([_placed(1000, "A", "Buy", 100, 5)])
    atoms1 = ws.build_level_atoms(ev1)
    assert _level_qty(atoms1, "buy", 100, 1000) == 5.0

    ev2 = _ev([_placed(1000, "A", "Buy", 100, 5), _cancelled(2000, "A", "Buy", 100, 5)])
    atoms2 = ws.build_level_atoms(ev2)
    assert _level_qty(atoms2, "buy", 100, 1000) == 5.0   # cancel at 2000 excluded
    assert _level_qty(atoms2, "buy", 100, 2000) == 0.0   # cancel included at anchor 2000


def test_level_episode_reset():
    raw = [_placed(1000, "A", "Buy", 100, 5),
           _cancelled(2000, "A", "Buy", 100, 5),   # level → 0, episode 1 ends
           _placed(3000, "B", "Buy", 100, 3)]       # new episode
    atoms = ws.build_level_atoms(_ev(raw))
    ep_seq = atoms[(atoms.side == "buy") & (atoms.price == 100)]["episode_id"].tolist()
    assert ep_seq[0] == ep_seq[1]        # place and cancel share episode 1
    assert ep_seq[2] > ep_seq[1]         # new placement = new (higher) episode_id


def test_price_replace_accounting():
    # Place uid=X at price=100, qty=5; then update price to 101 (same qty)
    # Expected: cancel atom at 100, add atom at 101; no double count
    raw = [_placed(1000, "X", "Buy", 100, 5),
           _updated(2000, "X", "Buy", 100, 5, 101, 5)]
    atoms = ws.build_level_atoms(_ev(raw))

    buy_100 = atoms[(atoms.side == "buy") & (atoms.price == 100)]
    buy_101 = atoms[(atoms.side == "buy") & (atoms.price == 101)]

    assert any(buy_100.action == "cancel")   # cancel at old price
    assert any(buy_101.action == "add")      # add at new price

    snap = ws.level_snapshot(atoms, 2001)
    lvl100 = snap[(snap.side == "buy") & (snap.price == 100)]
    lvl101 = snap[(snap.side == "buy") & (snap.price == 101)]
    assert len(lvl100) == 0 or lvl100.level_qty.iloc[0] == 0   # old level empty
    assert len(lvl101) > 0 and abs(lvl101.level_qty.iloc[0] - 5.0) < 1e-9  # new level has qty


def test_partial_fill_exec_reduce():
    # Place 10, exec 3 → level qty should be 7
    raw = [_placed(1000, "M", "Buy", 100, 10),
           _placed(1000, "T", "Sell", 100, 3)]
    execs = [_exec(1000, "M", "T", "Buy", "Sell", 100, 3)]
    atoms = ws.build_level_atoms(build_event_table(raw, execs))
    assert any(atoms.action == "exec_reduce")
    assert abs(_level_qty(atoms, "buy", 100, 2000) - 7.0) < 1e-9


def test_unknown_cancel_skipped():
    # Cancel for uid never placed → no atom emitted, no error
    raw = [_cancelled(1000, "GHOST", "Buy", 100, 5)]
    atoms = ws.build_level_atoms(_ev(raw))
    assert len(atoms) == 0


def test_update_same_price_qty_delta():
    # Place 10, update to 7 (same price) → cancel delta=3
    raw = [_placed(1000, "X", "Buy", 100, 10),
           _updated(2000, "X", "Buy", 100, 10, 100, 7)]
    atoms = ws.build_level_atoms(_ev(raw))
    cancels = atoms[(atoms.side == "buy") & (atoms.price == 100) & (atoms.action == "cancel")]
    assert len(cancels) == 1
    assert abs(cancels.iloc[0]["qty"] - 3.0) < 1e-9
    assert abs(_level_qty(atoms, "buy", 100, 3000) - 7.0) < 1e-9
