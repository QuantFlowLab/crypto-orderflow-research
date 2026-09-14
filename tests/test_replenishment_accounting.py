"""test_replenishment_accounting.py — R1 Part F:
  - no_exec_no_pairs: no execution → no refill pairs
  - refill_after_exec: add after exec is counted at correct delay
  - no_future_refill: add AFTER anchor_ts must NOT appear in features
  - replenishment_ratio: ratio = refilled / executed, exec not double-counted
  - same_ms_invariance: shuffling events within same ms doesn't change level atoms or pairs
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from orderflow import build_event_table
import wall_state as ws
import replenishment as rp


def _placed(ts, uid, direction, price, qty):
    return {"timestamp": ts, "uid": f"e{uid}", "event": {"OrderPlaced": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": "Post", "filled": "0"}}}}


def _cancelled(ts, uid, direction, price, qty):
    return {"timestamp": ts, "uid": f"c{uid}", "event": {"OrderCancelled": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": "Limit", "filled": "0"}}}}


def _exec(ts, maker, taker, m_dir, t_dir, price, qty):
    return {"timestamp": ts, "uid": f"x{ts}", "event": {"Execution": {"execution": {
        "makerOrder": {"uid": maker, "direction": m_dir},
        "takerOrder": {"uid": taker, "direction": t_dir},
        "quantity": str(qty), "price": str(price), "usdValue": str(float(price) * float(qty))}}}}


def _atoms_from(orders, execs=()):
    return ws.build_level_atoms(build_event_table(orders, list(execs)))


def test_no_exec_no_pairs():
    raw = [_placed(1000, "A", "Buy", 100, 5), _cancelled(2000, "A", "Buy", 100, 5)]
    atoms = _atoms_from(raw)
    pairs = rp.build_refill_pairs(atoms)
    assert len(pairs) == 0


def test_refill_after_exec():
    # Exec at T=1000, add at T=1500 → pair with delay=500
    raw = [_placed(1000, "M", "Buy", 100, 5),
           _placed(1000, "T", "Sell", 100, 5)]   # taker
    execs = [_exec(1000, "M", "T", "Buy", "Sell", 100, 5)]
    raw2 = [_placed(1500, "R", "Buy", 100, 5)]   # refill
    atoms = _atoms_from(raw + raw2, execs)
    pairs = rp.build_refill_pairs(atoms)
    assert len(pairs) == 1
    assert pairs.iloc[0]["delay_ms"] == 500
    assert abs(pairs.iloc[0]["refill_qty"] - 5.0) < 1e-9

    # Feature at anchor 2000: refill within 1s of exec visible
    f = rp.level_refill_features(pairs, "buy", 100, 2000, {"1s": 1000})
    assert abs(f["refilled_qty_1s"] - 5.0) < 1e-9
    assert f["refill_count_1s"] == 1


def test_no_future_refill():
    # Exec at T=1000, add at T=2000; anchor at T=1500 → add is AFTER anchor, excluded
    raw = [_placed(1000, "M", "Buy", 100, 5),
           _placed(1000, "T", "Sell", 100, 5)]
    execs = [_exec(1000, "M", "T", "Buy", "Sell", 100, 5)]
    raw2 = [_placed(2000, "R", "Buy", 100, 5)]   # future refill
    atoms = _atoms_from(raw + raw2, execs)
    pairs = rp.build_refill_pairs(atoms)
    assert len(pairs) == 1  # pair exists...

    f = rp.level_refill_features(pairs, "buy", 100, 1500, {"5s": 5000})
    # ...but anchor=1500 < add_ts=2000 → excluded from features
    assert f["refilled_qty_5s"] == 0.0
    assert f["refill_count_5s"] == 0


def test_replenishment_ratio_no_double_count():
    # One exec removes 5; two separate add events refill 3+2=5 → ratio ≈ 1.0
    raw = [_placed(1000, "M", "Buy", 100, 5),
           _placed(1000, "T", "Sell", 100, 5)]
    execs = [_exec(1000, "M", "T", "Buy", "Sell", 100, 5)]
    raw2 = [_placed(1200, "R1", "Buy", 100, 3), _placed(1400, "R2", "Buy", 100, 2)]
    atoms = _atoms_from(raw + raw2, execs)
    pairs = rp.build_refill_pairs(atoms)
    assert len(pairs) == 2  # two refill pairs from one exec

    f = rp.level_refill_features(pairs, "buy", 100, 5000, {"5s": 5000})
    # exec_qty counted ONCE (dedup by exec_ts), refilled_qty = 5
    assert abs(f["refilled_qty_5s"] - 5.0) < 1e-9
    assert abs(f["replenishment_ratio_5s"] - 1.0) < 0.01  # ≈ 1.0


def test_replenishment_same_ms_invariance():
    # Shuffle order events within same ms → same level atoms → same pairs
    orders_base = [
        _placed(1000, "M", "Buy", 100, 5),
        _placed(1000, "T", "Sell", 100, 5),
    ]
    execs_base = [_exec(1000, "M", "T", "Buy", "Sell", 100, 5)]
    refill = [_placed(1500, "R", "Buy", 100, 5)]

    def _get_pairs(orders, execs):
        atoms = _atoms_from(orders, execs)
        return frozenset(
            (r.exec_ts, r.add_ts, r.side, r.price, r.exec_qty, r.refill_qty)
            for r in rp.build_refill_pairs(atoms).itertuples(index=False)
        )

    ref = _get_pairs(orders_base + refill, execs_base)
    rng = random.Random(42)
    for _ in range(20):
        o2 = orders_base + refill; rng.shuffle(o2)
        e2 = list(execs_base); rng.shuffle(e2)
        assert _get_pairs(o2, e2) == ref
