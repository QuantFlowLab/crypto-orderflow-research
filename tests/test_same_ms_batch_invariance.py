"""test_same_ms_batch_invariance.py — R1 REGRESSION (critical).

Randomize event order WITHIN each identical timestamp_ms batch and assert the post-batch
market-state (BBO, depth, active count, QA stats) is invariant. This guards the same-ms
place-before-cancel bug that once produced a systematically crossed book.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from orderflow import build_event_table
from book_state import OrderBook


def _place(ts, uid, direction, price, qty, otype="Limit"):
    return {"timestamp": ts, "uid": f"e{uid}", "event": {"OrderPlaced": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": otype, "filled": "0"}}}}


def _cancel(ts, uid, direction, price, qty):
    return {"timestamp": ts, "uid": f"c{uid}", "event": {"OrderCancelled": {"order": {
        "uid": uid, "direction": direction, "limitPrice": str(price),
        "quantity": str(qty), "orderType": "Limit", "filled": "0"}}}}


def _update(ts, uid, direction, price, old_q, new_q):
    return {"timestamp": ts, "uid": f"u{uid}", "event": {"OrderUpdated": {
        "oldOrder": {"uid": uid, "direction": direction, "limitPrice": str(price), "quantity": str(old_q)},
        "newOrder": {"uid": uid, "direction": direction, "limitPrice": str(price),
                     "quantity": str(new_q), "orderType": "Limit", "filled": "0"}}}}


def _exec(ts, maker, taker, m_dir, t_dir, price, qty):
    return {"timestamp": ts, "uid": f"x{maker}{taker}", "event": {"Execution": {"execution": {
        "makerOrder": {"uid": maker, "direction": m_dir},
        "takerOrder": {"uid": taker, "direction": t_dir},
        "quantity": str(qty), "price": str(price), "usdValue": str(float(price) * float(qty))}}}}


def _scenario():
    # warmup (distinct ms) then one dense same-ms batch @2000 exercising every event type
    orders = [
        _place(1000, "A", "Buy", 100, 5),
        _place(1000, "B", "Sell", 101, 3),
        # same-ms batch @2000:
        _place(2000, "C", "Buy", 99, 2),     # placed AND cancelled same ms -> must vanish
        _cancel(2000, "C", "Buy", 99, 2),
        _place(2000, "T", "Buy", 101, 1),    # taker, filled same ms -> must vanish
        _update(2000, "A", "Buy", 100, 5, 4),
        _place(2000, "D", "Sell", 102, 7),
    ]
    execs = [_exec(2000, "B", "T", "Sell", "Buy", 101, 1)]  # B reduced 3->2, T removed
    return orders, execs


def _final_state(orders, execs):
    df = build_event_table(orders, execs)
    book = OrderBook()
    book.replay(df)
    snap = book.bbo_snapshot()
    depth = book.depth_bps(50)
    s = book.stats
    return {
        "best_bid": snap["best_bid"], "best_ask": snap["best_ask"],
        "best_bid_qty": snap["best_bid_qty"], "best_ask_qty": snap["best_ask_qty"],
        "n_active": snap["n_active"], "depth_bid": depth["depth_bid"], "depth_ask": depth["depth_ask"],
        "crossed": s.crossed_book, "neg": s.negative_qty, "dup": s.duplicate_active_uid,
    }


def test_intra_ms_shuffle_invariance():
    orders, execs = _scenario()
    ref = _final_state(orders, execs)
    rng = random.Random(0)
    for _ in range(30):
        o2 = orders[:]; rng.shuffle(o2)
        e2 = execs[:]; rng.shuffle(e2)
        assert _final_state(o2, e2) == ref


def test_batch_terminal_semantics_correct():
    ref = _final_state(*_scenario())
    # C (place+cancel same ms) and T (filled same ms) gone; A updated 5->4; B reduced 3->2; D rests
    assert ref["n_active"] == 3          # A, B, D
    assert ref["best_bid"] == 100 and ref["best_bid_qty"] == 4   # A after update
    assert ref["best_ask"] == 101 and ref["best_ask_qty"] == 2   # B after exec
    assert ref["crossed"] == 0 and ref["neg"] == 0 and ref["dup"] == 0
