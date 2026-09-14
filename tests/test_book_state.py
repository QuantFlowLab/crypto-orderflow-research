"""test_book_state.py — Structural invariants for OrderBook replay.

Tests verify:
  - placed → active
  - cancel removes from active
  - execution reduces maker qty / removes taker
  - no structural violations for well-formed input
  - known invariant violations are counted correctly
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from book_state import OrderBook
from orderflow import build_event_table


def _placed(ts, uid, side, price, qty):
    return {"uid": f"ev{uid}p", "timestamp": ts,
            "event": {"OrderPlaced": {"order": {
                "uid": uid, "side": side, "limitPrice": str(price),
                "quantity": str(qty), "filled": "0", "type": "Post",
            }, "reason": "new_user_order"}}}


def _cancelled(ts, uid, side, price, qty):
    return {"uid": f"ev{uid}c", "timestamp": ts,
            "event": {"OrderCancelled": {"order": {
                "uid": uid, "side": side, "limitPrice": str(price),
                "quantity": str(qty), "filled": "0", "type": "Post",
            }, "reason": "cancelled_by_user"}}}


def _exec(ts, ev_uid, maker, taker, price, qty):
    return {"uid": ev_uid, "timestamp": ts,
            "event": {"Execution": {"execution": {
                "uid": ev_uid,
                "makerOrder": {"uid": maker},
                "takerOrder": {"uid": taker},
                "price": str(price), "quantity": str(qty),
            }}}}


def _updated(ts, uid, side, new_price, new_qty, old_qty):
    return {"uid": f"ev{uid}u", "timestamp": ts,
            "event": {"OrderUpdated": {
                "oldOrder": {"uid": uid, "side": side,
                             "limitPrice": str(new_price),
                             "quantity": str(old_qty), "filled": "0", "type": "Post"},
                "newOrder": {"uid": uid, "side": side,
                             "limitPrice": str(new_price),
                             "quantity": str(new_qty), "filled": "0", "type": "Post"},
                "reason": "partial_fill",
            }}}


class TestOrderBookInvariants:
    def _book(self, raw_orders, raw_execs):
        df = build_event_table(raw_orders, raw_execs)
        book = OrderBook()
        book.replay(df)
        return book

    def test_placed_order_is_active(self):
        book = self._book([_placed(1000, "A", "buy", 100.0, 5.0)], [])
        assert "A" in book._orders
        assert book._orders["A"].qty == pytest.approx(5.0)

    def test_cancel_removes_order(self):
        book = self._book([
            _placed(1000, "A", "buy", 100.0, 5.0),
            _cancelled(2000, "A", "buy", 100.0, 5.0),
        ], [])
        assert "A" not in book._orders
        assert book.stats.unknown_cancel == 0

    def test_execution_reduces_maker_removes_taker(self):
        book = self._book([
            _placed(1000, "maker1", "sell", 101.0, 10.0),
            _placed(1001, "taker1", "buy",  101.0,  3.0),
        ], [_exec(1002, "ex1", "maker1", "taker1", 101.0, 3.0)])
        assert "taker1" not in book._orders
        assert "maker1" in book._orders
        assert book._orders["maker1"].qty == pytest.approx(7.0)

    def test_no_violations_for_valid_sequence(self):
        # M rests at 101; T is aggressive buy at 100 (limit, below ask → no crossing)
        book = self._book([
            _placed(1000, "M", "sell", 101.0, 5.0),
            _placed(1001, "T", "buy",  100.0, 2.0),
        ], [_exec(1002, "ex1", "M", "T", 101.0, 2.0)])
        assert book.stats.crossed_book       == 0
        assert book.stats.negative_qty       == 0
        assert book.stats.duplicate_active_uid == 0
        assert book.stats.unknown_cancel     == 0
        assert book.stats.unknown_update     == 0
        assert book.stats.unknown_exec_maker == 0
        assert book.stats.unknown_exec_taker == 0

    def test_cancel_unknown_uid_incremented(self):
        """Cancelling a UID not in book → unknown_cancel += 1."""
        book = OrderBook()
        df = build_event_table([_cancelled(1000, "GHOST", "buy", 100.0, 1.0)], [])
        book.replay(df)
        assert book.stats.unknown_cancel == 1

    def test_update_replacement_qty(self):
        """OrderUpdated.qty is replacement; book should reflect new qty."""
        book = self._book([
            _placed( 1000, "A", "sell", 101.0, 10.0),
            _updated(2000, "A", "sell", 101.0,  7.0, 10.0),
        ], [])
        assert book._orders["A"].qty == pytest.approx(7.0)

    def test_bbo_snapshot(self):
        book = self._book([
            _placed(1000, "b1", "buy",  100.0, 3.0),
            _placed(1000, "a1", "sell", 101.0, 5.0),
        ], [])
        snap = book.bbo_snapshot()
        assert snap["best_bid"] == pytest.approx(100.0)
        assert snap["best_ask"] == pytest.approx(101.0)
        assert snap["spread"]   == pytest.approx(1.0)

    def test_depth_bps(self):
        book = self._book([
            _placed(1000, "b1", "buy",  100.0, 3.0),
            _placed(1000, "b2", "buy",   99.0, 2.0),
            _placed(1000, "a1", "sell", 101.0, 4.0),
        ], [])
        # mid=100.5; 200bps → lo=99.495, hi=101.505 → includes bid@100 AND bid@99 AND ask@101
        d = book.depth_bps(200)
        assert d["depth_bid"] == pytest.approx(5.0)
        assert d["depth_ask"] == pytest.approx(4.0)

    def test_near_bbo_completeness_all_known(self):
        """When all orders have placed_ms (no left-censoring), pct_known=100%."""
        book = self._book([
            _placed(1000, "b1", "buy",  100.0, 5.0),
            _placed(1000, "a1", "sell", 101.0, 3.0),
        ], [])
        c = book.near_bbo_completeness()
        assert c["bid_L1"]["pct_known"] == pytest.approx(100.0)
        assert c["ask_L1"]["pct_known"] == pytest.approx(100.0)

    def test_near_bbo_completeness_left_censored(self):
        """Left-censored order (update without prior place) → placed_ms=None → pct_known < 100%."""
        book = OrderBook()
        # Inject a left-censored order directly (simulating warmup-less start)
        from book_state import Order
        book._orders["ghost"] = Order(uid="ghost", side="buy", price=100.0,
                                      qty=10.0, placed_ms=None, last_update_ms=2000)
        book._orders["known"] = Order(uid="known", side="buy", price=100.0,
                                      qty=5.0, placed_ms=1000, last_update_ms=1000)
        c = book.near_bbo_completeness()
        # total=15, known=5 → 33.3%
        assert c["bid_L1"]["pct_known"] == pytest.approx(33.33, abs=0.01)
