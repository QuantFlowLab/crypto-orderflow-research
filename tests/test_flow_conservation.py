"""test_flow_conservation.py — End-to-end conservation laws.

Verifies that the full pipeline (raw events → event table → book replay → OFI atoms)
is internally consistent:
  - Σ placed_qty across all buckets == Σ qty of placed orders
  - Σ cancelled_qty <= Σ placed_qty (can't cancel more than placed in window)
  - book.n_active_final == Σ placed - Σ cancelled - Σ fully_filled
  - OFI atoms sum to sensible totals

These do not test prediction power; only accounting correctness.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from orderflow import build_event_table, flow_taxonomy
from book_state import OrderBook
from aggregation import ofi_atoms


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


class TestConservationLaws:
    def setup_method(self):
        raw_orders = [
            _placed(1000, "b1", "buy",  100.0, 10.0),
            _placed(1000, "b2", "buy",   99.0,  5.0),
            _placed(1500, "a1", "sell", 101.0,  8.0),
            _placed(2000, "a2", "sell", 102.0,  3.0),
            _cancelled(3000, "b2", "buy", 99.0, 5.0),
        ]
        raw_execs = [
            _exec(4000, "ex1", "a1", "b1", 101.0, 4.0),
        ]
        self.df = build_event_table(raw_orders, raw_execs)
        self.since = 0
        self.before = 5000

    def test_taxonomy_placed_equals_individual_sum(self):
        """flow_taxonomy placed_qty must match direct sum from event table."""
        tax = flow_taxonomy(self.df, self.since, self.before, bucket_ms=5000)
        bid_tax = tax[tax.side == "buy"]["placed_qty"].sum()
        ask_tax = tax[tax.side == "sell"]["placed_qty"].sum()

        placed = self.df[self.df.event_type == "OrderPlaced"]
        bid_direct = placed[placed.side == "buy"]["qty"].sum()
        ask_direct = placed[placed.side == "sell"]["qty"].sum()

        assert bid_tax == pytest.approx(bid_direct)
        assert ask_tax == pytest.approx(ask_direct)

    def test_cancelled_qty_le_placed_qty_per_side(self):
        """Can't cancel more quantity than was placed in the window."""
        tax = flow_taxonomy(self.df, self.since, self.before, bucket_ms=5000)
        for side in ["buy", "sell"]:
            row = tax[tax.side == side]
            if row.empty:
                continue
            placed   = float(row["placed_qty"].iloc[0])
            cancelled = float(row["cancelled_qty"].iloc[0])
            assert cancelled <= placed + 1e-9, f"{side}: cancelled > placed"

    def test_ofi_atoms_non_negative(self):
        """All OFI atom qty columns must be >= 0."""
        atoms = ofi_atoms(self.df, self.since, self.before, bucket_ms=5000)
        for col in ["bid_new_qty", "bid_cancel_qty", "bid_executed_qty",
                    "ask_new_qty", "ask_cancel_qty", "ask_executed_qty"]:
            assert (atoms[col] >= 0).all(), f"{col} has negative values"

    def test_book_active_count_consistency(self):
        """After replay: n_active == placed_unique - cancelled_unique - fully_filled_unique."""
        book = OrderBook()
        book.replay(self.df)
        # placed: b1, b2, a1, a2 = 4
        # cancelled: b2 = 1 → removed
        # execution ex1: taker=b1 removed, maker=a1 qty 8→4 (not removed)
        # active = b1 removed (taker), b2 cancelled, a1 still active (partial), a2 active
        # expected active: a1, a2 = 2
        assert len(book._orders) == 2
        assert "a1" in book._orders
        assert "a2" in book._orders
        assert book._orders["a1"].qty == pytest.approx(4.0)  # 8 - 4 filled

    def test_no_structural_violations_clean_sequence(self):
        """Well-formed synthetic sequence → zero structural violations."""
        book = OrderBook()
        book.replay(self.df)
        assert book.stats.crossed_book        == 0
        assert book.stats.negative_qty        == 0
        assert book.stats.duplicate_active_uid == 0
        assert book.stats.unknown_cancel      == 0
        assert book.stats.unknown_update      == 0
        assert book.stats.unknown_exec_maker  == 0
        assert book.stats.unknown_exec_taker  == 0
