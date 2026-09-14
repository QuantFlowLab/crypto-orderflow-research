"""test_orderflow_accounting.py — Accounting conservation laws for order-flow atoms.

Spec: KRAKEN-OF-1 task 11, flow conservation.

These tests do not require live data: they use synthetic mini-scenarios.
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from orderflow import build_event_table, flow_taxonomy


def _order_placed(ts_ms: int, uid: str, side: str, price: float, qty: float) -> dict:
    return {
        "uid": f"ev_{uid}_placed",
        "timestamp": ts_ms,
        "event": {"OrderPlaced": {"order": {
            "uid": uid, "side": side, "limitPrice": str(price),
            "quantity": str(qty), "filled": "0", "type": "Post",
        }, "reason": "new_user_order"}},
    }


def _order_cancelled(ts_ms: int, uid: str, side: str, price: float, qty: float) -> dict:
    return {
        "uid": f"ev_{uid}_cancelled",
        "timestamp": ts_ms,
        "event": {"OrderCancelled": {"order": {
            "uid": uid, "side": side, "limitPrice": str(price),
            "quantity": str(qty), "filled": "0", "type": "Post",
        }, "reason": "cancelled_by_user"}},
    }


def _execution(ts_ms: int, ev_uid: str, maker: str, taker: str,
               price: float, qty: float) -> dict:
    return {
        "uid": ev_uid,
        "timestamp": ts_ms,
        "event": {"Execution": {"execution": {
            "uid": ev_uid,
            "makerOrder": {"uid": maker},
            "takerOrder": {"uid": taker},
            "price": str(price), "quantity": str(qty),
        }}},
    }


class TestFlowConservation:
    def test_placed_qty_equals_sum_of_individual_orders(self):
        """Sum of placed_qty must equal sum of each order's qty."""
        orders = [
            _order_placed(1000, "A", "buy",  100.0, 5.0),
            _order_placed(1000, "B", "buy",  100.0, 3.0),
            _order_placed(2000, "C", "sell", 101.0, 7.0),
        ]
        df = build_event_table(orders, [])
        tax = flow_taxonomy(df, 0, 3000, bucket_ms=3000)
        bid = tax[tax.side == "buy"].iloc[0]
        ask = tax[tax.side == "sell"].iloc[0]
        assert bid["placed_qty"] == pytest.approx(8.0)
        assert ask["placed_qty"] == pytest.approx(7.0)

    def test_cancelled_qty_conservation(self):
        """Cancelled orders contribute to cancelled_qty; placed orders that
        are NOT cancelled do not appear in cancelled_qty."""
        orders = [
            _order_placed(  1000, "A", "buy", 100.0, 5.0),
            _order_cancelled(2000, "A", "buy", 100.0, 5.0),
            _order_placed(  1000, "B", "buy", 100.0, 3.0),  # not cancelled
        ]
        df = build_event_table(orders, [])
        tax = flow_taxonomy(df, 0, 3000, bucket_ms=3000)
        bid = tax[tax.side == "buy"].iloc[0]
        assert bid["cancelled_qty"] == pytest.approx(5.0)
        assert bid["placed_qty"]    == pytest.approx(8.0)

    def test_execution_qty_in_atoms(self):
        """Executions appear in the executed_qty atom."""
        execs = [_execution(1500, "ex1", "maker1", "taker1", 100.0, 2.0)]
        df = build_event_table([], execs)
        tax = flow_taxonomy(df, 0, 3000, bucket_ms=3000)
        total_exec_qty = tax["executed_qty"].sum()
        assert total_exec_qty == pytest.approx(2.0)

    def test_no_negative_qty_in_taxonomy(self):
        """Qty columns in taxonomy must all be >= 0."""
        orders = [
            _order_placed(1000, "X", "buy",  100.0, 10.0),
            _order_cancelled(1500, "X", "buy", 100.0, 10.0),
        ]
        df = build_event_table(orders, [])
        tax = flow_taxonomy(df, 0, 2000, bucket_ms=2000)
        for col in ["placed_qty", "cancelled_qty", "executed_qty"]:
            assert (tax[col] >= 0).all(), f"{col} has negative values"

    def test_ordering_ambiguous_flagged(self):
        """Two events sharing a timestamp_ms must be flagged ordering_ambiguous."""
        orders = [
            _order_placed(1000, "P", "buy", 100.0, 1.0),
            _order_placed(1000, "Q", "buy", 100.0, 2.0),  # same ms
            _order_placed(2000, "R", "buy", 100.0, 1.0),  # different ms
        ]
        df = build_event_table(orders, [])
        assert df[df.timestamp_ms == 1000]["ordering_ambiguous"].all()
        assert not df[df.timestamp_ms == 2000]["ordering_ambiguous"].any()
