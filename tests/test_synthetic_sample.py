"""test_synthetic_sample.py — Regression guard for data/sample/ integrity.

Ensures no duplicate event_uid and that counts match documented expectations.
"""
import json
from pathlib import Path

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


def _load_orders():
    with open(SAMPLE / "example_orders.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def test_no_duplicate_event_uid():
    orders = _load_orders()
    uids = [e["uid"] for e in orders]
    assert len(uids) == len(set(uids)), (
        f"Duplicate event_uid in example_orders.jsonl: "
        f"{[u for u in set(uids) if uids.count(u) > 1]}"
    )


def test_sample_event_counts():
    orders = _load_orders()
    assert len(orders) == 17, f"Expected 17 order events, got {len(orders)}"
    by_type = {}
    for e in orders:
        et = next(iter(e.get("event", {})), "unknown")
        by_type[et] = by_type.get(et, 0) + 1
    assert by_type.get("OrderPlaced", 0) == 10, f"Expected 10 OrderPlaced, got {by_type}"
    assert by_type.get("OrderCancelled", 0) == 2, f"Expected 2 OrderCancelled, got {by_type}"
    assert by_type.get("OrderUpdated", 0) == 5, f"Expected 5 OrderUpdated, got {by_type}"


def test_sample_has_same_price_update():
    """Exactly one OrderUpdated must be a same-price quantity change."""
    orders = _load_orders()
    same_price = 0
    price_change = 0
    for e in orders:
        ev = e.get("event", {})
        if "OrderUpdated" not in ev:
            continue
        body = ev["OrderUpdated"]
        old_p = body.get("oldOrder", {}).get("limitPrice")
        new_p = body.get("newOrder", {}).get("limitPrice")
        if old_p == new_p:
            same_price += 1
        else:
            price_change += 1
    assert same_price == 1, f"Expected 1 same-price update, got {same_price}"
    assert price_change == 4, f"Expected 4 price-changing updates, got {price_change}"
