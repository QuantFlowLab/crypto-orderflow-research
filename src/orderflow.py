"""orderflow.py — Canonical order-flow event table and taxonomy.

Spec: KRAKEN-OF-1 task 1-2.

Rules:
  - timestamp_ms is the only time unit; sub-ms ordering is NOT imposed
  - Events with the same timestamp_ms form an unordered set
  - OrderUpdated.qty is replacement quantity (not delta)
  - No FIFO / queue-position features in this module
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

EventType = Literal[
    "OrderPlaced", "OrderCancelled", "OrderUpdated", "OrderRejected", "Execution"
]
Side = Literal["buy", "sell"]


@dataclass
class RawEvent:
    timestamp_ms: int
    event_type: EventType
    event_uid: str
    order_uid: str | None
    side: Side | None
    price: float | None       # limitPrice; None for market orders
    qty: float | None         # replacement qty (OrderUpdated) or placed qty
    old_qty: float | None     # only for OrderUpdated
    filled_qty: float | None  # cumulative filled at event time
    maker_uid: str | None     # Execution only
    taker_uid: str | None     # Execution only
    execution_price: float | None
    execution_qty: float | None
    reason: str | None
    ordering_ambiguous: bool  # True if n_events_same_ms > 1


def build_event_table(raw_orders: list[dict], raw_execs: list[dict]) -> pd.DataFrame:
    """Build canonical flat event table from raw API responses.

    Events with identical timestamp_ms are flagged as ordering_ambiguous.
    """
    rows: list[dict] = []

    for e in raw_orders:
        ev_inner = e.get("event", {})
        ev_type = next(iter(ev_inner), None)
        body = ev_inner.get(ev_type, {}) if ev_type else {}

        row: dict = {
            "timestamp_ms": e.get("timestamp"),
            "event_type": ev_type,
            "event_uid": e.get("uid"),
            "order_uid": None, "side": None, "price": None,
            "qty": None, "old_qty": None, "filled_qty": None,
            "maker_uid": None, "taker_uid": None,
            "execution_price": None, "execution_qty": None,
            "reason": body.get("reason"),
        }

        if ev_type == "OrderPlaced":
            o = body.get("order", {})
            row.update(_from_order(o))
        elif ev_type in ("OrderCancelled", "OrderRejected"):
            o = body.get("order", {})
            row.update(_from_order(o))
        elif ev_type == "OrderUpdated":
            old = body.get("oldOrder", {})
            new = body.get("newOrder", {})
            row.update(_from_order(new))
            row["old_qty"] = _to_float(old.get("quantity"))
        rows.append(row)

    for e in raw_execs:
        ev_inner = e.get("event", {})
        ex = ev_inner.get("Execution", {}).get("execution", {})
        rows.append({
            "timestamp_ms": e.get("timestamp"),
            "event_type": "Execution",
            "event_uid": e.get("uid"),
            "order_uid": None, "side": None, "price": None,
            "qty": None, "old_qty": None, "filled_qty": None,
            "maker_uid": ex.get("makerOrder", {}).get("uid"),
            "taker_uid": ex.get("takerOrder", {}).get("uid"),
            "execution_price": _to_float(ex.get("price")),
            "execution_qty": _to_float(ex.get("quantity")),
            "reason": None,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp_ms", "event_type", "event_uid", "order_uid",
            "side", "price", "qty", "old_qty", "filled_qty",
            "maker_uid", "taker_uid", "execution_price", "execution_qty",
            "reason", "n_events_same_ms", "ordering_ambiguous",
        ])
    df = pd.DataFrame(rows)
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
    df = df.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)

    # Flag ordering ambiguity
    same_ms_counts = df["timestamp_ms"].value_counts()
    df["n_events_same_ms"] = df["timestamp_ms"].map(same_ms_counts)
    df["ordering_ambiguous"] = df["n_events_same_ms"] > 1

    return df


def _from_order(o: dict) -> dict:
    raw_side = o.get("direction") or o.get("side")  # raw Kraken uses "direction" (Buy/Sell)
    return {
        "order_uid": o.get("uid"),
        "side": raw_side.lower() if isinstance(raw_side, str) else None,
        "price": _to_float(o.get("limitPrice")),
        "qty": _to_float(o.get("quantity")),
        "filled_qty": _to_float(o.get("filled")),
    }


def _to_float(v: object) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flow_taxonomy(df: pd.DataFrame, since_ms: int, before_ms: int,
                  bucket_ms: int = 1000) -> pd.DataFrame:
    """Aggregate canonical order-flow atoms per time bucket and side.

    Returns one row per (bucket_start_ms, side) with:
      placed_qty, placed_orders, cancelled_qty, cancelled_orders,
      executed_qty, executions, updated_qty, updates, price_moved_updates.

    Price-moving updates: price changed (removal from old level + placement on new level).
    Non-price updates: qty-only change (size reduction / increase in place).
    Both are counted separately; the raw OrderUpdated is also kept.
    """
    window = df[(df.timestamp_ms >= since_ms) & (df.timestamp_ms < before_ms)].copy()
    window["bucket"] = (window.timestamp_ms // bucket_ms) * bucket_ms

    records: list[dict] = []
    for (bucket, side), grp in window.groupby(["bucket", "side"], dropna=False):
        placed = grp[grp.event_type == "OrderPlaced"]
        cancelled = grp[grp.event_type == "OrderCancelled"]
        updated = grp[grp.event_type == "OrderUpdated"]
        # Price-moving: new price differs from old (not captured at raw level if old_price absent)
        execs = df[(df.timestamp_ms >= bucket) & (df.timestamp_ms < bucket + bucket_ms) &
                   (df.event_type == "Execution")]

        records.append({
            "bucket_ms": bucket,
            "side": side,
            "placed_qty": placed["qty"].sum(skipna=True),
            "placed_orders": len(placed),
            "cancelled_qty": cancelled["qty"].sum(skipna=True),
            "cancelled_orders": len(cancelled),
            "updated_qty": updated["qty"].sum(skipna=True),
            "updates": len(updated),
            "executed_qty": execs["execution_qty"].sum(skipna=True),
            "executions": len(execs),
            "n_ambiguous_events": int(grp["ordering_ambiguous"].sum()),
        })

    return pd.DataFrame(records)
