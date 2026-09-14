"""aggregation.py — Market-state snapshots on fixed time grids.

Spec: KRAKEN-OF-1 tasks 3-8.

Supported grids: 10ms, 50ms, 100ms, 250ms, 500ms, 1s.
Each snapshot is taken AFTER all events with timestamp_ms <= grid_point are applied.
"""
from __future__ import annotations

from typing import Literal
import pandas as pd

from book_state import OrderBook

GRID_MS = {
    "10ms": 10,
    "50ms": 50,
    "100ms": 100,
    "250ms": 250,
    "500ms": 500,
    "1s": 1000,
}

DepthSpec = Literal["L1", "L5", "L10"]


def build_market_state(
    event_df: pd.DataFrame,
    since_ms: int,
    before_ms: int,
    grid: str = "1s",
    depth_bps_levels: tuple[float, ...] = (5, 10, 25, 50),
    warmup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a regular market-state time series.

    Warmup events (before since_ms) are replayed first to seed the book;
    those snapshots are discarded.
    """
    bucket_ms = GRID_MS[grid]

    book = OrderBook()
    if warmup_df is not None and len(warmup_df) > 0:
        book.replay(warmup_df)

    # Grid points: first tick >= since_ms, step=bucket_ms
    grid_start = ((since_ms + bucket_ms - 1) // bucket_ms) * bucket_ms
    grid_points = range(grid_start, before_ms + 1, bucket_ms)

    # Split events into grid slices
    window_df = event_df[(event_df.timestamp_ms >= since_ms) &
                         (event_df.timestamp_ms < before_ms)].copy()

    records: list[dict] = []
    prev_gp = since_ms

    for gp in grid_points:
        # Apply all events in (prev_gp, gp]
        slice_df = window_df[(window_df.timestamp_ms > prev_gp) &
                             (window_df.timestamp_ms <= gp)]
        if len(slice_df) > 0:
            book.replay(slice_df)

        snap = book.bbo_snapshot()
        row: dict = {"ts_ms": gp, "grid": grid, **snap}

        for bps in depth_bps_levels:
            d = book.depth_bps(bps)
            row[f"bid_qty_{bps}bps"] = d["depth_bid"]
            row[f"ask_qty_{bps}bps"] = d["depth_ask"]
            row[f"obi_{bps}bps"]     = d["obi"]

        # Level counts (L1/L5/L10 distinct price levels)
        for spec, n_levels in [("L1", 1), ("L5", 5), ("L10", 10)]:
            bid_prices = sorted(
                {o.price for o in book._orders.values() if o.side == "buy"}, reverse=True
            )[:n_levels]
            ask_prices = sorted(
                {o.price for o in book._orders.values() if o.side == "sell"}
            )[:n_levels]
            row[f"bid_qty_{spec}"] = sum(
                o.qty for o in book._orders.values()
                if o.side == "buy" and o.price in set(bid_prices)
            )
            row[f"ask_qty_{spec}"] = sum(
                o.qty for o in book._orders.values()
                if o.side == "sell" and o.price in set(ask_prices)
            )

        prev_gp = gp
        records.append(row)

    return pd.DataFrame(records)


def ofi_atoms(event_df: pd.DataFrame, since_ms: int, before_ms: int,
              bucket_ms: int = 1000) -> pd.DataFrame:
    """Compute raw OFI atoms per time bucket.

    Atoms (per bucket, separate for bid/ask):
      new_liquidity, cancelled_liquidity, executed_qty

    Does NOT compute a single OFI number — atoms let the caller
    define any OFI formula without losing information.
    """
    window = event_df[(event_df.timestamp_ms >= since_ms) &
                      (event_df.timestamp_ms < before_ms)].copy()
    window["bucket"] = (window.timestamp_ms // bucket_ms) * bucket_ms

    records: list[dict] = []
    for bucket, grp in window.groupby("bucket"):
        # Bid atoms
        bid = grp[grp.side == "buy"]
        ask = grp[grp.side == "sell"]
        execs = event_df[(event_df.timestamp_ms >= bucket) &
                         (event_df.timestamp_ms < bucket + bucket_ms) &
                         (event_df.event_type == "Execution")]

        records.append({
            "bucket_ms": bucket,
            "bid_new_qty":        bid[bid.event_type == "OrderPlaced"]["qty"].sum(skipna=True),
            "bid_new_orders":     (bid.event_type == "OrderPlaced").sum(),
            "bid_cancel_qty":     bid[bid.event_type == "OrderCancelled"]["qty"].sum(skipna=True),
            "bid_cancel_orders":  (bid.event_type == "OrderCancelled").sum(),
            "bid_executed_qty":   execs["execution_qty"].sum(skipna=True),  # against ask
            "ask_new_qty":        ask[ask.event_type == "OrderPlaced"]["qty"].sum(skipna=True),
            "ask_new_orders":     (ask.event_type == "OrderPlaced").sum(),
            "ask_cancel_qty":     ask[ask.event_type == "OrderCancelled"]["qty"].sum(skipna=True),
            "ask_cancel_orders":  (ask.event_type == "OrderCancelled").sum(),
            "ask_executed_qty":   execs["execution_qty"].sum(skipna=True),  # against bid
            "n_updates":          (grp.event_type == "OrderUpdated").sum(),
            "n_ambiguous":        int(grp["ordering_ambiguous"].sum()),
        })

    return pd.DataFrame(records)
