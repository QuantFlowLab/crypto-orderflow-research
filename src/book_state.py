"""book_state.py — Order-level and price-level book reconstruction.

Spec: KRAKEN-OF-1 tasks 3-6.

Rules:
  - Same-ms events processed as unordered set; book state AT timestamp_ms
    is the state AFTER all events with that timestamp are applied.
  - qty on OrderUpdated = replacement (new total remaining).
  - age fields set to None for left-censored orders (unknown placement time).
  - No FIFO / queue-position assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Optional

import numpy as np
import pandas as pd


# Net-terminal order within a same-ms batch (no sequence_id): an order must exist before it
# can be updated/executed/cancelled; a same-ms place+cancel resolves to 'removed'.
_BATCH_PRIORITY = {"OrderPlaced": 0, "OrderUpdated": 1, "Execution": 2,
                   "OrderCancelled": 3, "OrderRejected": 3}


def _isnull(v) -> bool:
    """True for None or float NaN (numpy object/float columns yield either)."""
    return v is None or (isinstance(v, float) and v != v)



@dataclass
class Order:
    uid: str
    side: str
    price: float
    qty: float
    placed_ms: Optional[int]  # None = left-censored
    last_update_ms: int


@dataclass
class BookStats:
    """QA counters accumulated during replay."""
    crossed_book: int = 0          # true crosses: best_bid > best_ask (hard stop)
    locked_book: int = 0           # locked markets: best_bid == best_ask (informational)
    negative_qty: int = 0
    duplicate_active_uid: int = 0
    unknown_cancel: int = 0
    unknown_update: int = 0
    unknown_exec_maker: int = 0
    unknown_exec_taker: int = 0
    n_ambiguous_ms: int = 0      # timestamp_ms groups with >1 event
    crossing_details: list = field(default_factory=list)  # one dict per crossing batch


class OrderBook:
    """L3 order book: maintains per-UID order state.

    Replay rules for same-ms ambiguity:
      All events with identical timestamp_ms are processed before checking
      any invariant that would depend on ordering within that ms.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self.stats = BookStats()
        # Incremental BBO tracking for O(1) crossed-book check (price>0 multiset per side).
        # Lazy heaps: stale entries are skipped/popped when a price's count drops to 0.
        self._bid_cnt: dict[float, int] = {}
        self._ask_cnt: dict[float, int] = {}
        self._bid_heap: list[float] = []   # max-heap via negated price
        self._ask_heap: list[float] = []   # min-heap

    # ── Replay ────────────────────────────────────────────────────────────────

    def replay(self, df: pd.DataFrame) -> None:
        """Replay a canonicalized event DataFrame (output of orderflow.build_event_table).

        Same-ms events form an UNORDERED batch; the book state at a timestamp is the state
        AFTER all events in that ms are applied. Because the API has no sequence_id and may
        deliver a same-ms Cancel BEFORE its own Place, events within a ms are applied in
        net-terminal order: Placed -> Updated -> Execution -> Cancelled/Rejected. This makes
        a same-ms place+cancel resolve to 'not in book' (its physical terminal state).

        Vectorized ordering: a stable lexsort by (timestamp_ms, batch_priority, original_pos)
        reproduces the same within-ms net-terminal order as a per-group stable sort, but lets
        us iterate once over numpy columns (no per-row Series / iterrows overhead)."""
        n = len(df)
        if n == 0:
            return
        ts    = df["timestamp_ms"].to_numpy(dtype="int64")
        etype = df["event_type"].to_numpy(dtype=object)
        ouid  = df["order_uid"].to_numpy(dtype=object)
        side  = df["side"].to_numpy(dtype=object)
        price = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype="float64")
        qty   = pd.to_numeric(df["qty"], errors="coerce").to_numpy(dtype="float64")
        muid  = df["maker_uid"].to_numpy(dtype=object)
        tuid  = df["taker_uid"].to_numpy(dtype=object)
        exq   = pd.to_numeric(df["execution_qty"], errors="coerce").to_numpy(dtype="float64")
        bp    = df["event_type"].map(_BATCH_PRIORITY).fillna(9).to_numpy(dtype="int64")
        order = np.lexsort((np.arange(n), bp, ts))  # primary ts, then bp, then original order

        prev_ts: int | None = None
        group_size = 0
        for i in order:
            t = int(ts[i])
            if prev_ts is None:
                prev_ts = t
            elif t != prev_ts:
                if group_size > 1:
                    self.stats.n_ambiguous_ms += 1
                self._check_crossed(prev_ts)
                prev_ts = t
                group_size = 0
            self._apply(etype[i], ouid[i], side[i], price[i], qty[i],
                        muid[i], tuid[i], exq[i], t)
            group_size += 1
        if group_size > 1:
            self.stats.n_ambiguous_ms += 1
        self._check_crossed(prev_ts if prev_ts is not None else 0)

    # ── Incremental price multiset (for crossed-book) ──────────────────────────

    def _px_add(self, side: str, price: float) -> None:
        if not price or price <= 0:
            return
        d = self._bid_cnt if side == "buy" else self._ask_cnt
        if d.get(price, 0) == 0:
            heappush(self._bid_heap if side == "buy" else self._ask_heap,
                     -price if side == "buy" else price)
        d[price] = d.get(price, 0) + 1

    def _px_remove(self, side: str, price: float) -> None:
        if not price or price <= 0:
            return
        d = self._bid_cnt if side == "buy" else self._ask_cnt
        if price in d:
            d[price] -= 1
            if d[price] <= 0:
                del d[price]

    def _best_bid(self) -> float | None:
        while self._bid_heap:
            p = -self._bid_heap[0]
            if self._bid_cnt.get(p, 0) > 0:
                return p
            heappop(self._bid_heap)
        return None

    def _best_ask(self) -> float | None:
        while self._ask_heap:
            p = self._ask_heap[0]
            if self._ask_cnt.get(p, 0) > 0:
                return p
            heappop(self._ask_heap)
        return None

    def _apply(self, ev, uid_raw, side_raw, price, qty, m_uid_raw, t_uid_raw,
               exec_qty, ts_ms: int) -> None:
        uid = "" if _isnull(uid_raw) else str(uid_raw)

        if ev == "OrderPlaced":
            if uid in self._orders:
                self.stats.duplicate_active_uid += 1
            else:
                if not _isnull(qty) and qty < 0:
                    self.stats.negative_qty += 1
                p = 0.0 if _isnull(price) else float(price)
                s = "" if _isnull(side_raw) else str(side_raw)
                self._orders[uid] = Order(
                    uid=uid, side=s, price=p,
                    qty=0.0 if _isnull(qty) else float(qty),
                    placed_ms=ts_ms, last_update_ms=ts_ms,
                )
                self._px_add(s, p)

        elif ev in ("OrderCancelled", "OrderRejected"):
            if uid not in self._orders:
                self.stats.unknown_cancel += 1
            else:
                o = self._orders[uid]
                self._px_remove(o.side, o.price)
                del self._orders[uid]

        elif ev == "OrderUpdated":
            if uid not in self._orders:
                self.stats.unknown_update += 1
            else:
                o = self._orders[uid]
                if not _isnull(qty):
                    if float(qty) < 0:
                        self.stats.negative_qty += 1
                    o.qty = float(qty)
                if not _isnull(price):
                    new_price = float(price)
                    if new_price != o.price:
                        self._px_remove(o.side, o.price)
                        self._px_add(o.side, new_price)
                        o.price = new_price
                o.last_update_ms = ts_ms

        elif ev == "Execution":
            m_uid = "" if _isnull(m_uid_raw) else str(m_uid_raw)
            t_uid = "" if _isnull(t_uid_raw) else str(t_uid_raw)
            if m_uid and m_uid not in self._orders:
                self.stats.unknown_exec_maker += 1
            if t_uid and t_uid not in self._orders:
                self.stats.unknown_exec_taker += 1
            # Reduce maker qty by exec_qty; taker is removed (fully filled / IOC)
            if m_uid and m_uid in self._orders and not _isnull(exec_qty):
                mo = self._orders[m_uid]
                mo.qty = max(0.0, mo.qty - float(exec_qty))
                if mo.qty == 0.0:
                    self._px_remove(mo.side, mo.price)
                    del self._orders[m_uid]
            if t_uid and t_uid in self._orders:
                to = self._orders[t_uid]
                self._px_remove(to.side, to.price)
                del self._orders[t_uid]

    def _check_crossed(self, ts_ms: int = 0) -> None:
        bb = self._best_bid()
        ba = self._best_ask()
        if bb is None or ba is None:
            return
        if bb > ba:
            # True cross: negative spread — unexpected, captured for classification
            self.stats.crossed_book += 1
            bid_uids = [o.uid for o in self._orders.values() if o.side == "buy"  and o.price == bb]
            ask_uids = [o.uid for o in self._orders.values() if o.side == "sell" and o.price == ba]
            self.stats.crossing_details.append({
                "ts_ms": ts_ms, "best_bid": bb, "best_ask": ba,
                "cross_width": bb - ba, "bid_uids": bid_uids, "ask_uids": ask_uids,
            })
        elif bb == ba:
            # Locked market: zero spread — valid observable state, informational only
            self.stats.locked_book += 1


    # ── Snapshots ─────────────────────────────────────────────────────────────

    def price_level_snapshot(self, ts_ms: int) -> pd.DataFrame:
        """Return per-price-level aggregation at current book state."""
        from collections import defaultdict
        levels: dict[tuple, list] = defaultdict(list)
        for o in self._orders.values():
            levels[(o.side, o.price)].append(o)
        rows = []
        for (side, price), orders in levels.items():
            ages = [(ts_ms - o.placed_ms) for o in orders if o.placed_ms is not None]
            rows.append({
                "ts_ms": ts_ms, "side": side, "price": price,
                "bid_qty" if side == "buy" else "ask_qty": sum(o.qty for o in orders),
                "n_orders": len(orders),
                "mean_order_size": sum(o.qty for o in orders) / len(orders),
                "oldest_age_ms": max(ages) if ages else None,
                "newest_age_ms": min(ages) if ages else None,
                "n_known_age": len(ages),
                "n_censored_age": len(orders) - len(ages),
            })
        return pd.DataFrame(rows).sort_values(["side", "price"], ascending=[True, False])

    def bbo_snapshot(self) -> dict:
        """Best bid/offer + spread at current book state."""
        bids = [(o.price, o.qty) for o in self._orders.values() if o.side == "buy" and o.price > 0]
        asks = [(o.price, o.qty) for o in self._orders.values() if o.side == "sell" and o.price > 0]
        best_bid = max(bids, key=lambda x: x[0]) if bids else (None, None)
        best_ask = min(asks, key=lambda x: x[0]) if asks else (None, None)
        spread = None
        if best_bid[0] and best_ask[0]:
            spread = best_ask[0] - best_bid[0]
        return {
            "best_bid": best_bid[0], "best_bid_qty": best_bid[1],
            "best_ask": best_ask[0], "best_ask_qty": best_ask[1],
            "spread": spread,
            "mid": (best_bid[0] + best_ask[0]) / 2 if spread is not None else None,
            "n_active": len(self._orders),
        }

    def depth_bps(self, bps: float) -> dict:
        """Sum bid/ask quantity within `bps` basis points of mid."""
        snap = self.bbo_snapshot()
        mid = snap.get("mid")
        if mid is None:
            return {"depth_bid": None, "depth_ask": None, "obi": None, "bps": bps}
        lo = mid * (1 - bps / 10000)
        hi = mid * (1 + bps / 10000)
        bid_qty = sum(o.qty for o in self._orders.values()
                      if o.side == "buy" and lo <= o.price <= mid)
        ask_qty = sum(o.qty for o in self._orders.values()
                      if o.side == "sell" and mid <= o.price <= hi)
        obi = (bid_qty - ask_qty) / (bid_qty + ask_qty) if (bid_qty + ask_qty) > 0 else 0.0
        return {"depth_bid": bid_qty, "depth_ask": ask_qty, "obi": round(obi, 4), "bps": bps}

    def near_bbo_completeness(self, n_levels: int = 10) -> dict:
        """Primary QA gate: what fraction of near-BBO quantity has known origin?

        An order has known origin if placed_ms is not None (i.e., its OrderPlaced
        event was observed during the warmup or research window, not left-censored).

        Returns known_qty_pct for BBO, L5, L10, and within 10/25 bps.
        """
        snap = self.bbo_snapshot()
        mid = snap.get("mid")

        def _completeness(orders: list) -> dict:
            total = sum(o.qty for o in orders)
            known = sum(o.qty for o in orders if o.placed_ms is not None)
            return {
                "total_qty": total,
                "known_qty": known,
                "pct_known": round(known / total * 100, 2) if total > 0 else None,
            }

        bid_orders = sorted(
            [o for o in self._orders.values() if o.side == "buy" and o.price > 0],
            key=lambda o: -o.price,
        )
        ask_orders = sorted(
            [o for o in self._orders.values() if o.side == "sell" and o.price > 0],
            key=lambda o: o.price,
        )

        # Best N price levels
        def _top_levels(orders, n):
            if not orders:
                return []
            prices = []
            for o in orders:
                if o.price not in prices:
                    prices.append(o.price)
                if len(prices) >= n:
                    break
            return [o for o in orders if o.price in set(prices)]

        result: dict = {}
        for side_name, orders in [("bid", bid_orders), ("ask", ask_orders)]:
            result[f"{side_name}_L1"] = _completeness(_top_levels(orders, 1))
            result[f"{side_name}_L5"] = _completeness(_top_levels(orders, 5))
            result[f"{side_name}_L10"] = _completeness(_top_levels(orders, n_levels))
            if mid:
                for bps in (10, 25):
                    lo = mid * (1 - bps / 10000)
                    hi = mid * (1 + bps / 10000)
                    near = [o for o in orders
                            if (side_name == "bid" and lo <= o.price <= mid) or
                               (side_name == "ask" and mid <= o.price <= hi)]
                    result[f"{side_name}_{bps}bps"] = _completeness(near)

        return result
