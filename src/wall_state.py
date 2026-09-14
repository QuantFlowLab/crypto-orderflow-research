"""wall_state.py — R1 Part E: concentrated liquidity atoms per price level.

Produces raw continuous atoms per (side, price) level. No binary is_wall label.
`wall_candidate_*` prefixes are acceptable for ranking, but ontology decisions
(what deserves the name 'wall', 'defense', 'absorption') are deferred to the
post-Hard-QA battle ontology freeze.

KEY SEMANTICS (frozen):
  OrderUpdated with price change = REMOVE remaining qty from old level + ADD new qty at
  new level, resolved within the same ms batch. A UID must never appear at two prices.
  OrderUpdated with same price = qty delta only (replacement semantics).

  Episode reset: when a level's running qty reaches zero, the episode ends. The next add
  at the same (side, price) starts a fresh episode_id. This is NOT a continuation.

  Left-censored orders (no prior OrderPlaced observed) are silently skipped — consistent
  with book_state.py unknown_cancel/unknown_update handling.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from book_state import _BATCH_PRIORITY

_ATOM_COLS = ["ts", "side", "price", "action", "qty", "episode_id", "order_uid"]


def build_level_atoms(event_df: pd.DataFrame) -> pd.DataFrame:
    """Sequential pass: one atom per level-change event.

    Same-ms batch priority mirrors book_state (Placed→Updated→Execution→Cancelled/Rejected).
    93% of OrderUpdated events change the price; the sequential `order_price` dict captures
    the old price without modifying the frozen event_df schema.

    Returns DataFrame: ts, side, price, action(add|cancel|exec_reduce), qty, episode_id, order_uid
    """
    ev = event_df.dropna(subset=["timestamp_ms"]).copy()
    ev["_bp"] = ev["event_type"].map(_BATCH_PRIORITY).fillna(9)
    ev = ev.sort_values(["timestamp_ms", "_bp"], kind="stable").reset_index(drop=True)

    order_price: dict[str, float] = {}
    order_qty: dict[str, float] = {}
    order_side: dict[str, str] = {}
    level_rq: dict[tuple, float] = defaultdict(float)   # (side, price) -> running qty
    level_ep: dict[tuple, int] = defaultdict(int)        # (side, price) -> episode id

    rows: list[dict] = []

    def _fv(v):
        if v is None or v is pd.NA:
            return None
        try:
            f = float(v)
            return None if f != f else f
        except (TypeError, ValueError):
            return None

    def _sv(v):
        return None if (v is None or v is pd.NA) else str(v)

    def _add(ts, side, price, qty, uid):
        key = (side, price)
        if level_rq[key] <= 0.0:
            level_ep[key] += 1
        level_rq[key] += qty
        rows.append({"ts": ts, "side": side, "price": price, "action": "add",
                     "qty": qty, "episode_id": level_ep[key], "order_uid": uid})

    def _remove(ts, side, price, qty, uid, action):
        key = (side, price)
        ep = level_ep[key]
        level_rq[key] = max(0.0, level_rq[key] - qty)
        rows.append({"ts": ts, "side": side, "price": price, "action": action,
                     "qty": qty, "episode_id": ep, "order_uid": uid})

    for row in ev.itertuples(index=False):
        ts = int(row.timestamp_ms)
        etype = _sv(row.event_type)
        uid = _sv(row.order_uid) or ""

        if etype == "OrderPlaced":
            side = _sv(row.side)
            price = _fv(row.price)
            qty = _fv(row.qty)
            if uid and side in ("buy", "sell") and price is not None and qty and qty > 0:
                order_price[uid] = price
                order_qty[uid] = qty
                order_side[uid] = side
                _add(ts, side, price, qty, uid)

        elif etype in ("OrderCancelled", "OrderRejected"):
            if uid in order_price:
                remaining = order_qty.get(uid, 0.0)
                if remaining > 0:
                    _remove(ts, order_side[uid], order_price[uid], remaining, uid, "cancel")
                del order_price[uid]
                order_qty.pop(uid, None)
                order_side.pop(uid, None)

        elif etype == "OrderUpdated":
            if uid not in order_price:
                continue  # left-censored
            new_price = _fv(row.price)
            new_qty = _fv(row.qty)
            if new_price is None:
                new_price = order_price[uid]
            if new_qty is None:
                new_qty = order_qty.get(uid, 0.0)
            old_price = order_price[uid]
            old_qty = order_qty.get(uid, 0.0)

            if abs(new_price - old_price) > 1e-9:
                # price change: remove from old level, add to new (within same ms batch)
                if old_qty > 0:
                    _remove(ts, order_side[uid], old_price, old_qty, uid, "cancel")
                if new_qty > 0:
                    _add(ts, order_side[uid], new_price, new_qty, uid)
                order_price[uid] = new_price
                order_qty[uid] = new_qty
            else:
                # same price, replacement qty -> emit delta
                delta = new_qty - old_qty
                if delta > 1e-12:
                    _add(ts, order_side[uid], old_price, delta, uid)
                elif delta < -1e-12:
                    _remove(ts, order_side[uid], old_price, -delta, uid, "cancel")
                order_qty[uid] = new_qty

        elif etype == "Execution":
            m_uid = _sv(row.maker_uid) or ""
            exec_qty = _fv(row.execution_qty)
            if m_uid in order_price and exec_qty and exec_qty > 0:
                _remove(ts, order_side[m_uid], order_price[m_uid], exec_qty, m_uid, "exec_reduce")
                order_qty[m_uid] = max(0.0, order_qty.get(m_uid, 0.0) - exec_qty)
                if order_qty[m_uid] <= 1e-12:
                    del order_price[m_uid]
                    order_qty.pop(m_uid, None)
                    order_side.pop(m_uid, None)

    if not rows:
        return pd.DataFrame(columns=_ATOM_COLS)
    df = pd.DataFrame(rows)
    df["ts"] = df["ts"].astype("int64")
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def level_snapshot(level_atoms: pd.DataFrame, anchor_ts: int, *,
                   tick: float = 1.0, bbo: dict | None = None) -> pd.DataFrame:
    """Current-state snapshot of all active levels at anchor_ts.

    Pure function of level_atoms — strictly causal (ts <= anchor_ts only).
    bbo: dict from OrderBook.bbo_snapshot() for distance/share features.

    Returns one row per active (side, price): level_qty, episode_id, age_ms,
    share_of_side_depth, qty_vs_nearby_median, qty_percentile,
    and optionally distance_ticks, distance_bps.
    """
    past = level_atoms[level_atoms.ts <= anchor_ts].copy()
    if len(past) == 0:
        return pd.DataFrame()

    sign = past["action"].map({"add": 1.0, "cancel": -1.0, "exec_reduce": -1.0}).fillna(0.0)
    past["signed_qty"] = sign * past["qty"]
    lvl = past.groupby(["side", "price"])["signed_qty"].sum().clip(lower=0).reset_index()
    lvl.columns = ["side", "price", "level_qty"]
    active = lvl[lvl.level_qty > 0].copy()
    if len(active) == 0:
        return pd.DataFrame()

    # Current episode: max episode_id per level up to anchor
    cur_ep = (past.groupby(["side", "price"])["episode_id"].max()
              .reset_index().rename(columns={"episode_id": "cur_ep"}))
    active = active.merge(cur_ep, on=["side", "price"])

    # Episode start ts: first atom of the current episode at each level
    past2 = past.merge(cur_ep, on=["side", "price"])
    ep_start = (past2[past2.episode_id == past2.cur_ep]
                .groupby(["side", "price"])["ts"].min()
                .reset_index().rename(columns={"ts": "episode_start_ts"}))
    active = (active.merge(ep_start, on=["side", "price"], how="left")
              .rename(columns={"cur_ep": "episode_id"}))
    active["age_ms"] = anchor_ts - active["episode_start_ts"].fillna(anchor_ts).astype(int)

    # Relative features (within side)
    side_total = active.groupby("side")["level_qty"].transform("sum").clip(lower=1e-12)
    active["share_of_side_depth"] = active["level_qty"] / side_total
    active["qty_percentile"] = active.groupby("side")["level_qty"].transform(
        lambda x: x.rank(pct=True, method="average"))
    active["qty_vs_nearby_median"] = active.groupby("side")["level_qty"].transform(
        lambda x: x / (x.median() + 1e-12))

    # Distance features (require bbo)
    active["distance_ticks"] = None
    active["distance_bps"] = None
    if bbo is not None:
        mid = bbo.get("mid")
        best_bid, best_ask = bbo.get("best_bid"), bbo.get("best_ask")
        if mid and mid > 0:
            active["distance_bps"] = ((active["price"] - mid).abs() / mid * 10000).round(2)
        if best_bid is not None and best_ask is not None and tick > 0:
            ref = active["side"].map({"buy": best_bid, "sell": best_ask})
            active["distance_ticks"] = (active["price"] - ref).abs() / tick

    return active.reset_index(drop=True)
