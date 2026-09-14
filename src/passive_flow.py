"""passive_flow.py — R1 Part C / r2.2: passive liquidity atoms & causal windowed flow.

SIDE MAPPING (frozen): resting BUY order == BID; resting SELL order == ASK.

r2.1 atoms (explicit add/cancel/size-delta):
  POST_ADD    = OrderPlaced (Post/Limit) placed quantity
  EXPLICIT_CANCEL = OrderCancelled remaining quantity
  SIZE_DELTA  = OrderUpdated net qty-delta when price is unchanged (>0 add, <0 cancel)

r2.2 additions (price repricing — ~90% of all OrderUpdated events):
  REPRICE_OUT = OrderUpdated with price change: old_qty leaves old price level
  REPRICE_IN  = OrderUpdated with price change: new_qty arrives at new price level
  direction   = "away" (BID price decrease / ASK price increase)
              = "toward" (BID price increase / ASK price decrease)

Each repricing generates exactly two atoms (OUT, IN). Price+qty updates follow the
same rule — REPRICE_OUT carries old_qty, REPRICE_IN carries new_qty; the net qty
difference is implicit. No double-counting with SIZE_DELTA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aggressive_flow import EPS, WINDOWS_MS, _f  # reuse constants/helper

RESTING_TYPES = {"Post", "Limit"}


def build_passive_atoms(raw_orders: list[dict]) -> pd.DataFrame:
    """One row per passive add/cancel atom. `book_side` in {bid, ask}."""
    rows = []
    for e in raw_orders:
        ev = e.get("event", {})
        etype = next(iter(ev), None)
        body = ev.get(etype, {}) if etype else {}
        ts = e.get("timestamp")
        if etype == "OrderPlaced":
            o = body.get("order", {})
            if (o.get("orderType") or "") not in RESTING_TYPES:
                continue
            _emit(rows, ts, o, "add", _placed_qty(o))
        elif etype == "OrderCancelled":
            o = body.get("order", {})
            _emit(rows, ts, o, "cancel", _remaining_qty(o))
        elif etype == "OrderUpdated":
            new = body.get("newOrder", {})
            old = body.get("oldOrder", {})
            new_price = _f(new.get("limitPrice"))
            old_price = _f(old.get("limitPrice"))
            # Price-changing updates are exclusively represented by REPRICE_OUT/IN.
            # Including them here would double-count the qty change.
            if (new_price is not None and old_price is not None
                    and abs(new_price - old_price) >= 1e-9):
                continue
            nq, oq = _f(new.get("quantity")) or 0.0, _f(old.get("quantity")) or 0.0
            delta = nq - oq
            if abs(delta) < 1e-9:
                continue
            _emit(rows, ts, new, "add" if delta > 0 else "cancel", abs(delta))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


def windowed_passive_flow(atoms: pd.DataFrame, anchors: np.ndarray,
                          windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Causal backward per-anchor passive-flow features per book side."""
    windows = windows or WINDOWS_MS
    anchors = np.asarray(anchors, dtype=np.int64)
    out = {"anchor_ts": anchors}
    cols = ("bid_add_qty", "ask_add_qty", "bid_cancel_qty", "ask_cancel_qty",
            "bid_add_orders", "ask_add_orders", "bid_cancel_orders", "ask_cancel_orders",
            "bid_net_passive", "ask_net_passive")
    if len(atoms) == 0:
        for wl in windows:
            for c in cols:
                out[f"{c}_{wl}"] = np.zeros(len(anchors))
        return pd.DataFrame(out)

    ts = atoms["ts"].to_numpy(np.int64)
    is_bid = (atoms["book_side"].to_numpy() == "bid")
    is_add = (atoms["action"].to_numpy() == "add")
    qty = atoms["qty"].to_numpy(np.float64)

    def csum(v):
        c = np.zeros(len(v) + 1, np.float64); np.cumsum(v, out=c[1:]); return c

    def mask_q(m): return csum(np.where(m, qty, 0.0))
    def mask_c(m): return csum(m.astype(np.float64))

    c = {
        "bid_add_qty": mask_q(is_bid & is_add), "ask_add_qty": mask_q(~is_bid & is_add),
        "bid_cancel_qty": mask_q(is_bid & ~is_add), "ask_cancel_qty": mask_q(~is_bid & ~is_add),
        "bid_add_orders": mask_c(is_bid & is_add), "ask_add_orders": mask_c(~is_bid & is_add),
        "bid_cancel_orders": mask_c(is_bid & ~is_add), "ask_cancel_orders": mask_c(~is_bid & ~is_add),
    }

    def win(cc, W):
        r = np.searchsorted(ts, anchors, side="right"); l = np.searchsorted(ts, anchors - W, side="right")
        return cc[r] - cc[l]

    for wl, W in windows.items():
        vals = {k: win(v, W) for k, v in c.items()}
        for k, v in vals.items():
            out[f"{k}_{wl}"] = v
        out[f"bid_net_passive_{wl}"] = vals["bid_add_qty"] - vals["bid_cancel_qty"]
        out[f"ask_net_passive_{wl}"] = vals["ask_add_qty"] - vals["ask_cancel_qty"]
    return pd.DataFrame(out)


def _emit(rows, ts, o, action, qty):
    raw_side = o.get("direction") or ""
    book_side = "bid" if raw_side == "Buy" else "ask" if raw_side == "Sell" else None
    if book_side is None or ts is None:
        return
    rows.append({"ts": ts, "order_uid": o.get("uid"), "book_side": book_side,
                 "action": action, "qty": qty or 0.0, "price": _f(o.get("limitPrice"))})


def _placed_qty(o):
    q = _f(o.get("quantity")) or 0.0
    return max(q - (_f(o.get("filled")) or 0.0), 0.0)


def _remaining_qty(o):
    q = _f(o.get("quantity")) or 0.0
    return max(q - (_f(o.get("filled")) or 0.0), 0.0)


# ── r2.2: Reprice atoms (REPRICE_OUT / REPRICE_IN) ───────────────────────────

def build_reprice_atoms(raw_orders: list[dict]) -> pd.DataFrame:
    """OrderUpdated atoms where price changes (r2.2 addition).

    Each price-changing update emits TWO atoms per event:
      reprice_out: old qty leaves old price level
      reprice_in:  new qty enters new price level
    direction: 'away'   = BID price decreases or ASK price increases (retreat)
               'toward' = BID price increases or ASK price decreases (advance)
    """
    rows = []
    for e in raw_orders:
        ev = e.get("event", {})
        etype = next(iter(ev), None)
        if etype != "OrderUpdated":
            continue
        body = ev.get(etype, {})
        ts = e.get("timestamp")
        nw = body.get("newOrder", {})
        old = body.get("oldOrder", {})
        new_price = _f(nw.get("limitPrice"))
        old_price = _f(old.get("limitPrice"))
        if new_price is None or old_price is None:
            continue
        if abs(new_price - old_price) < 1e-9:
            continue  # price unchanged → handled by build_passive_atoms SIZE_DELTA
        new_qty = (_f(nw.get("quantity")) or 0.0)
        old_qty = (_f(old.get("quantity")) or 0.0)
        raw_side = (nw.get("direction") or "")
        book_side = "bid" if raw_side == "Buy" else "ask" if raw_side == "Sell" else None
        if book_side is None or ts is None:
            continue
        # Direction: away = retreating from market; toward = advancing toward market
        if book_side == "bid":
            direction = "away" if new_price < old_price else "toward"
        else:
            direction = "away" if new_price > old_price else "toward"
        uid = nw.get("uid")
        rows.append({"ts": ts, "order_uid": uid, "book_side": book_side,
                     "action": "reprice_out", "qty": old_qty,
                     "old_price": old_price, "new_price": new_price, "direction": direction})
        rows.append({"ts": ts, "order_uid": uid, "book_side": book_side,
                     "action": "reprice_in",  "qty": new_qty,
                     "old_price": old_price, "new_price": new_price, "direction": direction})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


def windowed_reprice_flow(reprice_atoms: pd.DataFrame, anchors: np.ndarray,
                          windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Causal windowed reprice features. Strictly backward (t-W, t].

    Features per side (bid/ask), per window:
      {side}_reprice_out_qty_{W}    qty leaving old levels
      {side}_reprice_in_qty_{W}     qty arriving at new levels
      {side}_reprice_net_{W}        in - out (net change from repricing)
      {side}_reprice_away_qty_{W}   qty repricing away from market
      {side}_reprice_toward_qty_{W} qty repricing toward market
      {side}_reprice_away_count_{W}
      {side}_reprice_toward_count_{W}
    """
    windows = windows or WINDOWS_MS
    anchors = np.asarray(anchors, dtype=np.int64)
    out = {"anchor_ts": anchors}
    if len(reprice_atoms) == 0:
        for wl in windows:
            for s in ("bid", "ask"):
                for f in ("reprice_out_qty", "reprice_in_qty", "reprice_net",
                          "reprice_away_qty", "reprice_toward_qty",
                          "reprice_away_count", "reprice_toward_count"):
                    out[f"{s}_{f}_{wl}"] = np.zeros(len(anchors))
        return pd.DataFrame(out)

    ts        = reprice_atoms["ts"].to_numpy(np.int64)
    is_bid    = reprice_atoms["book_side"].to_numpy() == "bid"
    is_out    = reprice_atoms["action"].to_numpy() == "reprice_out"
    is_away   = reprice_atoms["direction"].to_numpy() == "away"
    qty       = reprice_atoms["qty"].to_numpy(np.float64)

    def cs(v):
        c = np.zeros(len(v) + 1, np.float64); np.cumsum(v, out=c[1:]); return c

    def win(c_arr, W):
        r = np.searchsorted(ts, anchors, side="right")
        l = np.searchsorted(ts, anchors - W, side="right")
        return c_arr[r] - c_arr[l]

    # Precompute cumulative sums
    cum = {
        "bid_out_q":  cs(np.where(is_bid & is_out,  qty, 0.0)),
        "bid_in_q":   cs(np.where(is_bid & ~is_out, qty, 0.0)),
        "ask_out_q":  cs(np.where(~is_bid & is_out,  qty, 0.0)),
        "ask_in_q":   cs(np.where(~is_bid & ~is_out, qty, 0.0)),
        # away/toward: count only reprice_OUT qty — how much qty left the old level in that direction
        "bid_away_q": cs(np.where(is_bid & is_away & is_out,  qty, 0.0)),
        "bid_tw_q":   cs(np.where(is_bid & ~is_away & is_out, qty, 0.0)),
        "ask_away_q": cs(np.where(~is_bid & is_away & is_out,  qty, 0.0)),
        "ask_tw_q":   cs(np.where(~is_bid & ~is_away & is_out, qty, 0.0)),
        # counts: one per repricing event (counted at reprice_out)
        "bid_away_c": cs((is_bid & is_away & is_out).astype(float)),
        "bid_tw_c":   cs((is_bid & ~is_away & is_out).astype(float)),
        "ask_away_c": cs((~is_bid & is_away & is_out).astype(float)),
        "ask_tw_c":   cs((~is_bid & ~is_away & is_out).astype(float)),
    }

    for wl, W in windows.items():
        for s, pfx in [("bid", "bid"), ("ask", "ask")]:
            oq = win(cum[f"{s}_out_q"], W); iq = win(cum[f"{s}_in_q"], W)
            out[f"{s}_reprice_out_qty_{wl}"]     = oq
            out[f"{s}_reprice_in_qty_{wl}"]      = iq
            out[f"{s}_reprice_net_{wl}"]          = iq - oq
            out[f"{s}_reprice_away_qty_{wl}"]    = win(cum[f"{s}_away_q"], W)
            out[f"{s}_reprice_toward_qty_{wl}"]  = win(cum[f"{s}_tw_q"],   W)
            out[f"{s}_reprice_away_count_{wl}"]  = win(cum[f"{s}_away_c"], W)
            out[f"{s}_reprice_toward_count_{wl}"]= win(cum[f"{s}_tw_c"],   W)
    return pd.DataFrame(out)
