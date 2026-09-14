"""aggressive_flow.py — R1 Part B: aggressor atoms & causal windowed aggressive flow.

SIGN CONVENTION (unit-tested, frozen):
    BUY  TAKER = takerOrder.direction == "Buy"  -> consumes ASK liquidity, upward pressure  (+)
    SELL TAKER = takerOrder.direction == "Sell" -> consumes BID liquidity, downward pressure (-)

Raw atoms are mandatory (never keep only the ratio). AFI is a derived summary.
All windows are strictly causal/backward (events with ts <= anchor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
WINDOWS_MS = {"100ms": 100, "250ms": 250, "500ms": 500, "1s": 1000, "2s": 2000,
              "5s": 5000, "10s": 10000, "30s": 30000, "60s": 60000, "300s": 300000}


def build_taker_atoms(raw_execs: list[dict]) -> pd.DataFrame:
    """One row per execution leg with aggressor side. `notional` = Kraken usdValue when present."""
    rows = []
    for e in raw_execs:
        ex = e.get("event", {}).get("Execution", {}).get("execution", {})
        taker = ex.get("takerOrder", {})
        direction = (taker.get("direction") or "")
        side = "buy_taker" if direction == "Buy" else "sell_taker" if direction == "Sell" else None
        if side is None:
            continue
        qty = _f(ex.get("quantity"))
        price = _f(ex.get("price"))
        notional = _f(ex.get("usdValue"))
        if notional is None and qty is not None and price is not None:
            notional = qty * price
        rows.append({
            "ts": e.get("timestamp"), "taker_uid": taker.get("uid"),
            "aggressor": side, "sign": 1 if side == "buy_taker" else -1,
            "qty": qty or 0.0, "price": price, "notional": notional or 0.0,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


def windowed_aggressive_flow(atoms: pd.DataFrame, anchors: np.ndarray,
                             windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Causal backward per-anchor aggressive-flow features.

    For each anchor t0 and window W: sums over atoms with (t0 - W) < ts <= t0.
    Returns raw atoms (buy/sell qty, count, notional, levels) + AFI per window.
    `levels`: distinct price levels swept by takers of that side (proxy for depth consumed).
    """
    windows = windows or WINDOWS_MS
    anchors = np.asarray(anchors, dtype=np.int64)
    out = {"anchor_ts": anchors}
    if len(atoms) == 0:
        for wl in windows:
            for c in ("buy_taker_qty", "sell_taker_qty", "buy_taker_count", "sell_taker_count",
                      "buy_notional", "sell_notional", "ask_levels_consumed", "bid_levels_consumed", "afi"):
                out[f"{c}_{wl}"] = np.zeros(len(anchors))
        return pd.DataFrame(out)

    ts = atoms["ts"].to_numpy(np.int64)
    is_buy = (atoms["aggressor"].to_numpy() == "buy_taker")
    qty = atoms["qty"].to_numpy(np.float64)
    notl = atoms["notional"].to_numpy(np.float64)
    price = atoms["price"].to_numpy(np.float64)

    def csum(v):
        c = np.zeros(len(v) + 1, np.float64); np.cumsum(v, out=c[1:]); return c
    c_buy_q = csum(np.where(is_buy, qty, 0.0)); c_sell_q = csum(np.where(~is_buy, qty, 0.0))
    c_buy_c = csum(is_buy.astype(np.float64)); c_sell_c = csum((~is_buy).astype(np.float64))
    c_buy_n = csum(np.where(is_buy, notl, 0.0)); c_sell_n = csum(np.where(~is_buy, notl, 0.0))

    def win(c, W):
        r = np.searchsorted(ts, anchors, side="right"); l = np.searchsorted(ts, anchors - W, side="right")
        return c[r] - c[l]

    # distinct price levels swept per side per window (loop windows; sparse -> ok)
    buy_prices = price.copy(); buy_prices[~is_buy] = np.nan
    sell_prices = price.copy(); sell_prices[is_buy] = np.nan

    for wl, W in windows.items():
        bq = win(c_buy_q, W); sq = win(c_sell_q, W)
        out[f"buy_taker_qty_{wl}"] = bq
        out[f"sell_taker_qty_{wl}"] = sq
        out[f"buy_taker_count_{wl}"] = win(c_buy_c, W)
        out[f"sell_taker_count_{wl}"] = win(c_sell_c, W)
        out[f"buy_notional_{wl}"] = win(c_buy_n, W)
        out[f"sell_notional_{wl}"] = win(c_sell_n, W)
        out[f"afi_{wl}"] = (bq - sq) / (bq + sq + EPS)
        # levels consumed: distinct prices per side in window
        rgt = np.searchsorted(ts, anchors, side="right"); lft = np.searchsorted(ts, anchors - W, side="right")
        al = np.zeros(len(anchors)); bl = np.zeros(len(anchors))
        for k in range(len(anchors)):
            sl = slice(lft[k], rgt[k])
            if rgt[k] > lft[k]:
                al[k] = len(np.unique(buy_prices[sl][~np.isnan(buy_prices[sl])]))
                bl[k] = len(np.unique(sell_prices[sl][~np.isnan(sell_prices[sl])]))
        out[f"ask_levels_consumed_{wl}"] = al   # buy takers eat ASK levels
        out[f"bid_levels_consumed_{wl}"] = bl   # sell takers eat BID levels
    return pd.DataFrame(out)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
