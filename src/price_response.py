"""price_response.py — R1 Part D: descriptive price response within the SAME causal interval.

CRITICAL: price movement is measured over the same historical window (t-W, t] as the flow —
mid(t) - mid(t-W). This is NOT a forward outcome (never t..t+W). Any forward measurement would
be a future label and belongs to a later alpha study, not this descriptive dataset.

Neutral naming only (no absorption/defense/strength scores here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aggressive_flow import EPS, windowed_aggressive_flow

PR_WINDOWS = {"100ms": 100, "250ms": 250, "500ms": 500, "1s": 1000,
              "2s": 2000, "5s": 5000, "10s": 10000}
TICK = {"PF_XBTUSD": 1.0, "PF_ETHUSD": 0.1}


def build_price_response(book_series: pd.DataFrame, taker_atoms: pd.DataFrame,
                         anchors: np.ndarray, *, tick: float,
                         windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Per-anchor causal price-response features.

    book_series: DataFrame with columns [ts, mid] (mid may be NaN where book incomplete).
    taker_atoms: aggressive_flow.build_taker_atoms(raw_execs).
    tick       : instrument tick size (BTC 1.0, ETH 0.1).
    """
    windows = windows or PR_WINDOWS
    anchors = np.sort(np.asarray(anchors, dtype=np.int64))
    flow = windowed_aggressive_flow(taker_atoms, anchors, windows)
    out: dict[str, np.ndarray] = {"anchor_ts": anchors}

    bts = np.empty(0, np.int64); mid = np.empty(0)
    if book_series is not None and len(book_series):
        bs = book_series.dropna(subset=["ts"]).sort_values("ts")
        bts = bs["ts"].astype("int64").to_numpy()
        mid = bs["mid"].to_numpy(np.float64)

    def mid_at(times: np.ndarray) -> np.ndarray:
        res = np.full(len(times), np.nan)
        if len(bts) == 0:
            return res
        idx = np.searchsorted(bts, times, side="right") - 1
        valid = idx >= 0
        res[valid] = mid[idx[valid]]
        return res

    mid_t = mid_at(anchors)
    for wl, W in windows.items():
        mid_tw = mid_at(anchors - W)
        smc = mid_t - mid_tw                       # signed mid change over (t-W, t]
        ticks = smc / tick
        levels = np.abs(np.round(ticks)).astype(np.float64)
        bq = flow[f"buy_taker_qty_{wl}"].to_numpy()
        sq = flow[f"sell_taker_qty_{wl}"].to_numpy()
        bn = flow[f"buy_notional_{wl}"].to_numpy()
        sn = flow[f"sell_notional_{wl}"].to_numpy()
        out[f"signed_mid_change_{wl}"] = smc
        out[f"ticks_moved_{wl}"] = np.abs(ticks)          # unsigned magnitude; sign is in signed_mid_change
        out[f"levels_crossed_{wl}"] = levels
        # NaN when flow is near-zero: ratio is undefined (no flow to attribute price change to)
        _zero_buy = bq < 1e-8; _zero_sell = sq < 1e-8
        pr_buy = smc / (bq + EPS); pr_buy[_zero_buy] = np.nan
        pr_sell = smc / (sq + EPS); pr_sell[_zero_sell] = np.nan
        tk_buy = ticks / (bn + EPS); tk_buy[_zero_buy] = np.nan
        tk_sell = ticks / (sn + EPS); tk_sell[_zero_sell] = np.nan
        lv_buy = levels / (bq + EPS); lv_buy[_zero_buy] = np.nan
        lv_sell = levels / (sq + EPS); lv_sell[_zero_sell] = np.nan
        out[f"price_response_per_buy_qty_{wl}"] = pr_buy
        out[f"price_response_per_sell_qty_{wl}"] = pr_sell
        out[f"ticks_per_buy_notional_{wl}"] = tk_buy
        out[f"ticks_per_sell_notional_{wl}"] = tk_sell
        out[f"levels_per_buy_qty_{wl}"] = lv_buy
        out[f"levels_per_sell_qty_{wl}"] = lv_sell
    return pd.DataFrame(out)
