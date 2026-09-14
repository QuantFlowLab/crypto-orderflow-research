"""market_regime.py — R1 Part A: causal-backward market-activity & volatility regime.

Everything is strictly backward-looking (window (t-W, t]). Relative context (median /
percentile / robust-z) over trailing 5m/30m/120m returns explicit NaN (not_ready) when
trailing history is insufficient — the lookback is NEVER silently shortened.

Categorical states QUIET/NORMAL/ACTIVE/BURST/EXTREME are a *methodological* mapping of a
trailing activity percentile — frozen bands, NOT tuned on any future price behaviour.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from aggressive_flow import WINDOWS_MS

CONTEXT_WINDOWS = {"5m": 300_000, "30m": 1_800_000, "120m": 7_200_000}

# Frozen percentile bands (lower edge, inclusive). Methodological, not price-tuned.
REGIME_BANDS = [("QUIET", 0.0), ("NORMAL", 0.20), ("ACTIVE", 0.70),
                ("BURST", 0.90), ("EXTREME", 0.98)]
NOT_READY = "not_ready"


def build_regime_features(event_df: pd.DataFrame, taker_atoms: pd.DataFrame,
                          anchors: np.ndarray, *, windows: dict[str, int] | None = None,
                          context_windows: dict[str, int] | None = None,
                          base_activity_window: str = "1s",
                          regime_context: str = "30m",
                          book_series: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-anchor causal regime features.

    event_df   : canonical table (orderflow.build_event_table).
    taker_atoms: aggressive_flow.build_taker_atoms(raw_execs) — supplies aggressor side.
    anchors    : decision timestamps (ms). Assumed regular grid for relative context.
    book_series: optional DataFrame [ts, spread, near_book_depth, mid] for book-derived
                 features (causal last-value; bbo_change_count from mid changes).
    """
    windows = windows or WINDOWS_MS
    context_windows = context_windows or CONTEXT_WINDOWS
    anchors = np.sort(np.asarray(anchors, dtype=np.int64))
    out: dict[str, np.ndarray] = {"anchor_ts": anchors}

    ev = event_df.dropna(subset=["timestamp_ms"]).copy()
    ev = ev.sort_values("timestamp_ms", kind="stable")
    ts = ev["timestamp_ms"].astype("int64").to_numpy()
    etype = ev["event_type"].to_numpy()
    placed = (etype == "OrderPlaced"); cancel = (etype == "OrderCancelled")
    update = (etype == "OrderUpdated"); execu = (etype == "Execution")
    exq = pd.to_numeric(ev["execution_qty"], errors="coerce").fillna(0.0).to_numpy()
    exp = pd.to_numeric(ev["execution_price"], errors="coerce").fillna(0.0).to_numpy()
    exec_notional = exq * exp
    uid_codes = pd.factorize(ev["order_uid"], use_na_sentinel=True)[0]
    price_codes = pd.factorize(ev["price"], use_na_sentinel=True)[0]

    # taker atoms (may be empty)
    if taker_atoms is not None and len(taker_atoms):
        tk = taker_atoms.sort_values("ts", kind="stable")
        tts = tk["ts"].astype("int64").to_numpy()
        t_is_buy = (tk["aggressor"].to_numpy() == "buy_taker")
        tqty = tk["qty"].to_numpy(np.float64)
    else:
        tts = np.empty(0, np.int64); t_is_buy = np.empty(0, bool); tqty = np.empty(0)

    def cs(mask_or_vals):
        v = np.asarray(mask_or_vals, np.float64)
        c = np.zeros(len(v) + 1, np.float64); np.cumsum(v, out=c[1:]); return c

    c_all = cs(np.ones(len(ts))); c_pl = cs(placed); c_cx = cs(cancel)
    c_up = cs(update); c_ex = cs(execu); c_exq = cs(exq); c_exn = cs(exec_notional)
    c_tbc = cs(t_is_buy.astype(float)); c_tsc = cs((~t_is_buy).astype(float))
    c_tbq = cs(np.where(t_is_buy, tqty, 0.0)); c_tsq = cs(np.where(~t_is_buy, tqty, 0.0))

    def win(c, arr_ts, W):
        r = np.searchsorted(arr_ts, anchors, side="right")
        l = np.searchsorted(arr_ts, anchors - W, side="right")
        return c[r] - c[l]

    base_key = None
    for wl, W in windows.items():
        sec = W / 1000.0
        oec = win(c_all, ts, W)
        out[f"order_event_count_{wl}"] = oec
        out[f"placed_count_{wl}"] = win(c_pl, ts, W)
        out[f"cancel_count_{wl}"] = win(c_cx, ts, W)
        out[f"update_count_{wl}"] = win(c_up, ts, W)
        ec = win(c_ex, ts, W); eqv = win(c_exq, ts, W); env = win(c_exn, ts, W)
        out[f"execution_count_{wl}"] = ec
        out[f"execution_qty_{wl}"] = eqv
        out[f"execution_notional_{wl}"] = env
        out[f"buy_taker_count_{wl}"] = win(c_tbc, tts, W)
        out[f"sell_taker_count_{wl}"] = win(c_tsc, tts, W)
        out[f"buy_taker_qty_{wl}"] = win(c_tbq, tts, W)
        out[f"sell_taker_qty_{wl}"] = win(c_tsq, tts, W)
        out[f"unique_order_uids_{wl}"] = _sliding_distinct(ts, uid_codes, anchors, W)
        out[f"levels_touched_{wl}"] = _sliding_distinct(ts, price_codes, anchors, W)
        out[f"events_per_sec_{wl}"] = oec / sec
        out[f"trades_per_sec_{wl}"] = ec / sec
        out[f"qty_per_sec_{wl}"] = eqv / sec
        out[f"notional_per_sec_{wl}"] = env / sec
        if wl == base_activity_window:
            base_key = f"events_per_sec_{wl}"

    # book-derived (optional, causal)
    if book_series is not None and len(book_series):
        bs = book_series.dropna(subset=["ts"]).sort_values("ts")
        bts = bs["ts"].astype("int64").to_numpy()
        idx = np.searchsorted(bts, anchors, side="right") - 1  # last obs at or before anchor
        valid = idx >= 0
        for col in ("spread", "near_book_depth"):
            if col in bs.columns:
                vals = bs[col].to_numpy(np.float64)
                res = np.full(len(anchors), np.nan)
                res[valid] = vals[idx[valid]]
                out[col] = res
        if "mid" in bs.columns:
            mid = bs["mid"].to_numpy(np.float64)
            chg_mask = np.ones(len(mid), bool)
            chg_mask[1:] = mid[1:] != mid[:-1]
            c_chg = cs(chg_mask.astype(float))
            for wl, W in windows.items():
                out[f"bbo_change_count_{wl}"] = win(c_chg, bts, W)

    df = pd.DataFrame(out)

    # ── Relative context on base activity (explicit not_ready via min_periods) ──
    if base_key is None:
        base_key = "events_per_sec_1s" if "events_per_sec_1s" in df else None
    if base_key is not None:
        base = df[base_key]
        step = int(np.median(np.diff(anchors))) if len(anchors) > 1 else 1
        for cw_label, C in context_windows.items():
            n = max(int(round(C / step)), 1)
            med = base.rolling(n, min_periods=n).median()
            mad = base.rolling(n, min_periods=n).apply(
                lambda a: np.median(np.abs(a - np.median(a))), raw=True)
            pct = base.rolling(n, min_periods=n).apply(
                lambda a: float(np.mean(a <= a[-1])), raw=True)
            df[f"activity_vs_trailing_median_{cw_label}"] = base / med
            df[f"activity_robust_z_{cw_label}"] = (base - med) / (1.4826 * mad)
            df[f"activity_rolling_percentile_{cw_label}"] = pct
        rp = df.get(f"activity_rolling_percentile_{regime_context}")
        df["regime_percentile"] = rp
        df["regime_state"] = rp.map(_regime_label) if rp is not None else NOT_READY
    return df


def _regime_label(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return NOT_READY
    label = REGIME_BANDS[0][0]
    for name, edge in REGIME_BANDS:
        if p >= edge:
            label = name
    return label


def _sliding_distinct(ts: np.ndarray, codes: np.ndarray, anchors: np.ndarray,
                      W: int) -> np.ndarray:
    """Count distinct codes in causal window (anchor-W, anchor]. codes<0 ignored (nulls).

    Two-pointer sweep; O(N + M) since ts and anchors are both ascending.
    """
    n = len(ts); m = len(anchors)
    res = np.zeros(m, np.int64)
    counts: dict[int, int] = defaultdict(int)
    active = 0; left = 0; right = 0
    for k in range(m):
        a = anchors[k]; lo = a - W
        while right < n and ts[right] <= a:
            c = codes[right]
            if c >= 0:
                if counts[c] == 0:
                    active += 1
                counts[c] += 1
            right += 1
        while left < right and ts[left] <= lo:
            c = codes[left]
            if c >= 0:
                counts[c] -= 1
                if counts[c] == 0:
                    active -= 1
            left += 1
        res[k] = active
    return res
