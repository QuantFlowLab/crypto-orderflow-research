"""run_r2.py — KRAKEN-OF-1-R2: Frozen descriptive feature pipeline on real sessions.

Runs all 6 (symbol × label) combinations through frozen modules, saves per-session
feature parquets with provenance fields. No thresholds tuned, no future labels.

Output:  data/derived/features/{sym}_{label}_features.parquet
         data/derived/features/performance_log.csv
"""
from __future__ import annotations

import gc
import json
import sys
import time
import tracemalloc
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).parent
sys.path.insert(0, str(EXP / "src"))
from orderflow import build_event_table
from book_state import OrderBook
from aggressive_flow import (build_taker_atoms, windowed_aggressive_flow,
                              WINDOWS_MS, EPS)
from passive_flow import (build_passive_atoms, windowed_passive_flow,
                          build_reprice_atoms, windowed_reprice_flow)
from price_response import build_price_response, PR_WINDOWS
from market_regime import build_regime_features, CONTEXT_WINDOWS
from wall_state import build_level_atoms
from replenishment import build_refill_pairs

RAW     = EXP / "data" / "raw"
OUT     = EXP / "data" / "derived" / "features"
OUT.mkdir(parents=True, exist_ok=True)

WARMUP_MS            = 60 * 60 * 1000
ANCHOR_STEP_MS       = 1_000              # 1s grid → 3600 anchors/session
TICK                 = {"PF_XBTUSD": 1.0, "PF_ETHUSD": 0.1}
BBO_DEPTH_BPS        = (5, 10, 25)
DATA_VERSION         = "v0.2-orderflow"
FEATURE_VERSION      = "r2.2"   # r2.1 -> r2.2: explicit REPRICE_OUT/IN atoms; ~90% of OrderUpdated were price-only repricings omitted in r2.1
BOOK_SEMANTICS_VERSION = "ms-batch-v1"
SYMBOLS              = ["PF_XBTUSD", "PF_ETHUSD"]

# Replenishment windows (subset of WINDOWS_MS used for summary)
REPL_WINDOWS = {"100ms": 100, "500ms": 500, "1s": 1000, "5s": 5000}
LARGE_SESSION_THRESHOLD = 4_000_000   # events: use streaming path above this


# ── Data loading ──────────────────────────────────────────────────────────────

def _f(v):
    try: return float(v) if v else None
    except (TypeError, ValueError): return None

def _s(o):
    raw = o.get("direction") or o.get("side")
    return raw.lower() if isinstance(raw, str) else None


def _load(sym: str, suffix: str) -> tuple[list, list]:
    with open(RAW / f"{sym}_orders{suffix}.jsonl", encoding="utf-8") as f:
        ro = [json.loads(l) for l in f]
    with open(RAW / f"{sym}_executions{suffix}.jsonl", encoding="utf-8") as f:
        re = [json.loads(l) for l in f]
    return ro, re


def _stream_event_df(orders_path: Path, execs_path: Path) -> pd.DataFrame:
    """Memory-efficient: build event_df directly from files (no list-of-nested-dicts)."""
    rows: list[dict] = []
    with open(orders_path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ev = e.get("event", {}); et = next(iter(ev), None)
            body = ev.get(et, {}) if et else {}
            row: dict = {
                "timestamp_ms": e.get("timestamp"), "event_type": et,
                "event_uid": e.get("uid"), "order_uid": None,
                "side": None, "price": None, "qty": None, "old_qty": None,
                "filled_qty": None, "maker_uid": None, "taker_uid": None,
                "execution_price": None, "execution_qty": None,
                "reason": body.get("reason"),
            }
            if et == "OrderPlaced":
                o = body.get("order", {})
                row.update({"order_uid": o.get("uid"), "side": _s(o),
                             "price": _f(o.get("limitPrice")), "qty": _f(o.get("quantity")),
                             "filled_qty": _f(o.get("filled"))})
            elif et in ("OrderCancelled", "OrderRejected"):
                o = body.get("order", {})
                row.update({"order_uid": o.get("uid"), "side": _s(o),
                             "price": _f(o.get("limitPrice")), "qty": _f(o.get("quantity")),
                             "filled_qty": _f(o.get("filled"))})
            elif et == "OrderUpdated":
                nw = body.get("newOrder", {}); old = body.get("oldOrder", {})
                row.update({"order_uid": nw.get("uid"), "side": _s(nw),
                             "price": _f(nw.get("limitPrice")), "qty": _f(nw.get("quantity")),
                             "filled_qty": _f(nw.get("filled")),
                             "old_qty": _f(old.get("quantity"))})
            rows.append(row)
    with open(execs_path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ex = e.get("event", {}).get("Execution", {}).get("execution", {})
            rows.append({
                "timestamp_ms": e.get("timestamp"), "event_type": "Execution",
                "event_uid": e.get("uid"), "order_uid": None,
                "side": None, "price": None, "qty": None, "old_qty": None, "filled_qty": None,
                "maker_uid": ex.get("makerOrder", {}).get("uid"),
                "taker_uid": ex.get("takerOrder", {}).get("uid"),
                "execution_price": _f(ex.get("price")),
                "execution_qty": _f(ex.get("quantity")), "reason": None,
            })
    df = pd.DataFrame(rows)
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
    df = df.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)
    vc = df["timestamp_ms"].value_counts()
    df["n_events_same_ms"] = df["timestamp_ms"].map(vc)
    df["ordering_ambiguous"] = df["n_events_same_ms"] > 1
    return df


def _stream_taker_atoms(execs_path: Path) -> pd.DataFrame:
    """Build taker_atoms by streaming the executions file."""
    from aggressive_flow import _f as _af
    rows: list[dict] = []
    with open(execs_path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ex = e.get("event", {}).get("Execution", {}).get("execution", {})
            taker = ex.get("takerOrder", {})
            direction = taker.get("direction") or ""
            side = "buy_taker" if direction == "Buy" else "sell_taker" if direction == "Sell" else None
            if side is None: continue
            qty = _af(ex.get("quantity")); price = _af(ex.get("price"))
            notional = _af(ex.get("usdValue")) or ((qty * price) if qty and price else 0.0)
            rows.append({"ts": e.get("timestamp"), "taker_uid": taker.get("uid"),
                          "aggressor": side, "sign": 1 if side == "buy_taker" else -1,
                          "qty": qty or 0.0, "price": price, "notional": notional or 0.0})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


def _stream_reprice_atoms(orders_path: Path) -> pd.DataFrame:
    """Streaming equivalent of build_reprice_atoms (r2.2). Parity-tested."""
    rows = []
    with open(orders_path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ev = e.get("event", {}); et = next(iter(ev), None)
            if et != "OrderUpdated":
                continue
            body = ev.get(et, {})
            ts = e.get("timestamp")
            nw = body.get("newOrder", {}); old = body.get("oldOrder", {})
            new_price = _f(nw.get("limitPrice")); old_price = _f(old.get("limitPrice"))
            if new_price is None or old_price is None:
                continue
            if abs(new_price - old_price) < 1e-9:
                continue
            new_qty = _f(nw.get("quantity")) or 0.0
            old_qty = _f(old.get("quantity")) or 0.0
            raw_side = nw.get("direction") or ""
            book_side = "bid" if raw_side == "Buy" else "ask" if raw_side == "Sell" else None
            if not book_side or ts is None:
                continue
            direction = ("away" if (book_side == "bid" and new_price < old_price) or
                                    (book_side == "ask" and new_price > old_price) else "toward")
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


def _stream_passive_atoms(orders_path: Path) -> pd.DataFrame:
    """Build passive_atoms by streaming the orders file (avoids full in-memory list)."""
    RESTING = {"Post", "Limit"}
    rows: list[dict] = []
    with open(orders_path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ev = e.get("event", {}); et = next(iter(ev), None)
            body = ev.get(et, {}) if et else {}
            ts = e.get("timestamp")
            if et == "OrderPlaced":
                o = body.get("order", {})
                if (o.get("orderType") or "") not in RESTING: continue
                bs = "bid" if o.get("direction") == "Buy" else "ask" if o.get("direction") == "Sell" else None
                if not bs: continue
                qty = max((_f(o.get("quantity")) or 0) - (_f(o.get("filled")) or 0), 0.0)
                if qty > 0:
                    rows.append({"ts": ts, "order_uid": o.get("uid"), "book_side": bs,
                                  "action": "add", "qty": qty, "price": _f(o.get("limitPrice"))})
            elif et == "OrderCancelled":
                o = body.get("order", {})
                bs = "bid" if o.get("direction") == "Buy" else "ask" if o.get("direction") == "Sell" else None
                if not bs: continue
                qty = max((_f(o.get("quantity")) or 0) - (_f(o.get("filled")) or 0), 0.0)
                if qty > 0:
                    rows.append({"ts": ts, "order_uid": o.get("uid"), "book_side": bs,
                                  "action": "cancel", "qty": qty, "price": _f(o.get("limitPrice"))})
            elif et == "OrderUpdated":
                nw = body.get("newOrder", {}); old = body.get("oldOrder", {})
                delta = (_f(nw.get("quantity")) or 0) - (_f(old.get("quantity")) or 0)
                if abs(delta) < 1e-12: continue
                bs = "bid" if nw.get("direction") == "Buy" else "ask" if nw.get("direction") == "Sell" else None
                if not bs: continue
                rows.append({"ts": ts, "order_uid": nw.get("uid"), "book_side": bs,
                              "action": "add" if delta > 0 else "cancel", "qty": abs(delta),
                              "price": _f(nw.get("limitPrice"))})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
        df["ts"] = df["ts"].astype("int64")
    return df


# ── BBO timeseries via incremental replay ─────────────────────────────────────

def _build_bbo_series(warmup_df: pd.DataFrame, research_df: pd.DataFrame,
                      anchors: np.ndarray) -> pd.DataFrame:
    """Replay warmup then research; snapshot BBO+depth at each anchor. O(N_events)."""
    book = OrderBook()
    book.replay(warmup_df)

    res = research_df.sort_values("timestamp_ms").reset_index(drop=True)
    ts_arr = res["timestamp_ms"].to_numpy(np.int64)
    split_pts = np.searchsorted(ts_arr, anchors, side="right")

    rows: list[dict] = []
    prev_idx = 0
    for anchor, end_idx in zip(anchors, split_pts):
        if end_idx > prev_idx:
            book.replay(res.iloc[prev_idx:end_idx])
        prev_idx = end_idx

        snap = book.bbo_snapshot()
        row: dict = {"ts": anchor, "best_bid": snap["best_bid"],
                     "best_ask": snap["best_ask"], "spread": snap["spread"],
                     "mid": snap["mid"], "n_active": snap["n_active"]}
        for bps in BBO_DEPTH_BPS:
            d = book.depth_bps(bps)
            row[f"depth_bid_{bps}bps"] = d["depth_bid"]
            row[f"depth_ask_{bps}bps"] = d["depth_ask"]
            row[f"obi_{bps}bps"] = d["obi"]
        rows.append(row)

    bbo = pd.DataFrame(rows)
    # near_book_depth = combined 10bps depth (used by market_regime)
    bbo["near_book_depth"] = bbo["depth_bid_10bps"].fillna(0) + bbo["depth_ask_10bps"].fillna(0)
    return bbo


# ── Level summary (streaming) ─────────────────────────────────────────────────

def _build_level_summary(level_atoms: pd.DataFrame, anchors: np.ndarray,
                          bbo_series: pd.DataFrame, tick: float) -> pd.DataFrame:
    """Per-anchor level distribution statistics. Streaming O(N_atoms + N_anchors × N_active)."""
    if len(level_atoms) == 0:
        return pd.DataFrame({"timestamp_ms": anchors})

    atoms = level_atoms.sort_values("ts").reset_index(drop=True)
    ts_arr   = atoms["ts"].to_numpy(np.int64)
    side_arr = atoms["side"].to_numpy()
    price_arr= atoms["price"].to_numpy(np.float64)
    act_arr  = atoms["action"].to_numpy()
    qty_arr  = atoms["qty"].to_numpy(np.float64)

    bbo_ts  = bbo_series["ts"].to_numpy(np.int64)
    bbo_mid = bbo_series["mid"].to_numpy(np.float64)

    level_qty: dict[tuple, float] = {}
    atom_idx = 0
    rows: list[dict] = []

    for anchor in anchors:
        while atom_idx < len(atoms) and ts_arr[atom_idx] <= anchor:
            key = (side_arr[atom_idx], price_arr[atom_idx])
            sign = 1.0 if act_arr[atom_idx] == "add" else -1.0
            new_q = max(0.0, level_qty.get(key, 0.0) + sign * qty_arr[atom_idx])
            if new_q > 0:
                level_qty[key] = new_q
            elif key in level_qty:
                del level_qty[key]
            atom_idx += 1

        mid_idx = int(np.searchsorted(bbo_ts, anchor, side="right")) - 1
        mid = bbo_mid[mid_idx] if mid_idx >= 0 else np.nan

        row: dict = {"timestamp_ms": anchor}
        for side_name, side_key in [("bid", "buy"), ("ask", "sell")]:
            data = [(p, v) for (s, p), v in level_qty.items() if s == side_key]
            if not data:
                for f in ("n_levels", "total_qty", "max_qty", "p50_qty", "p75_qty",
                          "p90_qty", "p95_qty", "max_qty_vs_median", "p95_qty_vs_median",
                          "n_above_3x_median", "n_above_5x_median",
                          "max_share", "top1_dist_ticks", "top1_dist_bps"):
                    row[f"{side_name}_{f}"] = np.nan
                continue
            # Sort bid descending, ask ascending (near → far from mid)
            data.sort(key=lambda x: -x[0] if side_name == "bid" else x[0])
            prices = np.array([x[0] for x in data])
            qtys   = np.array([x[1] for x in data])
            med    = np.median(qtys) + EPS
            total  = qtys.sum()
            pct = lambda p: float(np.percentile(qtys, p))
            row[f"{side_name}_n_levels"]         = len(qtys)
            row[f"{side_name}_total_qty"]         = float(total)
            row[f"{side_name}_max_qty"]           = float(qtys.max())
            row[f"{side_name}_p50_qty"]           = pct(50)
            row[f"{side_name}_p75_qty"]           = pct(75)
            row[f"{side_name}_p90_qty"]           = pct(90)
            row[f"{side_name}_p95_qty"]           = pct(95)
            row[f"{side_name}_max_qty_vs_median"] = float(qtys.max() / med)
            row[f"{side_name}_p95_qty_vs_median"] = float((pct(95) if len(qtys) >= 20 else qtys.max()) / med)
            row[f"{side_name}_n_above_3x_median"] = int((qtys > 3 * med).sum())
            row[f"{side_name}_n_above_5x_median"] = int((qtys > 5 * med).sum())
            row[f"{side_name}_max_share"]         = float(qtys.max() / (total + EPS))
            if not np.isnan(mid) and tick > 0:
                dist_ticks = np.abs(prices - mid) / tick
                dist_bps   = np.abs(prices - mid) / mid * 10_000
                row[f"{side_name}_top1_dist_ticks"] = float(dist_ticks[0])
                row[f"{side_name}_top1_dist_bps"]   = float(dist_bps[0])
            else:
                row[f"{side_name}_top1_dist_ticks"] = np.nan
                row[f"{side_name}_top1_dist_bps"]   = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ── Replenishment summary (vectorised) ────────────────────────────────────────

def _build_replenishment_summary(refill_pairs: pd.DataFrame,
                                  anchors: np.ndarray) -> pd.DataFrame:
    """Per-anchor aggregate refill metrics across all levels."""
    if len(refill_pairs) == 0:
        rows = []
        for anchor in anchors:
            r: dict = {"timestamp_ms": anchor}
            for wl in REPL_WINDOWS:
                for f in ("total_exec_qty", "total_refill_qty", "replenishment_ratio", "refill_count"):
                    r[f"{f}_{wl}"] = 0.0
            rows.append(r)
        return pd.DataFrame(rows)

    exec_ts  = refill_pairs["exec_ts"].to_numpy(np.int64)
    exec_qty = refill_pairs["exec_qty"].to_numpy(np.float64)
    add_ts   = refill_pairs["add_ts"].to_numpy(np.int64)
    refill   = refill_pairs["refill_qty"].to_numpy(np.float64)
    delay    = refill_pairs["delay_ms"].to_numpy(np.int64)

    rows = []
    for anchor in anchors:
        row: dict = {"timestamp_ms": anchor}
        for wl, W in REPL_WINDOWS.items():
            m = (exec_ts >= anchor - W) & (exec_ts < anchor) & \
                (add_ts <= anchor) & (delay <= W)
            if m.any():
                total_refill = float(refill[m].sum())
                # dedup exec_ts to avoid counting same exec multiple times
                _, uniq = np.unique(exec_ts[m], return_index=True)
                total_exec = float(exec_qty[m][uniq].sum())
                row[f"total_exec_qty_{wl}"]       = total_exec
                row[f"total_refill_qty_{wl}"]     = total_refill
                row[f"replenishment_ratio_{wl}"]  = total_refill / (total_exec + EPS)
                row[f"refill_count_{wl}"]         = int(m.sum())
            else:
                row[f"total_exec_qty_{wl}"]       = 0.0
                row[f"total_refill_qty_{wl}"]     = 0.0
                row[f"replenishment_ratio_{wl}"]  = 0.0
                row[f"refill_count_{wl}"]         = 0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Derived acceleration features ─────────────────────────────────────────────

def _add_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Flow acceleration: ratio of recent-half to prior-half within causal window.
    NaN when prior window has no flow (ratio is undefined, not a huge EPS artifact)."""
    pairs = [("500ms", "1s"), ("1s", "2s"), ("2s", "5s"), ("5s", "10s")]
    for short_wl, long_wl in pairs:
        for side in ("buy", "sell"):
            sq = f"{side}_taker_qty_{short_wl}"
            lq = f"{side}_taker_qty_{long_wl}"
            if sq in df and lq in df:
                prior = (df[lq] - df[sq]).clip(lower=0)
                accel = df[sq] / (prior + EPS)
                accel = accel.where(prior >= 1e-8, other=np.nan)  # undefined when no prior flow
                df[f"{side}_flow_accel_{long_wl}"] = accel
    return df


# ── Main per-session builder ───────────────────────────────────────────────────

def build_session_features(sym: str, label: str, since_ms: int, before_ms: int) -> tuple:
    suffix      = f"_S_{label}"
    research_ms = since_ms + WARMUP_MS
    anchors     = np.arange(research_ms + ANCHOR_STEP_MS,
                             before_ms + ANCHOR_STEP_MS, ANCHOR_STEP_MS, dtype=np.int64)
    orders_path = RAW / f"{sym}_orders{suffix}.jsonl"
    execs_path  = RAW / f"{sym}_executions{suffix}.jsonl"

    # Detect large sessions (heuristic: >4M events) and use streaming loader
    est_events = orders_path.stat().st_size // 380
    large = est_events > LARGE_SESSION_THRESHOLD
    mode = "streaming" if large else "normal"
    print(f"\n[{sym} {label}] loading ({mode}, ~{est_events//1_000_000:.1f}M events) ...",
          flush=True)

    t0 = time.time()

    if large:
        print(f"  atoms (streaming) ...", flush=True)
        taker_atoms   = _stream_taker_atoms(execs_path)
        passive_atoms = _stream_passive_atoms(orders_path)
        reprice_atoms = _stream_reprice_atoms(orders_path)
        ev = _stream_event_df(orders_path, execs_path)
        gc.collect()
    else:
        ro, re = _load(sym, suffix)
        taker_atoms   = build_taker_atoms(re)
        passive_atoms = build_passive_atoms(ro)
        reprice_atoms = build_reprice_atoms(ro)
        ev = build_event_table(ro, re)
        del ro, re; gc.collect()

    warmup_df   = ev[ev.timestamp_ms <  research_ms]
    research_df = ev[ev.timestamp_ms >= research_ms]
    print(f"  events: warmup={len(warmup_df)} research={len(research_df)} "
          f"anchors={len(anchors)}", flush=True)

    print(f"  level_atoms + refill_pairs ...", flush=True)
    level_atoms  = build_level_atoms(ev)
    refill_pairs = build_refill_pairs(level_atoms)

    print(f"  BBO series ...", flush=True)
    bbo_series = _build_bbo_series(warmup_df, research_df, anchors)

    print(f"  market_regime ...", flush=True)
    regime_df = build_regime_features(
        ev, taker_atoms, anchors,
        context_windows={"5m": 300_000, "30m": 1_800_000},
        regime_context="30m",
        book_series=bbo_series,
    ).rename(columns={"anchor_ts": "timestamp_ms"})

    # Aggressive flow
    print(f"  aggressive_flow ...", flush=True)
    agg_df = windowed_aggressive_flow(taker_atoms, anchors, WINDOWS_MS
                                       ).rename(columns={"anchor_ts": "timestamp_ms"})

    # Passive flow (r2.1: explicit add/cancel/size-delta)
    print(f"  passive_flow ...", flush=True)
    pass_df = windowed_passive_flow(passive_atoms, anchors, WINDOWS_MS
                                    ).rename(columns={"anchor_ts": "timestamp_ms"})

    # Reprice flow (r2.2: REPRICE_OUT/IN with away/toward direction)
    print(f"  reprice_flow ...", flush=True)
    reprice_df = windowed_reprice_flow(reprice_atoms, anchors, WINDOWS_MS
                                       ).rename(columns={"anchor_ts": "timestamp_ms"})

    # Price response
    print(f"  price_response ...", flush=True)
    price_df = build_price_response(
        bbo_series, taker_atoms, anchors,
        tick=TICK[sym], windows=PR_WINDOWS,
    ).rename(columns={"anchor_ts": "timestamp_ms"})

    # Level summary
    print(f"  level_summary ...", flush=True)
    level_df = _build_level_summary(level_atoms, anchors, bbo_series, TICK[sym])

    # Replenishment summary
    print(f"  replenishment_summary ...", flush=True)
    repl_df = _build_replenishment_summary(refill_pairs, anchors)

    # Merge on timestamp_ms
    # Drop columns from regime_df that are computed more fully in dedicated modules
    _regime_drop = (
        [c for c in regime_df.columns if c.startswith(("buy_taker_", "sell_taker_"))]
        + [c for c in regime_df.columns if c in ("spread", "near_book_depth")]
    )
    regime_df = regime_df.drop(columns=[c for c in _regime_drop if c in regime_df.columns])

    dfs = [regime_df, agg_df, pass_df, reprice_df, price_df,
           bbo_series.rename(columns={"ts": "timestamp_ms"}),
           level_df, repl_df]
    features = reduce(lambda a, b: a.merge(b, on="timestamp_ms", how="outer"), dfs)
    # Drop any residual _x/_y suffix columns from unexpected overlaps
    x_y_cols = [c for c in features.columns if c.endswith("_x") or c.endswith("_y")]
    features = features.drop(columns=x_y_cols)
    features = features.sort_values("timestamp_ms").reset_index(drop=True)

    # Acceleration features (derived, no new atoms)
    features = _add_acceleration(features)

    # Provenance
    features["symbol"]                = sym
    features["session_label"]         = label
    features["source_session"]        = f"{sym}{suffix}"
    features["data_version"]          = DATA_VERSION
    features["feature_version"]       = FEATURE_VERSION
    features["book_semantics_version"]= BOOK_SEMANTICS_VERSION

    wall_s = round(time.time() - t0, 1)
    print(f"  done: {len(features)} rows x {features.shape[1]} cols  [{wall_s}s]", flush=True)
    return features, wall_s


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    man = pd.read_csv(EXP / "reports" / "SESSION_MANIFEST.csv", dtype={"selected": str})
    sel = man[man.selected.isin(["LOW", "MEDIAN", "HIGH"])]
    label_order = {"LOW": 0, "MEDIAN": 1, "HIGH": 2}
    sel = sel.sort_values("selected", key=lambda c: c.map(label_order))

    perf_rows = []
    all_features = []
    for _, row in sel.iterrows():
        label     = row["selected"]
        since_ms  = int(pd.Timestamp(row["warmup_start"]).timestamp() * 1000)
        before_ms = int(pd.Timestamp(row["research_end"]).timestamp() * 1000)
        for sym in SYMBOLS:
            out_path = OUT / f"{sym}_{label}_features.parquet"
            if out_path.exists():
                print(f"  SKIP {sym} {label} — parquet exists ({out_path.stat().st_size//1048576}MB)",
                      flush=True)
                continue
            t_wall = time.time()
            features, wall_s = build_session_features(sym, label, since_ms, before_ms)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            out_path = OUT / f"{sym}_{label}_features.parquet"
            features.to_parquet(out_path, index=False, compression="snappy")
            size_mb = round(out_path.stat().st_size / 1_048_576, 1)
            n_rows  = len(features)
            n_events= len(features.get("order_event_count_1s", pd.Series()))  # proxy
            print(f"  saved: {out_path.name}  [{size_mb} MB]", flush=True)

            perf_rows.append({
                "symbol": sym, "label": label,
                "n_rows": n_rows, "n_cols": features.shape[1],
                "parquet_mb": size_mb,
                "wall_s": wall_s,
                "peak_ram_mb": round(peak_bytes / 1_048_576, 1),
                "rows_per_sec": round(n_rows / max(wall_s, 0.1), 1),
            })
            all_features.append(features)

    pd.DataFrame(perf_rows).to_csv(OUT / "performance_log.csv", index=False)
    print("\nPerformance summary:")
    print(pd.DataFrame(perf_rows)[["symbol", "label", "n_rows", "n_cols", "wall_s",
                                    "peak_ram_mb", "parquet_mb"]].to_string(index=False))
    print(f"\nAll features saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
