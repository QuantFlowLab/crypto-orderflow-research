"""replenishment.py — R1 Part F: physical refill tracking after level execution.

'iceberg-like replenishment candidate' NOT 'iceberg' — hidden volume is not directly
observable from public REST data. Ontology decisions (candidate thresholds, naming)
are deferred to post-Hard-QA battle ontology freeze.

CAUSAL CONSTRAINT (frozen): a feature at anchor time t may only include refills whose
add_ts <= t. An execution at t-800ms and its refill at t-200ms is observable at t.
A refill at t+300ms is future and must NEVER appear in features at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFILL_WINDOWS = {"100ms": 100, "500ms": 500, "1s": 1000, "5s": 5000}
_PAIR_COLS = ["exec_ts", "add_ts", "side", "price", "exec_qty", "refill_qty", "delay_ms"]
EPS = 1e-12


def build_refill_pairs(level_atoms: pd.DataFrame,
                       max_window_ms: int = 10_000) -> pd.DataFrame:
    """For each exec_reduce at (side, price), pair with all subsequent add events at the
    same level within max_window_ms. One row per (exec, add) pair.

    Returns: exec_ts, add_ts, side, price, exec_qty, refill_qty, delay_ms
    """
    execs = level_atoms[level_atoms.action == "exec_reduce"]
    adds = level_atoms[level_atoms.action == "add"]

    if len(execs) == 0 or len(adds) == 0:
        return pd.DataFrame(columns=_PAIR_COLS)

    pairs: list[dict] = []
    for (side, price), eg in execs.groupby(["side", "price"]):
        ag = adds[(adds.side == side) & (adds.price == price)]
        if len(ag) == 0:
            continue
        ats = ag["ts"].to_numpy(np.int64)
        aqs = ag["qty"].to_numpy(np.float64)
        for er in eg.itertuples(index=False):
            et = int(er.ts)
            mask = (ats > et) & (ats <= et + max_window_ms)
            for at, aq in zip(ats[mask], aqs[mask]):
                pairs.append({"exec_ts": et, "add_ts": int(at), "side": side,
                              "price": price, "exec_qty": float(er.qty),
                              "refill_qty": float(aq), "delay_ms": int(at) - et})
    if not pairs:
        return pd.DataFrame(columns=_PAIR_COLS)
    df = pd.DataFrame(pairs)
    df[["exec_ts", "add_ts", "delay_ms"]] = df[["exec_ts", "add_ts", "delay_ms"]].astype("int64")
    return df


def level_refill_features(refill_pairs: pd.DataFrame, side: str, price: float,
                          anchor_ts: int, windows: dict[str, int] | None = None) -> dict:
    """Causal per-level refill features at anchor_ts.

    Only pairs where exec_ts < anchor_ts AND add_ts <= anchor_ts are included.
    `replenishment_ratio_{W}` = total refilled / total executed at level in trailing W.
    Exec deduplication: if multiple pairs share the same exec_ts, exec_qty counted once.
    """
    windows = windows or REFILL_WINDOWS
    lp = refill_pairs[
        (refill_pairs.side == side) & (refill_pairs.price == price) &
        (refill_pairs.exec_ts < anchor_ts) & (refill_pairs.add_ts <= anchor_ts)
    ]
    result: dict = {"anchor_ts": anchor_ts, "side": side, "price": price}
    for wl, W in windows.items():
        # refill pairs where: exec within trailing W, add within W of exec, add <= anchor
        in_window = lp[(lp.exec_ts >= anchor_ts - W) & (lp.delay_ms <= W)]
        refill_qty = float(in_window["refill_qty"].sum())
        refill_count = len(in_window)
        delays = in_window["delay_ms"].to_numpy()
        # exec qty: sum unique execs (one exec can produce multiple pairs)
        exec_qty = float(in_window.drop_duplicates("exec_ts")["exec_qty"].sum())
        result[f"refilled_qty_{wl}"] = refill_qty
        result[f"refill_count_{wl}"] = refill_count
        result[f"max_refill_{wl}"] = float(in_window["refill_qty"].max()) if refill_count else 0.0
        result[f"replenishment_ratio_{wl}"] = refill_qty / (exec_qty + EPS) if exec_qty > 0 else 0.0
        result[f"median_refill_delay_{wl}"] = float(np.median(delays)) if len(delays) else None
    return result
