"""crossing_audit.py — Classify crossed batches as explained or unexplained.

EXPLAINED = CROSS_STREAM_TS_AMBIGUITY:
  The /orders and /executions streams share millisecond-resolution timestamps,
  but the matching engine's atomic transition may be split across adjacent API ms groups.
  Pattern: aggressive/crossing order observed at ts_ms; resolving execution observed at ts_ms+N
  (N in 1..MAX_LOOK_AHEAD_MS). The crossing lasts exactly one ms batch, then is gone.

UNEXPLAINED:
  No resolving execution within the look-ahead window, OR the crossing persists more than
  one ms. Requires further investigation; blocks the feature pipeline.

Hard stop criterion: unexplained_crossed_batches == 0 AND persistent_crossed_batches == 0.
"""
from __future__ import annotations

import pandas as pd

MAX_LOOK_AHEAD_MS = 3  # search up to 3ms ahead for a resolving execution


def classify_crossings(crossing_details: list[dict], event_df: pd.DataFrame, *,
                       max_look_ahead_ms: int = MAX_LOOK_AHEAD_MS) -> list[dict]:
    """Classify each crossing batch.

    crossing_details: list of dicts from BookStats.crossing_details
      {ts_ms, best_bid, best_ask, cross_width, bid_uids, ask_uids}
    event_df: canonical event table from orderflow.build_event_table
    """
    if not crossing_details:
        return []

    # Index executions by ts_ms for fast lookup
    exec_df = event_df[event_df.event_type == "Execution"][
        ["timestamp_ms", "maker_uid", "taker_uid"]].copy()
    exec_by_ts: dict[int, set[str]] = {}
    for _, row in exec_df.iterrows():
        ts = int(row.timestamp_ms) if not pd.isna(row.timestamp_ms) else None
        if ts is None:
            continue
        uids = exec_by_ts.setdefault(ts, set())
        if not pd.isna(row.maker_uid):
            uids.add(str(row.maker_uid))
        if not pd.isna(row.taker_uid):
            uids.add(str(row.taker_uid))

    crossing_ts_set = {d["ts_ms"] for d in crossing_details}
    results = []
    for detail in crossing_details:
        ts = detail["ts_ms"]
        cross_width = detail.get("cross_width", 0)
        cross_uids = set(detail["bid_uids"]) | set(detail["ask_uids"])

        # Locked market (cross_width == 0) resolves via cancel, not necessarily execution
        if cross_width == 0:
            results.append({
                **detail,
                "explanation": "LOCKED_MARKET",
                "resolving_exec_ts_ms": None,
                "resolution_delay_ms": None,
                "persistent": (ts + 1) in crossing_ts_set,
            })
            continue

        # True cross (cross_width > 0): search for resolving execution in adjacent ms
        resolved_at: int | None = None
        for dt in range(1, max_look_ahead_ms + 1):
            if cross_uids & exec_by_ts.get(ts + dt, set()):
                resolved_at = ts + dt
                break

        persistent = (ts + 1) in crossing_ts_set

        results.append({
            **detail,
            "explanation": "CROSS_STREAM_TS_AMBIGUITY" if resolved_at is not None else "UNEXPLAINED",
            "resolving_exec_ts_ms": resolved_at,
            "resolution_delay_ms": (resolved_at - ts) if resolved_at else None,
            "persistent": persistent,
        })
    return results


def summarize(classified: list[dict]) -> dict:
    total  = len(classified)
    locked = sum(1 for c in classified if c["explanation"] == "LOCKED_MARKET")
    expl   = sum(1 for c in classified if c["explanation"] == "CROSS_STREAM_TS_AMBIGUITY")
    unexpl = sum(1 for c in classified if c["explanation"] == "UNEXPLAINED")
    pers   = sum(1 for c in classified if c["persistent"] and c["explanation"] != "LOCKED_MARKET")
    return {"crossed_raw": total, "crossed_locked": locked,
            "crossed_explained": expl, "crossed_unexplained": unexpl,
            "crossed_persistent": pers}
