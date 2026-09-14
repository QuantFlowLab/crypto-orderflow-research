"""investigate_crossings.py — Micro-trace and classification of crossed batches.

For each crossing batch: events in batch, events in adjacent ms, full lifecycle of
crossing UIDs, and the cross-stream timestamp ambiguity classification.

Usage:
  python investigate_crossings.py                      # BTC LOW default
  python investigate_crossings.py --suffix _S_MEDIAN   # other session
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).parent
sys.path.insert(0, str(EXP / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from orderflow import build_event_table
from book_state import OrderBook
from crossing_audit import classify_crossings, MAX_LOOK_AHEAD_MS

RAW = EXP / "data" / "raw"


def load(sym: str, suffix: str):
    with open(RAW / f"{sym}_orders{suffix}.jsonl",     encoding="utf-8") as f:
        ro = [json.loads(l) for l in f]
    with open(RAW / f"{sym}_executions{suffix}.jsonl", encoding="utf-8") as f:
        re = [json.loads(l) for l in f]
    return ro, re


def lifecycle(event_df: pd.DataFrame, uid: str) -> pd.DataFrame:
    """All events where uid appears as order_uid, maker_uid, or taker_uid."""
    mask = ((event_df.order_uid  == uid) |
            (event_df.maker_uid  == uid) |
            (event_df.taker_uid  == uid))
    cols = ["timestamp_ms", "event_type", "order_uid", "side", "price", "qty",
            "maker_uid", "taker_uid", "execution_qty", "execution_price"]
    return event_df.loc[mask, [c for c in cols if c in event_df.columns]]


def window(event_df: pd.DataFrame, ts_ms: int, half_width_ms: int = 5) -> pd.DataFrame:
    lo, hi = ts_ms - half_width_ms, ts_ms + half_width_ms
    sub = event_df[(event_df.timestamp_ms >= lo) & (event_df.timestamp_ms <= hi)].copy()
    cols = ["timestamp_ms", "event_type", "order_uid", "side", "price", "qty",
            "maker_uid", "taker_uid", "execution_qty"]
    return sub[[c for c in cols if c in sub.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="PF_XBTUSD")
    ap.add_argument("--suffix", default="_S_LOW")
    ap.add_argument("--tick",   type=float, default=1.0)
    args = ap.parse_args()

    print(f"Loading {args.symbol}{args.suffix} ...", flush=True)
    ro, re = load(args.symbol, args.suffix)
    ev = build_event_table(ro, re)
    print(f"  {len(ev)} events", flush=True)

    book = OrderBook()
    book.replay(ev)
    details = book.stats.crossing_details
    print(f"\ncrossed_book raw = {book.stats.crossed_book}", flush=True)

    if not details:
        print("No crossings found — book reconstruction is clean.")
        return

    classified = classify_crossings(details, ev, max_look_ahead_ms=MAX_LOOK_AHEAD_MS)

    SEP  = "=" * 72
    SEP2 = "-" * 72

    for k, c in enumerate(classified, 1):
        ts      = c["ts_ms"]
        bb, ba  = c["best_bid"], c["best_ask"]
        w_ticks = (bb - ba) / args.tick
        expl    = c["explanation"]
        res_ts  = c.get("resolving_exec_ts_ms")
        delay   = c.get("resolution_delay_ms")
        pers    = c["persistent"]

        print(f"\n{SEP}")
        print(f"CROSSING {k}/{len(classified)}")
        print(f"  ts_ms          = {ts}")
        print(f"  best_bid       = {bb}  (at crossing)")
        print(f"  best_ask       = {ba}  (at crossing)")
        print(f"  cross_width    = {w_ticks:.1f} ticks")
        print(f"  explanation    = {expl}")
        print(f"  persistent     = {pers}")
        if res_ts:
            print(f"  resolving_exec = {res_ts}  (+{delay}ms)")
        print(SEP2)

        # Events in the crossing batch
        print(f"\n[Events in crossing batch  ts={ts}]")
        batch_df = ev[ev.timestamp_ms == ts]
        print(batch_df[["event_type", "order_uid", "side", "price", "qty",
                         "maker_uid", "taker_uid", "execution_qty"]].to_string(index=False))

        # Events in next 1..3ms
        for dt in range(1, MAX_LOOK_AHEAD_MS + 1):
            nb = ev[ev.timestamp_ms == ts + dt]
            if len(nb):
                print(f"\n[Events at ts+{dt}ms  ts={ts+dt}]")
                print(nb[["event_type", "order_uid", "side", "price", "qty",
                           "maker_uid", "taker_uid", "execution_qty"]].to_string(index=False))

        # Full ±5ms context
        print(f"\n[±5ms window around ts={ts}]")
        print(window(ev, ts, 5)[["timestamp_ms", "event_type", "order_uid", "side",
                                  "price", "qty"]].to_string(index=False))

        # Lifecycle of crossing UIDs
        all_uids = list(set(c["bid_uids"]) | set(c["ask_uids"]))
        for uid in all_uids[:6]:  # cap at 6 for readability
            lc = lifecycle(ev, uid)
            print(f"\n[Lifecycle of uid={uid[:16]}...]")
            print(lc.to_string(index=False) if len(lc) else "  (not found in event_df)")

    # Summary
    print(f"\n{SEP}")
    print("CLASSIFICATION SUMMARY")
    print(SEP2)
    print(f"  total crossings      : {len(classified)}")
    expl_n = sum(1 for c in classified if c["explanation"] == "CROSS_STREAM_TS_AMBIGUITY")
    unex_n = len(classified) - expl_n
    pers_n = sum(1 for c in classified if c["persistent"])
    print(f"  explained (ambiguity): {expl_n}")
    print(f"  unexplained          : {unex_n}")
    print(f"  persistent           : {pers_n}")
    print()
    if unex_n == 0 and pers_n == 0:
        print("  VERDICT: unexplained_crossed_batches=0  persistent=0  → QA PASS")
    else:
        print("  VERDICT: UNEXPLAINED OR PERSISTENT CROSSINGS → QA HOLD / FAIL")
    print(SEP)


if __name__ == "__main__":
    main()
