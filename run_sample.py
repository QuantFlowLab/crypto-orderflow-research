"""run_sample.py — End-to-end pipeline demonstration using the synthetic sample.

Loads the schema-example files from data/sample/, runs the full reconstruction
and feature pipeline, and prints a summary. Requires no internet connection
and completes in a few seconds.

Usage: python run_sample.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running from repo root without installing as package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orderflow import build_event_table
from book_state import OrderBook
from aggressive_flow import build_taker_atoms, windowed_aggressive_flow, WINDOWS_MS
from passive_flow import (build_passive_atoms, windowed_passive_flow,
                           build_reprice_atoms, windowed_reprice_flow)

import numpy as np

SAMPLE = Path(__file__).parent / "data" / "sample"


def load_sample():
    orders_path = SAMPLE / "example_orders.jsonl"
    execs_path  = SAMPLE / "example_executions.jsonl"
    if not orders_path.exists() or not execs_path.exists():
        print("ERROR: data/sample/ files not found. "
              "Ensure you have a complete checkout of the repository.")
        sys.exit(1)
    with open(orders_path, encoding="utf-8") as f:
        ro = [json.loads(l) for l in f]
    with open(execs_path, encoding="utf-8") as f:
        re = [json.loads(l) for l in f]
    return ro, re


def run():
    print("Crypto Order Flow Research — Sample Pipeline Demo", flush=True)
    print("=" * 55, flush=True)

    t0 = time.time()

    # 1. Load synthetic data
    ro, re = load_sample()
    print(f"\n[1] Loaded sample: {len(ro)} order events, {len(re)} executions", flush=True)

    # 2. Build canonical event table
    ev = build_event_table(ro, re)
    print(f"\n[2] Event table: {len(ev)} rows", flush=True)
    print(f"    Event types: {ev.event_type.value_counts().to_dict()}", flush=True)
    print(f"    Timestamp range: {int(ev.timestamp_ms.min())} .. {int(ev.timestamp_ms.max())}", flush=True)
    pct_ambiguous = ev.ordering_ambiguous.mean() * 100
    print(f"    Same-ms ambiguous batches: {pct_ambiguous:.1f}%", flush=True)

    # 3. Replay book state
    book = OrderBook()
    book.replay(ev)
    snap = book.bbo_snapshot()
    print(f"\n[3] Book state after replay:", flush=True)
    print(f"    Active orders:  {snap['n_active']}", flush=True)
    print(f"    Best bid:       {snap['best_bid']}", flush=True)
    print(f"    Best ask:       {snap['best_ask']}", flush=True)
    print(f"    Spread:         {snap['spread']}", flush=True)
    print(f"    QA — crossed_book: {book.stats.crossed_book}", flush=True)
    print(f"    QA — negative_qty: {book.stats.negative_qty}", flush=True)
    print(f"    QA — dup_active:   {book.stats.duplicate_active_uid}", flush=True)

    # 4. Build feature atoms (r2.2: passive + reprice)
    taker_atoms   = build_taker_atoms(re)
    passive_atoms = build_passive_atoms(ro)
    reprice_atoms = build_reprice_atoms(ro)
    print(f"\n[4] Feature atoms (r2.2):", flush=True)
    print(f"    Taker atoms:   {len(taker_atoms)} (buy: {(taker_atoms.aggressor=='buy_taker').sum()}, "
          f"sell: {(taker_atoms.aggressor=='sell_taker').sum()})", flush=True)
    print(f"    Passive atoms: {len(passive_atoms)} "
          f"(add: {(passive_atoms.action=='add').sum()}, cancel: {(passive_atoms.action=='cancel').sum()})",
          flush=True)
    if len(reprice_atoms):
        n_out = (reprice_atoms.action == "reprice_out").sum()
        n_in  = (reprice_atoms.action == "reprice_in").sum()
        n_away   = (reprice_atoms.direction == "away").sum()
        n_toward = (reprice_atoms.direction == "toward").sum()
        print(f"    Reprice atoms: {len(reprice_atoms)} (out={n_out} in={n_in} | "
              f"away={n_away} toward={n_toward})", flush=True)

    # 5. Compute windowed features at a single anchor (end of sample)
    if len(ev):
        anchor = np.array([int(ev.timestamp_ms.max())], dtype=np.int64)
        windows = {"1s": 1000, "5s": 5000}
        agg     = windowed_aggressive_flow(taker_atoms, anchor, windows)
        pasv    = windowed_passive_flow(passive_atoms, anchor, windows)
        reprice = windowed_reprice_flow(reprice_atoms, anchor, windows)
        print(f"\n[5] Windowed features at anchor={anchor[0]}:", flush=True)
        print(f"    Aggressive flow:", flush=True)
        print(f"      buy_taker_qty_1s:        {agg.iloc[0].get('buy_taker_qty_1s', 0):.4f}", flush=True)
        print(f"      sell_taker_qty_1s:       {agg.iloc[0].get('sell_taker_qty_1s', 0):.4f}", flush=True)
        print(f"      afi_1s:                  {agg.iloc[0].get('afi_1s', 0):.4f}", flush=True)
        print(f"    Passive flow (explicit add/cancel):", flush=True)
        print(f"      bid_add_qty_1s:          {pasv.iloc[0].get('bid_add_qty_1s', 0):.4f}", flush=True)
        print(f"      bid_cancel_qty_1s:       {pasv.iloc[0].get('bid_cancel_qty_1s', 0):.4f}", flush=True)
        print(f"    Reprice flow (r2.2 — price-level movements):", flush=True)
        print(f"      bid_reprice_away_qty_1s: {reprice.iloc[0].get('bid_reprice_away_qty_1s', 0):.4f}  "
              f"(BID moving away from market)", flush=True)
        print(f"      bid_reprice_toward_qty_1s:{reprice.iloc[0].get('bid_reprice_toward_qty_1s', 0):.4f}  "
              f"(BID moving toward market)", flush=True)
        print(f"      ask_reprice_away_qty_1s: {reprice.iloc[0].get('ask_reprice_away_qty_1s', 0):.4f}  "
              f"(ASK moving away from market)", flush=True)
        print(f"      ask_reprice_toward_qty_1s:{reprice.iloc[0].get('ask_reprice_toward_qty_1s', 0):.4f}  "
              f"(ASK moving toward market)", flush=True)

    print(f"\nDemo complete in {time.time() - t0:.2f}s", flush=True)
    print("=" * 55, flush=True)
    print("To analyse real data: python download_raw.py --help", flush=True)


if __name__ == "__main__":
    run()
