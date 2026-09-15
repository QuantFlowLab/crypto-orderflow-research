"""session_select.py — R1 Part 2: mechanical selection of research sessions.

Ranks candidate 60-min windows ONLY by BTC order_event_count/sec (no returns/vol/
future). Probes a short slice at the start of each window, estimates eps, selects LOW(~15pct)/
MEDIAN(~50pct)/HIGH(~92pct). Writes SESSION_CANDIDATES.csv (all candidates + selection logic);
SESSION_MANIFEST.csv is the separate frozen 3-row artifact with the final selected sessions.

Usage: python session_select.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://futures.kraken.com"
REPORTS = Path(__file__).parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RANK_SYMBOL = "PF_XBTUSD"        # selection variable source (partner: BTC order-event rate only)
PROBE_SEC = 60                   # short slice to estimate eps (cheap, unbiased ranking)
DAYS = ["2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"]
TIMES = ["02:00", "08:00", "14:00", "20:00"]  # Asia / Europe / US-pre / US-peak
RESEARCH_MIN = 60
WARMUP_MIN = 60
PCTL = {"LOW": 15, "MEDIAN": 50, "HIGH": 92}


def probe_eps(symbol: str, start_iso: str) -> tuple[int, int]:
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    since_ms = int(start.timestamp() * 1000)
    before_ms = since_ms + PROBE_SEC * 1000
    path = f"/api/history/v3/market/{symbol}/orders"
    cont = None; total = 0; pages = 0
    while True:
        params = {"since": since_ms, "before": before_ms, "count": 1000, "sort": "desc"}
        if cont:
            params["continuation_token"] = cont  # request param (snake_case per Kraken API docs)
        r = requests.get(BASE + path, params=params, timeout=30)
        j = r.json()
        elems = j.get("elements", [])
        total += len(elems); cont = j.get("continuationToken"); pages += 1
        if not cont or not elems or pages > 200:
            break
    return total, pages


def main():
    candidates = []
    for d in DAYS:
        for t in TIMES:
            start_iso = f"{d}T{t}:00Z"
            n, pages = probe_eps(RANK_SYMBOL, start_iso)
            eps = round(n / PROBE_SEC, 1)
            candidates.append({"day": d, "time": t, "research_start": start_iso,
                               "probe_events": n, "probe_pages": pages, "eps": eps})
            print(f"  {start_iso}: eps={eps} ({n} ev / {PROBE_SEC}s)", flush=True)
            time.sleep(0.1)

    df = pd.DataFrame(candidates)
    eps = df["eps"].to_numpy()
    df["pctl"] = [round(float((eps <= v).mean()) * 100, 1) for v in eps]

    # pick candidate whose eps is closest to the target percentile value
    sel = {}
    for label, p in PCTL.items():
        target = np.percentile(eps, p)
        idx = int(np.argmin(np.abs(eps - target)))
        sel[label] = idx
    df["selected"] = ""
    for label, idx in sel.items():
        df.loc[idx, "selected"] = label

    # research/warmup/download windows for selected
    def windows(row):
        rs = datetime.fromisoformat(row["research_start"].replace("Z", "+00:00"))
        re = rs + timedelta(minutes=RESEARCH_MIN)
        ws = rs - timedelta(minutes=WARMUP_MIN)
        return pd.Series({"warmup_start": ws.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "research_end": re.strftime("%Y-%m-%dT%H:%M:%SZ")})
    df = pd.concat([df, df.apply(windows, axis=1)], axis=1)

    df.to_csv(REPORTS / "SESSION_CANDIDATES.csv", index=False)
    meta = {"rank_symbol": RANK_SYMBOL, "selection_variable": "order_event_count/sec (BTC)",
            "probe_sec": PROBE_SEC, "n_candidates": len(df), "percentile_targets": PCTL,
            "selected": {label: df.loc[idx, ["day", "time", "eps", "pctl"]].to_dict()
                         for label, idx in sel.items()},
            "note": "Mechanical selection by activity only. No returns/volatility/future used. SESSION_MANIFEST.csv contains only the frozen selected rows."}
    (REPORTS / "SESSION_CANDIDATES_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nSELECTED:")
    print(df[df.selected != ""][["selected", "research_start", "eps", "pctl", "warmup_start", "research_end"]].to_string(index=False))


if __name__ == "__main__":
    main()
