"""download_raw.py — Download raw order and execution events from Kraken Futures API.

Streams individual order events and execution records from the public
Kraken Futures historical REST API (v3) for a given symbol and time window.
Writes JSONL output consumed by orderflow.build_event_table.

Usage:
    python download_raw.py --symbol PF_XBTUSD \
                           --start 2026-09-01T00:00:00Z \
                           --end   2026-09-01T02:00:00Z
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://futures.kraken.com"
OUT = Path(__file__).parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
MAX_PAGES = 50_000  # safety fuse — if reached, download FAILS (not silently truncated)

_SESSION = requests.Session()


class _NonRetryableError(Exception):
    """HTTP 4xx (not 429): client error, never transient — fail immediately."""


def _get_with_retry(url: str, params: dict, *, max_attempts: int = 8) -> dict:
    """GET + parse JSON with exponential backoff.

    Retries: timeout, 429, 5xx, network errors.
    Immediate fail: HTTP 4xx (not 429) — bad request, retrying won't help.
    """
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            r = _SESSION.get(url, params=params, timeout=(10, 60))
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise _NonRetryableError(f"HTTP {r.status_code} (non-retryable): {url}")
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.exceptions.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            return r.json()
        except _NonRetryableError:
            raise
        except (requests.exceptions.RequestException, ValueError) as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"    [retry {attempt+1}/{max_attempts}] {type(e).__name__}: {e} -> sleep {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"exhausted {max_attempts} retries; last error: {last_err}")


def stream(sym: str, stream_name: str, since_ms: int, before_ms: int, suffix: str) -> int:
    """Download one stream to a .part file; atomic rename + manifest sentinel on success.

    A canonical .jsonl only appears when download is fully complete. A .manifest file
    (status=COMPLETE) is the authoritative completion record; it also stores first/last
    timestamps so Hard QA can verify the downloaded interval matches the manifest request.
    """
    api_path = f"/api/history/v3/market/{sym}/{stream_name}"
    out_file = OUT / f"{sym}_{stream_name}{suffix}.jsonl"
    part_file = out_file.with_name(out_file.name + ".part")
    manifest_file = out_file.with_suffix(".manifest")

    part_file.unlink(missing_ok=True)  # remove stale .part from previous crash

    cont = None
    pages = total = 0
    first_ts: int | None = None
    last_ts: int | None = None
    pagination_exhausted = False
    t0 = time.time()
    with open(part_file, "w", encoding="utf-8") as f:
        while True:
            params: dict = {"since": since_ms, "before": before_ms, "count": 1000, "sort": "desc"}
            if cont:
                params["continuation_token"] = cont  # request param (snake_case per Kraken API docs)
            j = _get_with_retry(BASE + api_path, params)
            elems = j.get("elements", [])
            for e in elems:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
                ts = e.get("timestamp")
                if ts is not None:
                    ts_int = int(ts)
                    if first_ts is None:
                        first_ts = ts_int
                    last_ts = ts_int
            total += len(elems)
            cont = j.get("continuationToken")
            pages += 1
            if pages % 50 == 0:
                print(f"  {sym} {stream_name}{suffix}: {pages}p {total}ev {time.time()-t0:.0f}s", flush=True)
            if not cont or not elems:
                pagination_exhausted = True
                break
            if pages >= MAX_PAGES:
                break  # safety fuse; handled below

    if not pagination_exhausted:
        # Safety fuse was triggered: partial download; do NOT create canonical file
        part_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"MAX_PAGES safety fuse ({MAX_PAGES}) triggered for {sym}/{stream_name}{suffix} "
            f"after {pages}p/{total}ev. Download is incomplete. Investigate or raise MAX_PAGES."
        )

    # Atomic overwrite: os.replace() handles existing destination on Windows
    os.replace(part_file, out_file)

    # Write completion sentinel with interval truth for Hard QA
    manifest = {
        "symbol": sym, "stream": stream_name, "suffix": suffix,
        "requested_since_ms": since_ms, "requested_before_ms": before_ms,
        "first_ts": first_ts, "last_ts": last_ts,
        "n_rows": total, "pages": pages,
        "pagination_exhausted": pagination_exhausted,
        "wall_clock_s": round(time.time() - t0, 1),
        "status": "COMPLETE",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"  {sym} {stream_name}{suffix}: DONE {pages}p {total}ev -> {out_file.name} ({dt:.0f}s)", flush=True)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", default="2026-09-11T21:25:00Z")
    ap.add_argument("--end", default="2026-09-11T22:00:00Z")
    ap.add_argument("--suffix", default="")  # e.g. "_post" for post-window
    a = ap.parse_args()
    start = datetime.fromisoformat(a.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(a.end.replace("Z", "+00:00"))
    since_ms = int(start.timestamp() * 1000)
    before_ms = int(end.timestamp() * 1000)
    print(f"[download] {a.symbol} window {start:%H:%M}-{end:%H:%M}Z suffix='{a.suffix}'", flush=True)
    n_ord = stream(a.symbol, "orders", since_ms, before_ms, a.suffix)
    n_ex = stream(a.symbol, "executions", since_ms, before_ms, a.suffix)
    print(json.dumps({"symbol": a.symbol, "suffix": a.suffix, "orders": n_ord, "executions": n_ex}), flush=True)
