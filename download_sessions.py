"""download_sessions.py — Download LOW/MEDIAN/HIGH sessions from SESSION_MANIFEST.

Sequentially downloads each selected session (warmup + research window) for
each symbol into data/raw/ with suffix _S_{LOW|MEDIAN|HIGH}.
Idempotent: skips already-complete files based on manifest sentinels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EXP = Path(__file__).parent
SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD"]
INTERVAL_TOL_MS = 5 * 60 * 1000  # 5 min: partial file has last_ts far before before_ms


def _iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def _read_first_last_ts(path: Path) -> tuple[int | None, int | None]:
    """Read first/last event timestamps from a jsonl cheaply (no full scan)."""
    first_ts = last_ts = None
    try:
        with open(path, encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                v = json.loads(line).get("timestamp")
                if v is not None:
                    first_ts = int(v)
        with open(path, "rb") as f:
            f.seek(0, 2)
            chunk = min(4096, f.tell())
            f.seek(-chunk, 2)
            tail = f.read().decode("utf-8", errors="replace")
            last_lines = [l for l in tail.splitlines() if l.strip()]
            if last_lines:
                v = json.loads(last_lines[-1]).get("timestamp")
                if v is not None:
                    last_ts = int(v)
    except Exception:
        pass
    return first_ts, last_ts


def _complete(sym: str, suffix: str, since_ms: int, before_ms: int) -> bool:
    """True iff both streams are complete.

    Trusts manifest (COMPLETE / COMPLETE_LEGACY_VERIFIED).
    For legacy files without a manifest: validates the downloaded interval against
    the requested window; backfills manifest on success. A partial file whose
    last_ts is far before before_ms will fail this check and be re-downloaded.
    """
    raw = EXP / "data" / "raw"
    for stream_name in ("orders", "executions"):
        mf = raw / f"{sym}_{stream_name}{suffix}.manifest"
        if mf.exists():
            try:
                status = json.loads(mf.read_text(encoding="utf-8")).get("status", "")
                if status in ("COMPLETE", "COMPLETE_LEGACY_VERIFIED"):
                    continue
            except (json.JSONDecodeError, OSError):
                pass
            return False  # manifest present but not a valid COMPLETE status
        # No manifest: validate interval to distinguish complete vs partial files
        jl = raw / f"{sym}_{stream_name}{suffix}.jsonl"
        if not (jl.exists() and jl.stat().st_size > 0):
            return False
        first_ts, last_ts = _read_first_last_ts(jl)
        if first_ts is None or last_ts is None:
            return False
        # Raw stream is newest-first (descending ts); truncation leaves oldest part missing.
        lo, hi = min(first_ts, last_ts), max(first_ts, last_ts)
        if stream_name == "orders" and (lo > since_ms + INTERVAL_TOL_MS or hi < before_ms - INTERVAL_TOL_MS):
            return False  # partial download
        # Valid: backfill manifest so future calls skip validation (one-time cost)
        m = {"symbol": sym, "stream": stream_name, "suffix": suffix,
             "requested_since_ms": since_ms, "requested_before_ms": before_ms,
             "first_ts": first_ts, "last_ts": last_ts,
             "n_rows": None, "status": "COMPLETE_LEGACY_VERIFIED"}
        mf.write_text(json.dumps(m, indent=2), encoding="utf-8")
    return True


def main():
    man = pd.read_csv(EXP / "reports" / "SESSION_MANIFEST.csv", dtype={"selected": str})
    sel = man[man.selected.isin(["LOW", "MEDIAN", "HIGH"])]
    failures = []
    for _, row in sel.iterrows():
        label = row["selected"]
        since_ms = _iso_to_ms(row["warmup_start"])
        before_ms = _iso_to_ms(row["research_end"])
        for sym in SYMBOLS:
            suffix = f"_S_{label}"
            if _complete(sym, suffix, since_ms, before_ms):
                print(f"=== SKIP {label} {sym} (already complete)", flush=True)
                continue
            cmd = [sys.executable, "-u", str(EXP / "download_raw.py"),
                   "--symbol", sym, "--start", row["warmup_start"],
                   "--end", row["research_end"], "--suffix", suffix]
            print(f">>> {label} {sym} {row['warmup_start']}..{row['research_end']}", flush=True)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! FAILED {label} {sym}: {e}", flush=True)
                failures.append((label, sym))
    print(json.dumps({"done": True, "sessions": list(sel.selected), "failures": failures}), flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
