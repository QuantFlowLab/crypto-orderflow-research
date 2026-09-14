"""feature_qa.py — R2: Feature dataset quality checks before analysis.

Checks per (sym, label) parquet:
  - Row count and timestamp coverage
  - Duplicate timestamp keys
  - NaN / inf / not_ready share per feature
  - Range violations (AFI, percentile, spread, depth, qty)
  - Constant columns
  - not_ready behaviour for short-history context windows

Exit code 0 = all checks pass; exit code 1 = at least one violation.
Output: reports/FEATURE_QA_REPORT.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP     = Path(__file__).parent
DERIVED = EXP / "data" / "derived" / "features"
REPORTS = EXP / "reports"
REPORTS.mkdir(exist_ok=True)

WARMUP_MS = 60 * 60 * 1000
ANCHOR_STEP_MS = 1_000
EXPECTED_ROWS = WARMUP_MS // ANCHOR_STEP_MS  # 3600

# Range checks: (column_fragment, min, max)
RANGE_CHECKS = [
    ("afi_",      -1.0,  1.0),
    ("_percentile", 0.0,  1.0),
    ("spread",      0.0,  None),
    ("depth_bid",   0.0,  None),
    ("depth_ask",   0.0,  None),
    ("execution_qty_", 0.0, None),
    ("execution_count_", 0.0, None),
    ("order_event_count_", 0.0, None),
    ("placed_count_", 0.0, None),
    ("cancel_count_", 0.0, None),
    ("buy_taker_qty_", 0.0, None),
    ("sell_taker_qty_", 0.0, None),
]


def _check_range(df: pd.DataFrame, col_frag: str, lo, hi) -> dict:
    cols = [c for c in df.columns if col_frag in c and pd.api.types.is_numeric_dtype(df[c])]
    violations = []
    for c in cols:
        v = df[c].dropna()
        if lo is not None and (v < lo - 1e-9).any():
            violations.append(f"{c}<{lo}:{int((v < lo - 1e-9).sum())}")
        if hi is not None and (v > hi + 1e-9).any():
            violations.append(f"{c}>{hi}:{int((v > hi + 1e-9).sum())}")
    return {"checked": len(cols), "violations": "; ".join(violations) if violations else ""}


def qa_one(path: Path, sym: str, label: str,
           since_ms: int, before_ms: int) -> dict:
    df = pd.read_parquet(path)
    research_ms = since_ms + WARMUP_MS
    expected_first = research_ms + ANCHOR_STEP_MS
    expected_last  = before_ms

    r: dict = {"sym": sym, "label": label}

    # Row count
    r["n_rows"]     = len(df)
    r["n_cols"]     = df.shape[1]
    r["rows_ok"]    = len(df) == EXPECTED_ROWS

    # Timestamp coverage
    if "timestamp_ms" in df.columns and len(df):
        ts = df["timestamp_ms"].dropna().astype(np.int64)
        r["first_ts"]  = int(ts.min())
        r["last_ts"]   = int(ts.max())
        r["first_ts_ok"] = abs(int(ts.min()) - expected_first) <= ANCHOR_STEP_MS
        r["last_ts_ok"]  = abs(int(ts.max()) - expected_last)  <= ANCHOR_STEP_MS
    else:
        r.update({"first_ts": None, "last_ts": None, "first_ts_ok": False, "last_ts_ok": False})

    # Duplicate timestamps
    if "timestamp_ms" in df.columns:
        r["dup_ts"] = int(df["timestamp_ms"].duplicated().sum())

    # NaN / inf share (numeric columns only)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    total_cells = len(df) * len(num_cols) if num_cols else 1
    nan_count = df[num_cols].isna().sum().sum() if num_cols else 0
    inf_count = np.isinf(df[num_cols].values.astype(float)).sum() if num_cols else 0
    r["nan_share"]   = round(nan_count / total_cells, 4) if total_cells else 0.0
    r["inf_count"]   = int(inf_count)
    r["n_num_cols"]  = len(num_cols)

    # Columns with NaN share > 50% (excluding context-window columns expected to be not_ready)
    high_nan = [c for c in num_cols
                if df[c].isna().mean() > 0.5 and "30m" not in c and "120m" not in c]
    r["high_nan_cols"] = len(high_nan)
    r["high_nan_examples"] = "; ".join(high_nan[:5])

    # not_ready share for regime_state (should be >0 for first 30m anchors)
    if "regime_state" in df.columns:
        nr_share = (df["regime_state"] == "not_ready").mean()
        r["regime_not_ready_share"] = round(nr_share, 3)
        # Expect ~0.5 not_ready (first 30m of 60m research) for 30m context
        r["regime_not_ready_ok"] = 0.40 <= nr_share <= 0.60

    # Range checks
    range_violations = []
    for frag, lo, hi in RANGE_CHECKS:
        chk = _check_range(df, frag, lo, hi)
        if chk["violations"]:
            range_violations.append(f"{frag}: {chk['violations']}")
    r["range_violations"] = "; ".join(range_violations) if range_violations else ""

    # Constant columns (variance = 0 over non-NaN values, excluding provenance strings)
    const_cols = [c for c in num_cols
                  if df[c].dropna().nunique() <= 1 and len(df[c].dropna()) > 10]
    r["constant_cols"] = len(const_cols)
    r["constant_examples"] = "; ".join(const_cols[:5])

    # 120m context should never have not_ready=False (always not_ready for 60-min research)
    col_120m = [c for c in df.columns if "120m" in c and pd.api.types.is_numeric_dtype(df[c])]
    if col_120m:
        any_valid_120m = any(df[c].notna().any() for c in col_120m)
        r["120m_context_all_nan"] = not any_valid_120m  # should be True

    # Overall verdict
    hard_fails = [
        not r.get("rows_ok", True),
        r.get("dup_ts", 0) > 0,
        r.get("inf_count", 0) > 0,
        bool(r.get("range_violations", "")),
        r.get("120m_context_all_nan") == False,
    ]
    r["verdict"] = "FAIL" if any(hard_fails) else "PASS"
    return r


def main():
    man = pd.read_csv(EXP / "reports" / "SESSION_MANIFEST.csv", dtype={"selected": str})
    sel = man[man.selected.isin(["LOW", "MEDIAN", "HIGH"])]

    records = []
    all_pass = True
    SYMBOLS = ["PF_XBTUSD", "PF_ETHUSD"]
    label_order = {"LOW": 0, "MEDIAN": 1, "HIGH": 2}

    for label in sorted(sel.selected.unique(), key=lambda x: label_order.get(x, 9)):
        row_man = sel[sel.selected == label].iloc[0]
        since_ms  = int(pd.Timestamp(row_man["warmup_start"]).timestamp() * 1000)
        before_ms = int(pd.Timestamp(row_man["research_end"]).timestamp() * 1000)
        for sym in SYMBOLS:
            path = DERIVED / f"{sym}_{label}_features.parquet"
            if not path.exists():
                print(f"  MISSING: {path.name}", flush=True)
                records.append({"sym": sym, "label": label, "verdict": "MISSING"})
                all_pass = False
                continue
            print(f"  checking {sym} {label} ...", flush=True)
            r = qa_one(path, sym, label, since_ms, before_ms)
            records.append(r)
            status = "✓" if r["verdict"] == "PASS" else "✗"
            print(f"  {status} {sym} {label}: rows={r['n_rows']} cols={r['n_cols']} "
                  f"nan={r['nan_share']:.3f} inf={r.get('inf_count',0)} "
                  f"range_viol={bool(r.get('range_violations',''))} "
                  f"const_cols={r.get('constant_cols',0)} "
                  f"verdict={r['verdict']}", flush=True)
            if r["verdict"] != "PASS":
                all_pass = False
                for k in ("high_nan_examples", "range_violations", "constant_examples"):
                    if r.get(k):
                        print(f"    [{k}] {r[k]}", flush=True)

    report = pd.DataFrame(records)
    report.to_csv(REPORTS / "FEATURE_QA_REPORT.csv", index=False)
    print(f"\nFeature QA: {'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)
    print(f"Report: {REPORTS / 'FEATURE_QA_REPORT.csv'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
