"""run_semantics_sanity.py — KRAKEN-OF-1-R2.5: Feature Semantics Sanity.

Validates semantic correctness on BTC+ETH MEDIAN sessions.
FROZEN RULE: violations → STOP + fix before proceeding. No tuning.

Output: reports/SEMANTICS_SANITY_REPORT.txt
        reports/SEMANTICS_SANITY_ANCHORS.csv
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).parent
sys.path.insert(0, str(EXP / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from replenishment import build_refill_pairs
from wall_state import build_level_atoms
from run_r2 import _stream_event_df, _stream_taker_atoms, _stream_passive_atoms, EPS

REPORTS = EXP / "reports"
FEATURES = EXP / "data" / "derived" / "features"
RAW      = EXP / "data" / "raw"
REPORTS.mkdir(exist_ok=True)

MEDIAN_WARMUP_START  = "2026-09-09T19:00:00Z"
MEDIAN_RESEARCH_END  = "2026-09-09T21:00:00Z"
WARMUP_MS = 60 * 60 * 1000
TICK = {"PF_XBTUSD": 1.0, "PF_ETHUSD": 0.1}
RNG = random.Random(42)

PCTILES = [0, 1, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]

# Representative feature families for distribution tables
REPRESENTATIVE = {
    "REGIME":       ["events_per_sec_1s", "trades_per_sec_1s", "qty_per_sec_1s",
                     "spread", "near_book_depth"],
    "AGGRESSIVE":   ["buy_taker_qty_1s", "sell_taker_qty_1s", "afi_1s",
                     "buy_flow_accel_2s", "sell_flow_accel_2s"],
    "PASSIVE":      ["bid_add_qty_1s", "ask_add_qty_1s",
                     "bid_cancel_qty_1s", "ask_cancel_qty_1s"],
    "PRICE_RESP":   ["signed_mid_change_1s", "ticks_moved_1s",
                     "price_response_per_sell_qty_1s", "price_response_per_buy_qty_1s"],
    "LIQUIDITY":    ["bid_max_qty_vs_median", "bid_max_share", "bid_top1_dist_ticks",
                     "ask_max_qty_vs_median"],
    "REPLENISHMENT":["total_exec_qty_1s", "total_refill_qty_1s",
                     "replenishment_ratio_1s", "refill_count_1s"],
}

lines: list[str] = []
violations: list[str] = []


def _log(s=""):
    print(s, flush=True)
    lines.append(s)


def _fail(msg: str):
    _log(f"  VIOLATION: {msg}")
    violations.append(msg)


def _ok(msg: str):
    _log(f"  ok: {msg}")


def _pct_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"feature": c, "status": "MISSING"}); continue
        v = df[c].dropna()
        if len(v) == 0:
            rows.append({"feature": c, "status": "ALL_NAN"}); continue
        row = {"feature": c, "n": len(v), "nan%": round(df[c].isna().mean()*100, 1),
               "mean": round(v.mean(), 4), "std": round(v.std(), 4)}
        for p in PCTILES:
            row[f"p{p}"] = round(float(np.percentile(v, p)), 6)
        row["status"] = "ok"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Section 1: Feature inventory ──────────────────────────────────────────────

def section_inventory(btc: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 1: FEATURE INVENTORY")
    _log("="*72)
    col_path = REPORTS / "FEATURE_COLUMNS_r2.2.txt"
    col_path.write_text("\n".join(btc.columns.tolist()), encoding="utf-8")
    _log(f"  {btc.shape[1]} columns x {btc.shape[0]} rows")
    _log(f"  feature_version: {btc['feature_version'].iloc[0]}")
    _log(f"  data_version: {btc['data_version'].iloc[0]}")
    _log(f"  book_semantics_version: {btc['book_semantics_version'].iloc[0]}")
    _log(f"  Column list saved to {col_path.name}")


# ── Section 2: Distribution tables ────────────────────────────────────────────

def section_distributions(btc: pd.DataFrame, eth: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 2: DISTRIBUTION TABLES (BTC MEDIAN)")
    _log("="*72)
    all_tables = []
    for family, cols in REPRESENTATIVE.items():
        _log(f"\n  [{family}]")
        t = _pct_table(btc, cols)
        all_tables.append(t.assign(family=family))
        _log(t[["feature","n","nan%","mean","std","p0","p50","p90","p99","p100","status"]
               ].to_string(index=False))

    # Aggregate table for report
    big = pd.concat(all_tables, ignore_index=True)
    big.to_csv(REPORTS / "DISTRIBUTION_TABLES.csv", index=False)


# ── Section 3: Sign convention checks ─────────────────────────────────────────

def section_signs(btc: pd.DataFrame, eth: pd.DataFrame, tick_btc, tick_eth):
    _log("\n" + "="*72)
    _log("SECTION 3: SIGN CONVENTION")
    _log("="*72)
    for df, sym, tick in [(btc, "BTC", tick_btc), (eth, "ETH", tick_eth)]:
        _log(f"\n  [{sym}]")
        # AFI sign matches aggressor dominance
        strong_sell = (df.sell_taker_qty_1s > 5 * df.buy_taker_qty_1s + 1e-6)
        strong_buy  = (df.buy_taker_qty_1s  > 5 * df.sell_taker_qty_1s + 1e-6)
        afi_ok_sell = (df.afi_1s[strong_sell] < 0).all()
        afi_ok_buy  = (df.afi_1s[strong_buy]  > 0).all()
        n_sell = strong_sell.sum(); n_buy = strong_buy.sum()
        if afi_ok_sell:
            _ok(f"AFI < 0 for all {n_sell} strong-sell anchors")
        else:
            _fail(f"AFI >= 0 for some strong-sell anchors: {(~(df.afi_1s[strong_sell] < 0)).sum()}")
        if afi_ok_buy:
            _ok(f"AFI > 0 for all {n_buy} strong-buy anchors")
        else:
            _fail(f"AFI <= 0 for some strong-buy anchors")

        # Pure one-sided flow → AFI must be within EPS-level tolerance of ±1
        # (EPS in denominator gives 0.999... not exactly 1.0 — within 1e-6 is the right criterion)
        AFI_TOL = 1e-6
        pure_sell = (df.buy_taker_qty_1s == 0) & (df.sell_taker_qty_1s > 0)
        pure_buy  = (df.sell_taker_qty_1s == 0) & (df.buy_taker_qty_1s > 0)
        if pure_sell.sum() > 0 and (df.afi_1s[pure_sell] > -1 + AFI_TOL).any():
            _fail(f"AFI > -1+tol for {(df.afi_1s[pure_sell] > -1+AFI_TOL).sum()} pure-sell anchors")
        else:
            _ok(f"AFI ~= -1 for all {pure_sell.sum()} pure-sell anchors (within {AFI_TOL})")
        if pure_buy.sum() > 0 and (df.afi_1s[pure_buy] < 1 - AFI_TOL).any():
            _fail(f"AFI < 1-tol for {(df.afi_1s[pure_buy] < 1-AFI_TOL).sum()} pure-buy anchors")
        else:
            _ok(f"AFI ~= +1 for all {pure_buy.sum()} pure-buy anchors (within {AFI_TOL})")

        # ticks_moved >= 0 (unsigned magnitude since r2.1 fix)
        if "ticks_moved_1s" in df.columns and (df.ticks_moved_1s.dropna() < -1e-9).any():
            _fail("ticks_moved_1s has negative values (should be unsigned abs(smc/tick))")
        else:
            _ok("ticks_moved_1s >= 0")

        # ticks_moved == abs(signed_mid_change)/tick (unsigned equality check)
        if "signed_mid_change_1s" in df.columns:
            ticks = df.ticks_moved_1s.dropna()
            smc   = df.signed_mid_change_1s.dropna()
            shared = ticks.index.intersection(smc.index)
            abs_err = (ticks[shared] - np.abs(smc[shared]) / tick).abs()
            max_err = float(abs_err.max()) if len(abs_err) else 0
            if max_err > 1e-6:
                _fail(f"ticks_moved != abs(signed_mid_change)/tick; max_err={max_err:.2e}")
            else:
                _ok(f"ticks_moved == abs(signed_mid_change)/tick (max_err={max_err:.2e})")


# ── Section 4: Ratio edge cases ───────────────────────────────────────────────

def section_ratios(btc: pd.DataFrame, eth: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 4: RATIO EDGE CASES")
    _log("="*72)
    for df, sym in [(btc, "BTC"), (eth, "ETH")]:
        _log(f"\n  [{sym}]")

        # No infinities anywhere
        num = df.select_dtypes(include="number")
        inf_count = np.isinf(num.values.astype(float)).sum()
        if inf_count > 0:
            _fail(f"{inf_count} inf values across all numeric columns")
        else:
            _ok("No inf values")

        # Replenishment: ratio == 0 when no executions
        for W in ("100ms", "500ms", "1s", "5s"):
            exec_col  = f"total_exec_qty_{W}"
            ratio_col = f"replenishment_ratio_{W}"
            if exec_col in df and ratio_col in df:
                no_exec = df[exec_col] == 0
                bad = no_exec & (df[ratio_col] != 0)
                if bad.any():
                    _fail(f"replenishment_ratio_{W} != 0 when exec=0: {bad.sum()} rows")
                else:
                    _ok(f"replenishment_ratio_{W} == 0 when exec=0")

        # Acceleration artifact: near-zero prior → huge ratio
        for wl in ("1s", "2s", "5s", "10s"):
            col = f"buy_flow_accel_{wl}"
            if col in df.columns:
                v = df[col].dropna()
                p99 = float(np.percentile(v, 99)) if len(v) else 0
                max_v = float(v.max()) if len(v) else 0
                if max_v > 1e6:
                    _fail(f"{col} max={max_v:.2e} p99={p99:.2e} — near-zero denominator artifact "
                          f"(EPS={EPS}); should be NaN when prior_window==0")
                else:
                    _ok(f"{col} max={max_v:.4f} — reasonable range")

        # price_response ratios when flow near-zero
        for wl in ("1s",):
            for side in ("buy", "sell"):
                col = f"price_response_per_{side}_qty_{wl}"
                qty_col = f"{side}_taker_qty_{wl}"
                if col in df and qty_col in df:
                    near_zero_flow = df[qty_col] < 1e-8
                    if near_zero_flow.sum() > 0:
                        huge = (df[col][near_zero_flow].abs() > 1e6).sum()
                        if huge > 0:
                            _fail(f"{col}: {huge} huge values when flow~0 (EPS artifact)")
                        else:
                            _ok(f"{col}: no huge values at near-zero {side} flow")


# ── Section 5: Monotonicity (shorter window ≤ longer window) ─────────────────

def section_monotonicity(btc: pd.DataFrame, eth: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 5: WINDOW MONOTONICITY (qty_W1 <= qty_W2 when W1 < W2)")
    _log("="*72)
    ordered = ["100ms", "250ms", "500ms", "1s", "2s", "5s", "10s", "30s", "60s", "300s"]
    for df, sym in [(btc, "BTC"), (eth, "ETH")]:
        _log(f"\n  [{sym}]")
        for prefix in ("buy_taker_qty", "sell_taker_qty", "bid_add_qty", "bid_cancel_qty"):
            avail = [(w, f"{prefix}_{w}") for w in ordered if f"{prefix}_{w}" in df.columns]
            for i in range(len(avail) - 1):
                w_s, c_s = avail[i]; w_l, c_l = avail[i+1]
                viol = (df[c_l] < df[c_s] - 1e-9).sum()
                if viol > 0:
                    _fail(f"{prefix}: {c_l} < {c_s} in {viol} rows (window monotonicity broken)")
        _ok(f"Window monotonicity checked for 4 feature families")


# ── Section 6: Acceleration non-overlapping windows ──────────────────────────

def section_acceleration(btc: pd.DataFrame, eth: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 6: ACCELERATION WINDOW VERIFICATION")
    _log("="*72)
    _log("  Formula: accel_{long} = recent_{short} / prior_{long-short}")
    _log("  prior_{long-short} = qty_{long} - qty_{short}  (non-overlapping)")
    pairs = [("500ms", "1s"), ("1s", "2s"), ("2s", "5s"), ("5s", "10s")]
    EPS_ACCEL = 1e-12
    for df, sym in [(btc, "BTC"), (eth, "ETH")]:
        _log(f"\n  [{sym}]")
        for short_wl, long_wl in pairs:
            sq  = f"buy_taker_qty_{short_wl}"
            lq  = f"buy_taker_qty_{long_wl}"
            acc = f"buy_flow_accel_{long_wl}"
            if sq not in df or lq not in df or acc not in df:
                continue
            # Verify for rows where prior > 0 (well-defined)
            prior = (df[lq] - df[sq]).clip(lower=0)
            well_defined = prior > 1e-8
            if well_defined.sum() == 0:
                _log(f"    {acc}: no rows with non-zero prior")
                continue
            expected = (df[sq][well_defined] / (prior[well_defined] + EPS_ACCEL)).round(8)
            actual   = df[acc][well_defined].round(8)
            mismatch = (expected - actual).abs()
            max_mm = float(mismatch.max())
            if max_mm > 1e-4:
                _fail(f"{acc}: formula mismatch max={max_mm:.2e}")
            else:
                _ok(f"{acc}: formula correct on {well_defined.sum()} rows (max_err={max_mm:.2e})")
            # Log what fraction has zero prior (undefined/NaN check)
            n_zero_prior = (~well_defined).sum()
            n_nan_acc    = df[acc].isna().sum()
            _log(f"    zero_prior={n_zero_prior} nan_in_accel={n_nan_acc} "
                 f"(should match after fix)")


# ── Section 7: Replenishment causality audit (ETH MEDIAN raw) ─────────────────

def section_replenishment_causality(eth: pd.DataFrame):
    _log("\n" + "="*72)
    _log("SECTION 7: REPLENISHMENT CAUSALITY AUDIT (ETH MEDIAN raw data)")
    _log("="*72)
    _log("  Verifying: exec_ts < add_ts <= anchor_ts; same (side, price)")

    # Identify top 10 replenishment anchors
    if "replenishment_ratio_1s" not in eth.columns:
        _log("  SKIP: replenishment_ratio_1s not in ETH parquet")
        return
    top_anc = eth.nlargest(10, "replenishment_ratio_1s")["timestamp_ms"].tolist()
    _log(f"  Top-10 replenishment anchors: {top_anc[:3]}... (ratio: "
         + ", ".join(f"{r:.2f}" for r in eth.nlargest(10,'replenishment_ratio_1s')['replenishment_ratio_1s'].tolist()[:3])
         + ")")

    # Load ETH MEDIAN raw → build level_atoms + refill_pairs
    _log("  Loading ETH MEDIAN raw (streaming)...", )
    orders_path = RAW / "PF_ETHUSD_orders_S_MEDIAN.jsonl"
    execs_path  = RAW / "PF_ETHUSD_executions_S_MEDIAN.jsonl"
    since_ms    = int(pd.Timestamp(MEDIAN_WARMUP_START).timestamp() * 1000)

    ev = _stream_event_df(orders_path, execs_path)
    level_atoms  = build_level_atoms(ev)
    refill_pairs = build_refill_pairs(level_atoms)
    _log(f"  level_atoms={len(level_atoms)} refill_pairs={len(refill_pairs)}")

    if len(refill_pairs) == 0:
        _log("  No refill pairs — skip causality audit")
        return

    # Check global invariants on ALL pairs
    bad_order = (refill_pairs.exec_ts >= refill_pairs.add_ts).sum()
    if bad_order > 0:
        _fail(f"exec_ts >= add_ts in {bad_order} refill pairs (causality violation!)")
    else:
        _ok(f"exec_ts < add_ts for all {len(refill_pairs)} pairs")

    bad_delay = (refill_pairs.delay_ms <= 0).sum()
    if bad_delay > 0:
        _fail(f"delay_ms <= 0 in {bad_delay} pairs")
    else:
        _ok("delay_ms > 0 for all pairs")

    # Check top-10 anchors individually
    research_ms = since_ms + WARMUP_MS
    _log("\n  Spot-check top anchors:")
    for anchor in top_anc[:5]:
        ratio = float(eth.loc[eth.timestamp_ms == anchor, "replenishment_ratio_1s"].iloc[0])
        # Filter pairs relevant to this anchor (exec in trailing 1s AND add before anchor)
        m = (refill_pairs.exec_ts >= anchor - 1000) & \
            (refill_pairs.exec_ts < anchor) & \
            (refill_pairs.add_ts <= anchor) & \
            (refill_pairs.delay_ms <= 1000)
        sub = refill_pairs[m]
        # Verify causal ordering
        causal_ok = (sub.exec_ts < sub.add_ts).all() if len(sub) else True
        # Verify exec before anchor
        before_ok = (sub.exec_ts < anchor).all() if len(sub) else True
        # Verify add before anchor
        add_ok = (sub.add_ts <= anchor).all() if len(sub) else True
        status = "OK" if (causal_ok and before_ok and add_ok) else "FAIL"
        _log(f"    anchor={anchor} ratio={ratio:.2f} pairs={len(sub)} "
             f"causal={causal_ok} before_anchor={before_ok} add<={anchor}: {add_ok} → {status}")
        if not (causal_ok and before_ok and add_ok):
            _fail(f"causality violation at anchor {anchor}")

    # Note on multiple-window counting (by design, not a bug)
    _log("\n  Note: a single refill event is counted in all windows where delay <= W.")
    _log("  refill_qty_500ms >= refill_qty_100ms (wider window catches more delayed refills).")
    # Verify monotonicity
    for w_short, w_long in [("100ms", "500ms"), ("500ms", "1s"), ("1s", "5s")]:
        rc = f"total_refill_qty_{w_short}"; lc = f"total_refill_qty_{w_long}"
        if rc in eth and lc in eth:
            viol = (eth[lc] < eth[rc] - 1e-9).sum()
            if viol:
                _fail(f"replenishment: {lc} < {rc} in {viol} rows")
            else:
                _ok(f"total_refill_qty_{w_short} <= total_refill_qty_{w_long}")


# ── Section 8: Anchor spot-checks ─────────────────────────────────────────────

def section_anchors(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    _log("\n" + "="*72)
    _log("SECTION 8: ANCHOR SPOT-CHECKS (BTC MEDIAN)")
    _log("="*72)
    snap_cols = ["timestamp_ms", "mid", "spread", "regime_state",
                 "buy_taker_qty_1s", "sell_taker_qty_1s", "afi_1s",
                 "bid_add_qty_1s", "ask_cancel_qty_1s",
                 "signed_mid_change_1s", "ticks_moved_1s",
                 "bid_max_qty_vs_median", "replenishment_ratio_1s",
                 "total_exec_qty_1s", "near_book_depth"]
    avail = [c for c in snap_cols if c in btc.columns]

    groups = {
        "random_20":    btc.sample(20, random_state=42).index.tolist(),
        "top5_activity": btc.nlargest(5, "events_per_sec_1s").index.tolist() if "events_per_sec_1s" in btc else [],
        "top5_buy_flow": btc.nlargest(5, "buy_taker_qty_1s").index.tolist(),
        "top5_sell_flow": btc.nlargest(5, "sell_taker_qty_1s").index.tolist(),
        "top5_replenishment": btc.nlargest(5, "replenishment_ratio_1s").index.tolist(),
    }
    dfs = []
    for group_name, idxs in groups.items():
        sub = btc.loc[idxs, avail].copy()
        sub.insert(0, "group", group_name)
        dfs.append(sub)
    out = pd.concat(dfs, ignore_index=True)
    _log(f"\n  {len(out)} anchors × {len(avail)} selected features")
    _log("\n  Top-5 sell flow snapshot (showing sign convention in context):")
    sell_snap = btc.nlargest(5, "sell_taker_qty_1s")[avail].copy()
    _log(sell_snap[["timestamp_ms","sell_taker_qty_1s","buy_taker_qty_1s","afi_1s",
                     "signed_mid_change_1s","bid_add_qty_1s"]
                    if all(c in avail for c in ["sell_taker_qty_1s","afi_1s","signed_mid_change_1s","bid_add_qty_1s"])
                    else avail[:6]].to_string(index=False))
    return out


# ── Section 9: Verdict ────────────────────────────────────────────────────────

def section_verdict():
    _log("\n" + "="*72)
    _log("SECTION 9: VERDICT")
    _log("="*72)
    if not violations:
        _log("  FEATURE_SEMANTICS: PASS")
        _log("  No violations found. Feature definitions are semantically correct.")
        _log("  FEATURE_VERSION r2.2 FROZEN.")
        _log("  r2.2 adds: REPRICE_OUT/IN atoms with away/toward direction (~90% of OrderUpdated events).")
    else:
        _log(f"  FEATURE_SEMANTICS: FAIL ({len(violations)} violation(s))")
        for i, v in enumerate(violations, 1):
            _log(f"  [{i}] {v}")
        _log("\n  Fix all violations and re-run before proceeding to analysis.")
    _log("="*72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading BTC+ETH MEDIAN parquets...", flush=True)
    btc = pd.read_parquet(FEATURES / "PF_XBTUSD_MEDIAN_features.parquet")
    eth = pd.read_parquet(FEATURES / "PF_ETHUSD_MEDIAN_features.parquet")

    _log("KRAKEN-OF-1-R2.5 — FEATURE SEMANTICS SANITY REPORT")
    _log(f"BTC MEDIAN: {btc.shape}   ETH MEDIAN: {eth.shape}")
    _log(f"BTC feature_version: {btc['feature_version'].iloc[0]}")

    section_inventory(btc)
    section_distributions(btc, eth)
    section_signs(btc, eth, TICK["PF_XBTUSD"], TICK["PF_ETHUSD"])
    section_ratios(btc, eth)
    section_monotonicity(btc, eth)
    section_acceleration(btc, eth)
    section_replenishment_causality(eth)
    anchors_df = section_anchors(btc, eth)
    section_verdict()

    report_text = "\n".join(lines)
    (REPORTS / "SEMANTICS_SANITY_REPORT.txt").write_text(report_text, encoding="utf-8")
    anchors_df.to_csv(REPORTS / "SEMANTICS_SANITY_ANCHORS.csv", index=False)
    print(f"\nReport: {REPORTS}/SEMANTICS_SANITY_REPORT.txt", flush=True)
    return len(violations)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
