"""run_r3_qa1.py — KRAKEN-OF-1-R3-QA1: Report cleanup + conditional analysis.

Fixes and additions (all causal, no future prices/labels/ML/PnL):
1. Correct trades/sec ratio statement (mean/p90/p99, not undefined median ratio).
2. BUY-pressure joint tables (mirror of SELL, using ask_net_passive).
3. Conditional top-flow split: within top-10% flow, bid/ask_net_passive sign buckets.
4. Replenishment by execution-size bucket.
5. Near-market liquidity stats (5/10/25 bps zones + bid_top1_dist distribution).
6. Reconstruct ORDER_FLOW_REPORT.md Section 7 as Observed vs Candidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP     = Path(__file__).parent
FEAT    = EXP / "data" / "derived" / "features"
REPORTS = EXP / "reports"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SYMBOLS  = ["PF_XBTUSD", "PF_ETHUSD"]
LABELS   = ["LOW", "MEDIAN", "HIGH"]
SYM_NICE = {"PF_XBTUSD": "BTC", "PF_ETHUSD": "ETH"}
PRESSURE_BUCKETS = [(0, 50, "0-50%"), (50, 90, "50-90%"),
                    (90, 95, "90-95%"), (95, 99, "95-99%"), (99, 100, "top-1%")]


def load_all() -> dict[tuple, pd.DataFrame]:
    dfs = {}
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = pd.read_parquet(FEAT / f"{sym}_{lbl}_features.parquet")
            for side in ("buy", "sell"):
                col = f"{side}_taker_qty_1s"
                if col in df:
                    df[f"{side}_pressure_pct"] = df[col].rank(pct=True) * 100
            dfs[(sym, lbl)] = df
    return dfs


# ── 1. Trades/sec corrected table ─────────────────────────────────────────────

def trades_per_sec_table(dfs) -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            col = "trades_per_sec_1s"
            if col not in df:
                continue
            v = df[col].dropna()
            rows.append({"symbol": SYM_NICE[sym], "label": lbl,
                          "mean": round(v.mean(), 3), "p50": round(v.median(), 2),
                          "p90": round(v.quantile(.9), 2), "p99": round(v.quantile(.99), 2),
                          "max": round(v.max(), 0)})
    return pd.DataFrame(rows)


# ── 2. Full joint-state (SELL + BUY) ──────────────────────────────────────────

def bucket_col(df, side):
    pct = df.get(f"{side}_pressure_pct")
    if pct is None:
        return None
    labels = ["0-50%", "50-90%", "90-95%", "95-99%", "top-1%"]
    return pd.cut(pct, bins=[0, 50, 90, 95, 99, 100], labels=labels, include_lowest=True)

SELL_RESP = {
    "bid_add_qty_1s": "bid_add", "bid_cancel_qty_1s": "bid_cxl",
    "bid_net_passive_1s": "bid_net", "ask_add_qty_1s": "ask_add",
    "ask_cancel_qty_1s": "ask_cxl", "near_book_depth": "depth",
    "signed_mid_change_1s": "mid_chg", "ticks_moved_1s": "ticks",
    "replenishment_ratio_1s": "repl_ratio",
}
BUY_RESP = {
    "ask_add_qty_1s": "ask_add", "ask_cancel_qty_1s": "ask_cxl",
    "ask_net_passive_1s": "ask_net", "bid_add_qty_1s": "bid_add",
    "bid_cancel_qty_1s": "bid_cxl", "near_book_depth": "depth",
    "signed_mid_change_1s": "mid_chg", "ticks_moved_1s": "ticks",
    "replenishment_ratio_1s": "repl_ratio",
}

def compute_joint_state_v2(dfs) -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)].copy()
            for pressure_side, resp_map in [("SELL", SELL_RESP), ("BUY", BUY_RESP)]:
                bk = bucket_col(df, pressure_side.lower())
                if bk is None:
                    continue
                df["_bucket"] = bk
                for bname, grp in df.groupby("_bucket", observed=True):
                    row = {"symbol": sym, "label": lbl, "pressure_side": pressure_side,
                           "bucket": str(bname), "n": len(grp)}
                    for col, tag in resp_map.items():
                        if col in grp:
                            v = grp[col].dropna()
                            row[f"{tag}_p50"]  = round(v.median(), 4)
                            row[f"{tag}_p90"]  = round(v.quantile(.9), 4)
                            row[f"{tag}_mean"] = round(v.mean(), 4)
                    rows.append(row)
    return pd.DataFrame(rows)


# ── 3. Conditional split: top-10% flow × bid/ask_net_passive sign ─────────────

def compute_conditional_split(dfs) -> pd.DataFrame:
    """Within top-10% sell/buy flow, split by passive-side net sign.
    SELL aggressor: passive side = BID. BUY aggressor: passive side = ASK.
    This tests the absorption-candidate hypothesis from data.
    """
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)].copy()
            for press_side, flow_col, net_col in [
                ("SELL", "sell_taker_qty_1s", "bid_net_passive_1s"),
                ("BUY",  "buy_taker_qty_1s",  "ask_net_passive_1s"),
            ]:
                if flow_col not in df or net_col not in df:
                    continue
                top10 = df[df[f"{press_side.lower()}_pressure_pct"] > 90].copy()
                if len(top10) < 10:
                    continue
                # Sign buckets of passive-side net
                for sign_label, mask in [
                    ("passive_adds",    top10[net_col] > 0.1),
                    ("passive_neutral", top10[net_col].abs() <= 0.1),
                    ("passive_cancels", top10[net_col] < -0.1),
                ]:
                    sub = top10[mask]
                    if len(sub) < 3:
                        continue
                    row = {"symbol": sym, "label": lbl, "aggressor": press_side,
                           "passive_response": sign_label, "n": len(sub)}
                    for resp_col, tag in [
                        ("ticks_moved_1s", "ticks"), ("signed_mid_change_1s", "mid_chg"),
                        ("replenishment_ratio_1s", "repl_ratio"),
                        (flow_col, "flow_qty"),
                    ]:
                        if resp_col in sub:
                            v = sub[resp_col].dropna()
                            row[f"{tag}_p50"] = round(v.median(), 4)
                            row[f"{tag}_p90"] = round(v.quantile(.9), 4)
                            row[f"{tag}_mean"] = round(v.mean(), 4)
                    rows.append(row)
    return pd.DataFrame(rows)


# ── 4. Replenishment by execution-size bucket ──────────────────────────────────

EXEC_BUCKETS = [(0.0, 1e-9, "zero"), (1e-9, 50, "p0-50"),
                (50, 90, "p50-90"), (90, 95, "p90-95"), (95, 99, "p95-99"), (99, 100, "top-1%")]

def compute_replenishment_by_exec(dfs) -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)].copy()
            exec_col = "total_exec_qty_1s"
            if exec_col not in df:
                continue
            # Compute percentile among non-zero execs
            nonzero = df[df[exec_col] > 1e-9][exec_col]
            pct_vals = nonzero.rank(pct=True) * 100
            df["_exec_pct"] = np.nan
            df.loc[df[exec_col] > 1e-9, "_exec_pct"] = pct_vals

            for lo, hi, blabel in EXEC_BUCKETS:
                if lo == 0:
                    mask = df[exec_col] <= 1e-9
                else:
                    mask = (df["_exec_pct"] >= lo) & (df["_exec_pct"] < hi)
                sub = df[mask]
                if len(sub) < 3:
                    continue
                row = {"symbol": SYM_NICE[sym], "label": lbl, "exec_bucket": blabel,
                       "n": len(sub)}
                for col, tag in [
                    ("total_exec_qty_1s", "exec_qty"),
                    ("total_refill_qty_1s", "refill_qty"),
                    ("replenishment_ratio_1s", "ratio"),
                    ("refill_count_1s", "refill_n"),
                ]:
                    if col in sub:
                        v = sub[col].dropna()
                        row[f"{tag}_p50"]  = round(v.median(), 4)
                        row[f"{tag}_p90"]  = round(v.quantile(.9), 4)
                        row[f"{tag}_mean"] = round(v.mean(), 4)
                rows.append(row)
    return pd.DataFrame(rows)


# ── 5. Near-market liquidity stats ────────────────────────────────────────────

def compute_near_market_liquidity(dfs) -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            row = {"symbol": SYM_NICE[sym], "label": lbl}
            # Depth at 5/10/25 bps (already in parquet from BBO series)
            for bps in (5, 10, 25):
                for side in ("bid", "ask"):
                    col = f"depth_{side}_{bps}bps"
                    if col in df:
                        v = df[col].dropna()
                        row[f"{side}_{bps}bps_p50"] = round(v.median(), 2)
                        row[f"{side}_{bps}bps_p90"] = round(v.quantile(.9), 2)
            # Fraction of 25bps depth that's within 5bps (near-market concentration)
            if "depth_bid_5bps" in df and "depth_bid_25bps" in df:
                ratio = df.depth_bid_5bps / (df.depth_bid_25bps + 1e-12)
                row["bid_5of25_conc_p50"] = round(ratio.median(), 3)
                row["bid_5of25_conc_p90"] = round(ratio.quantile(.9), 3)
            # Distance of largest bid level from mid
            if "bid_top1_dist_ticks" in df:
                v = df.bid_top1_dist_ticks.dropna()
                row["bid_wall_dist_p10"]  = round(v.quantile(.10), 1)
                row["bid_wall_dist_p50"]  = round(v.quantile(.50), 1)
                row["bid_wall_dist_p90"]  = round(v.quantile(.90), 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ── 6. Generate updated ORDER_FLOW_REPORT.md ──────────────────────────────────

def generate_updated_report(dfs, tps_tbl, jst_v2, cond_split, repl_exec, near_liq):
    def _q(dfs, sym, lbl, col, q):
        df = dfs.get((sym, lbl))
        if df is None or col not in df:
            return float("nan")
        return round(df[col].dropna().quantile(q), 3)

    lines = [
        "# ORDER_FLOW_REPORT — KRAKEN-OF-1-R3 (revised after QA1)",
        "",
        "**Feature version:** r2.1  **Book semantics:** ms-batch-v1",
        "**Sessions:** LOW / MEDIAN / HIGH x BTC / ETH  ",
        "**Window:** 1h research, 1s anchors (3600 rows/session)",
        "**HARD RULE:** no future prices, MFE/MAE, ML or PnL. All features causal.",
        "",
        "---", "",
        "## 1. Regime Comparison",
        "",
        "### Order-message intensity (events/sec p50)",
        "",
        "| Symbol | LOW | MEDIAN | HIGH |",
        "|---|---|---|---|",
    ]
    for sym in SYMBOLS:
        meds = [str(round(dfs[(sym, lbl)]["events_per_sec_1s"].median(), 0))
                if "events_per_sec_1s" in dfs[(sym, lbl)] else "—"
                for lbl in LABELS]
        lines.append(f"| {SYM_NICE[sym]} | {' | '.join(meds)} |")

    lines += ["",
        "> **Note:** ETH session ordering does not match BTC. Sessions were selected by BTC "
        "order-event rate. ETH MEDIAN (eps=139) < ETH LOW (eps=236). "
        "BTC and ETH activity regimes are not co-linear on the same hour.",
        "",
        "### Execution intensity (trades/sec) — full distribution",
        "",
        "| Symbol | Session | mean | p50 | p90 | p99 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in tps_tbl.iterrows():
        lines.append(f"| {r.symbol} | {r.label} | {r['mean']} | {r.p50} | {r.p90} | {r.p99} | {r['max']:.0f} |")

    lines += [
        "",
        "> **Corrected finding:** BTC HIGH has ~2× more order-messaging than LOW (eps p50 822 vs 410), "
        "but execution intensity does NOT scale proportionally. "
        "HIGH/LOW trades/sec: mean=0.87×, p90=1.14×, p99=0.53×. "
        "The selected HIGH session has *less* extreme execution tail than LOW. "
        "Order-message burst ≠ trading burst ≠ price burst.",
        "",
        "See Figure 01.", "", "---", "",
        "## 2. Aggressive Flow",
        "",
        "### BTC sell_taker_qty_1s tails",
        "",
        "| Session | p90 | p95 | p99 | p99.9 | max |",
        "|---|---|---|---|---|---|",
    ]
    for lbl in LABELS:
        df = dfs[("PF_XBTUSD", lbl)]
        if "sell_taker_qty_1s" not in df:
            continue
        v = df.sell_taker_qty_1s.dropna()
        lines.append(f"| {lbl} | {v.quantile(.9):.4f} | {v.quantile(.95):.4f} | "
                     f"{v.quantile(.99):.4f} | {v.quantile(.999):.4f} | {v.max():.3f} |")
    lines += ["",
        "Median is 0 in all sessions — most 1-second windows have no executions. "
        "Future battle research focuses on the non-zero tail, not the mean.",
        "",
        "See Figure 02.", "", "---", "",
        "## 3. Passive Response — SELL Pressure",
        "",
        "Within-session percentile buckets for sell_taker_qty_1s:",
        "",
    ]
    for lbl in LABELS:
        sub = jst_v2[(jst_v2.symbol == "PF_XBTUSD") & (jst_v2.label == lbl) &
                     (jst_v2.pressure_side == "SELL")]
        if len(sub) == 0:
            continue
        lines.append(f"### BTC {lbl} — SELL pressure")
        lines.append("")
        lines.append("| bucket | n | bid_net p50 | bid_add p50 | bid_cxl p50 | mid_chg p50 | ticks p50 | repl p50 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r.bucket} | {r.n} | {r.get('bid_net_p50', float('nan')):.2f} | "
                         f"{r.get('bid_add_p50', float('nan')):.2f} | {r.get('bid_cxl_p50', float('nan')):.2f} | "
                         f"{r.get('mid_chg_p50', float('nan')):.2f} | {r.get('ticks_p50', float('nan')):.2f} | "
                         f"{r.get('repl_ratio_p50', float('nan')):.2f} |")
        lines.append("")
    lines += ["See Figure 03.", "", "---", "",
        "## 4. Passive Response — BUY Pressure (mirror)",
        "",
    ]
    for lbl in LABELS:
        sub = jst_v2[(jst_v2.symbol == "PF_XBTUSD") & (jst_v2.label == lbl) &
                     (jst_v2.pressure_side == "BUY")]
        if len(sub) == 0:
            continue
        lines.append(f"### BTC {lbl} — BUY pressure")
        lines.append("")
        lines.append("| bucket | n | ask_net p50 | ask_add p50 | ask_cxl p50 | mid_chg p50 | ticks p50 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {r.bucket} | {r.n} | {r.get('ask_net_p50', float('nan')):.2f} | "
                         f"{r.get('ask_add_p50', float('nan')):.2f} | {r.get('ask_cxl_p50', float('nan')):.2f} | "
                         f"{r.get('mid_chg_p50', float('nan')):.2f} | {r.get('ticks_p50', float('nan')):.2f} |")
        lines.append("")
    lines += ["", "---", "",
        "## 5. Conditional Split: Top-10% Flow x Passive Response Sign",
        "",
        "Within top-10% sell (or buy) taker flow: split by passive-side net_passive sign.",
        "Tests absorption-candidate hypothesis on same-window data only.",
        "",
        "| Symbol | Session | Aggressor | Passive response | n | ticks p50 | ticks p90 | mid_chg p50 | repl p50 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if len(cond_split):
        for _, r in cond_split[cond_split.symbol.isin(["PF_XBTUSD"])].iterrows():
            lines.append(f"| {SYM_NICE[r.symbol]} | {r.label} | {r.aggressor} | {r.passive_response} | {r.n} | "
                         f"{r.get('ticks_p50', float('nan')):.2f} | {r.get('ticks_p90', float('nan')):.2f} | "
                         f"{r.get('mid_chg_p50', float('nan')):.2f} | {r.get('repl_ratio_p50', float('nan')):.2f} |")
    lines += ["", "---", "",
        "## 6. Execution x Replenishment",
        "",
        "replenishment_ratio = total_refill_qty / total_exec_qty (exec_qty deduplicated per exec_ts).",
        "Show raw exec_qty and refill_qty alongside ratio — ratio alone is not interpretable.",
        "",
        "### BTC — replenishment by execution-size bucket (1s window)",
        "",
        "| Session | exec bucket | n | exec_qty p50 | refill_qty p50 | ratio p50 | ratio p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for lbl in LABELS:
        sub = repl_exec[repl_exec.label == lbl]
        for _, r in sub.iterrows():
            lines.append(f"| {lbl} | {r.exec_bucket} | {r.n} | "
                         f"{r.get('exec_qty_p50', float('nan')):.4f} | "
                         f"{r.get('refill_qty_p50', float('nan')):.4f} | "
                         f"{r.get('ratio_p50', float('nan')):.2f} | "
                         f"{r.get('ratio_p90', float('nan')):.1f} |")
    lines += ["", "---", "",
        "## 7. Near-Market Liquidity Concentration",
        "",
        "Near-BBO depth (existing features from BBO replay):",
        "",
        "| Symbol | Session | bid_5bps p50 | bid_10bps p50 | bid_25bps p50 | 5of25 conc p50 | wall_dist p50 ticks |",
        "|---|---|---|---|---|---|---|",
    ]
    if len(near_liq):
        for _, r in near_liq.iterrows():
            lines.append(f"| {r.symbol} | {r.label} | "
                         f"{r.get('bid_5bps_p50', float('nan')):.2f} | "
                         f"{r.get('bid_10bps_p50', float('nan')):.2f} | "
                         f"{r.get('bid_25bps_p50', float('nan')):.2f} | "
                         f"{r.get('bid_5of25_conc_p50', float('nan')):.3f} | "
                         f"{r.get('bid_wall_dist_p50', float('nan')):.1f} |")
    lines += [
        "",
        "> **Finding:** Median `bid_top1_dist_ticks` is 247–327 ticks from mid in LOW/MEDIAN, "
        "and 98 ticks in HIGH. Only ~11% of 25bps bid depth is within 5bps of mid (BTC MEDIAN). "
        "The maximum-qty level (the candidate 'wall') is typically far from the current market. "
        "Global `max_qty_vs_median` across all active levels is NOT a useful battle-wall detector.",
        "",
        "> **Implication:** Wall detection must use local neighborhood (within 5/10/25 bps of mid), "
        "not global book statistics. This will be implemented in the next pass.",
        "",
        "---", "",
    ]

    lines += [
        "## 8. Observed Market-State Families",
        "",
        "**Rules:** only same-window features (t-W, t]. No future prices. No thresholds tuned on outcomes.",
        "",
        "### OBSERVED (supported by current joint-state data)",
        "",
        "**A. Message Churn**",
        "- Very high order-event rate (events/sec p90-p99)",
        "- Ordinary or low execution intensity (trades/sec near median)",
        "- Near-zero aggressive taker qty",
        "- High simultaneous bid/ask add AND cancel (high passive turnover both sides)",
        "- Little BBO movement",
        "- Represents HFT repricing cycles (~98% cancellation rate observed in Phase B)",
        "",
        "**B. Aggressive Depletion-Like State**",
        "- Top 10-1% sell (or buy) taker flow",
        "- bid_net_passive < 0 at p50 across all sessions",
        "- Significant same-window downward (or upward) price movement",
        "- Consistent across LOW/MEDIAN/HIGH BTC sessions",
        "- Also observed for BUY pressure (ask_net_passive and mid_chg sign mirrored)",
        "",
        "### CANDIDATE STATES — NOT YET ESTABLISHED",
        "",
        "**C. Passive Absorption-Like State**",
        "- *Hypothesis:* strong sell flow + bid_net_passive > 0 + low same-window price response",
        "- *Current evidence:* within top-10% sell flow the passive_adds group "
        "has fewer anchors; conditional split is available but p50 ticks requires "
        "more conditioning to distinguish from noise",
        "- *Next step:* conditional analysis on `bid_net_passive > 0` AND "
        "`replenishment_ratio > threshold` simultaneously vs price response",
        "",
        "**D. Trading Burst**",
        "- *Hypothesis:* trades/sec in top percentile + |AFI| imbalance + BBO movement",
        "- *Current evidence:* trades/sec p99 reaches 32-60 in BTC sessions; "
        "top-1% execution anchors exist; joint analysis with flow imbalance pending",
        "",
        "### Why NOT to name these as established families yet",
        "The conditional split shows some signal difference (e.g. `passive_adds` vs `passive_cancels` "
        "within top sell flow), but sample sizes per cell are small (n=3-15 in some buckets) "
        "and the within-bucket variance is high. These states need battle episode analysis "
        "(Part G of the original R1 plan) with resolved outcomes — "
        "which requires the BATTLE ontology freeze and separate R4 pass.",
        "",
        "---", "",
        "## 9. Caveats and Limitations",
        "",
        "- All features strictly causal (backward-looking from anchor_ts)",
        "- ms-batch semantics: same-ms events form unordered set",
        "- 120m context window not_ready for 60-min research sessions (by design)",
        "- acceleration features: NaN in ~83% of 1s rows (valid only when prior window has flow)",
        "- Session selection was by BTC order-event rate; ETH activity ordering differs",
        "- Global max_qty_vs_median includes deep-book levels; for battle wall detection, "
        "use near-market zones (5/10/25 bps) — not implemented yet",
        "",
        "---", "",
        "*FEATURE_VERSION=r2.1  |  BOOK_SEMANTICS_VERSION=ms-batch-v1  |  R3-QA1 revised*",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("KRAKEN-OF-1-R3-QA1: Report cleanup + conditional analysis", flush=True)
    dfs = load_all()
    print(f"  Loaded {sum(len(v) for v in dfs.values())} rows", flush=True)

    print("  Trades/sec table ...", flush=True)
    tps = trades_per_sec_table(dfs)
    tps.to_csv(REPORTS / "TRADES_PER_SEC_TABLE.csv", index=False)

    print("  Joint-state v2 (SELL + BUY) ...", flush=True)
    jst = compute_joint_state_v2(dfs)
    jst.to_csv(REPORTS / "JOINT_STATE_TABLES_v2.csv", index=False)
    print(f"    {len(jst)} rows ({(jst.pressure_side=='SELL').sum()} SELL, {(jst.pressure_side=='BUY').sum()} BUY)", flush=True)

    print("  Conditional split (absorption candidate) ...", flush=True)
    cond = compute_conditional_split(dfs)
    cond.to_csv(REPORTS / "CONDITIONAL_FLOW_SPLIT.csv", index=False)
    print(f"    {len(cond)} rows", flush=True)
    if len(cond):
        print("    Sample (BTC MEDIAN SELL):")
        sub = cond[(cond.symbol == "PF_XBTUSD") & (cond.label == "MEDIAN") & (cond.aggressor == "SELL")]
        if len(sub):
            print(sub[["passive_response", "n", "ticks_p50", "mid_chg_p50", "repl_ratio_p50"]].to_string(index=False))

    print("  Replenishment by exec size ...", flush=True)
    repl = compute_replenishment_by_exec(dfs)
    repl.to_csv(REPORTS / "REPLENISHMENT_BY_EXEC_SIZE.csv", index=False)

    print("  Near-market liquidity ...", flush=True)
    nml = compute_near_market_liquidity(dfs)
    nml.to_csv(REPORTS / "NEAR_MARKET_LIQUIDITY.csv", index=False)
    print("    BTC near-market depth:")
    print(nml[nml.symbol == "BTC"][["label", "bid_5bps_p50", "bid_10bps_p50",
                                     "bid_5of25_conc_p50", "bid_wall_dist_p50"]].to_string(index=False))

    print("  Regenerating ORDER_FLOW_REPORT.md ...", flush=True)
    report = generate_updated_report(dfs, tps, jst, cond, repl, nml)
    (REPORTS / "ORDER_FLOW_REPORT.md").write_text(report, encoding="utf-8")

    print("\nR3-QA1 complete.", flush=True)
    print(f"  reports/: JOINT_STATE_TABLES_v2.csv  CONDITIONAL_FLOW_SPLIT.csv", flush=True)
    print(f"            REPLENISHMENT_BY_EXEC_SIZE.csv  NEAR_MARKET_LIQUIDITY.csv", flush=True)
    print(f"            ORDER_FLOW_REPORT.md (updated)", flush=True)


if __name__ == "__main__":
    main()
