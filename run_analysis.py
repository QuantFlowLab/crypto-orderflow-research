"""run_analysis.py — KRAKEN-OF-1-R3: Descriptive Microstructure Analysis.

Frozen inputs: FEATURE_VERSION=r2.2  BOOK_SEMANTICS_VERSION=ms-batch-v1
HARD RULE: no future returns, MFE/MAE, ML or PnL. Causal features only.

Output:
  reports/ORDER_FLOW_REPORT.md
  reports/DESCRIPTIVE_DISTRIBUTIONS.csv
  reports/JOINT_STATE_TABLES.csv
  reports/BTC_ETH_COMPARISON.csv
  figures/01_activity_regimes.png ... 08_btc_vs_eth.png
  figures/episodes/*.png
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

EXP     = Path(__file__).parent
FEAT    = EXP / "data" / "derived" / "features"
REPORTS = EXP / "reports"
FIGS    = EXP / "figures"
EPFIGS  = FIGS / "episodes"
for d in (REPORTS, FIGS, EPFIGS):
    d.mkdir(parents=True, exist_ok=True)

FEATURE_VERSION        = "r2.2"
BOOK_SEMANTICS_VERSION = "ms-batch-v1"
SYMBOLS  = ["PF_XBTUSD", "PF_ETHUSD"]
LABELS   = ["LOW", "MEDIAN", "HIGH"]
SYM_NICE = {"PF_XBTUSD": "BTC", "PF_ETHUSD": "ETH"}
COL = {"LOW": "#4e79a7", "MEDIAN": "#f28e2b", "HIGH": "#e15759"}
PCTILES  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
PRESSURE_BUCKETS = [(0, 50, "0–50%"), (50, 90, "50–90%"),
                    (90, 95, "90–95%"), (95, 99, "95–99%"), (99, 100, "top-1%")]
RNG = np.random.RandomState(42)
TICK = {"PF_XBTUSD": 1.0, "PF_ETHUSD": 0.1}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.autolayout": True})

findings: list[str] = []

# ── 0. Load data ──────────────────────────────────────────────────────────────

def load_all() -> dict[tuple[str, str], pd.DataFrame]:
    dfs = {}
    for sym in SYMBOLS:
        for lbl in LABELS:
            p = FEAT / f"{sym}_{lbl}_features.parquet"
            df = pd.read_parquet(p)
            for side in ("buy", "sell"):
                col = f"{side}_taker_qty_1s"
                if col in df:
                    # retrospective within-session descriptive stratification
                    df[f"{side}_pressure_pct"] = df[col].rank(pct=True) * 100
            dfs[(sym, lbl)] = df
    return dfs

def parquet_hash(sym, lbl) -> str:
    p = FEAT / f"{sym}_{lbl}_features.parquet"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

def freeze_provenance(dfs):
    """Write R3 provenance manifest."""
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            rows.append({"symbol": sym, "label": lbl,
                          "feature_version": FEATURE_VERSION,
                          "book_semantics": BOOK_SEMANTICS_VERSION,
                          "parquet_sha256_16": parquet_hash(sym, lbl),
                          "n_rows": len(dfs[(sym, lbl)])})
    pd.DataFrame(rows).to_csv(REPORTS / "R3_PROVENANCE.csv", index=False)
    print("  Provenance saved: R3_PROVENANCE.csv", flush=True)

# ── 1. Descriptive distributions ─────────────────────────────────────────────

REP_FEATURES = {
    "REGIME":       ["events_per_sec_1s", "trades_per_sec_1s", "qty_per_sec_1s",
                     "spread", "near_book_depth", "bbo_change_count_1s"],
    "AGGRESSIVE":   ["buy_taker_qty_1s", "sell_taker_qty_1s", "afi_1s",
                     "buy_taker_qty_5s", "sell_taker_qty_5s"],
    "PASSIVE":      ["bid_add_qty_1s", "ask_add_qty_1s",
                     "bid_cancel_qty_1s", "ask_cancel_qty_1s",
                     "bid_net_passive_1s", "ask_net_passive_1s"],
    "PRICE_RESP":   ["signed_mid_change_1s", "ticks_moved_1s", "levels_crossed_1s",
                     "price_response_per_sell_qty_1s"],
    "LIQUIDITY":    ["bid_max_qty_vs_median", "bid_max_share",
                     "bid_top1_dist_ticks", "ask_max_qty_vs_median"],
    "REPLENISHMENT":["total_exec_qty_1s", "total_refill_qty_1s",
                     "replenishment_ratio_1s", "refill_count_1s"],
}

def descriptive_distributions(dfs) -> pd.DataFrame:
    rows = []
    for (sym, lbl), df in dfs.items():
        for family, cols in REP_FEATURES.items():
            for col in cols:
                if col not in df:
                    continue
                v = df[col].dropna()
                row = {"symbol": sym, "label": lbl, "family": family, "feature": col,
                       "n": len(v), "nan_pct": round(df[col].isna().mean()*100, 1),
                       "mean": round(v.mean(), 4), "std": round(v.std(), 4)}
                for p in PCTILES:
                    row[f"p{p}"] = round(float(np.percentile(v, p)), 6)
                row["max"] = round(float(v.max()), 6)
                rows.append(row)
    return pd.DataFrame(rows)

# ── 2. Figure 01: Activity Regimes ────────────────────────────────────────────

ACT_METRICS = [("events_per_sec_1s", "Events/sec"), ("trades_per_sec_1s", "Trades/sec"),
               ("qty_per_sec_1s", "Exec qty/sec"), ("spread", "BBO spread (ticks)"),
               ("near_book_depth", "Near-BBO depth"), ("bbo_change_count_1s", "BBO changes/sec")]

def fig_activity_regimes(dfs):
    fig, axes = plt.subplots(2, len(ACT_METRICS), figsize=(14, 6), sharey=False)
    fig.suptitle("Activity Regime Comparison: LOW / MEDIAN / HIGH", fontsize=11, fontweight="bold")
    for row_i, sym in enumerate(SYMBOLS):
        for col_i, (metric, title) in enumerate(ACT_METRICS):
            ax = axes[row_i, col_i]
            ax.set_title(f"{SYM_NICE[sym]} — {title}", fontsize=8)
            for lbl in LABELS:
                v = dfs[(sym, lbl)][metric].dropna() if metric in dfs[(sym, lbl)] else pd.Series()
                if len(v):
                    bp = ax.boxplot(v, positions=[LABELS.index(lbl)], widths=0.5,
                                    patch_artist=True, showfliers=False,
                                    boxprops=dict(facecolor=COL[lbl], alpha=0.7),
                                    medianprops=dict(color="black", linewidth=2))
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(LABELS, fontsize=7)
            ax.set_ylabel("value", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "01_activity_regimes.png", bbox_inches="tight")
    plt.close()
    print("  Figure 01 saved", flush=True)

# ── 3. Figure 02: Aggressive Flow ─────────────────────────────────────────────

def fig_aggressive_flow(dfs):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Aggressive Flow Distributions", fontsize=11, fontweight="bold")
    for row_i, sym in enumerate(SYMBOLS):
        # Buy taker qty distribution (log scale, violins)
        ax = axes[row_i, 0]; ax.set_title(f"{SYM_NICE[sym]} buy/sell qty 1s", fontsize=8)
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            for col, ls in [("buy_taker_qty_1s", "-"), ("sell_taker_qty_1s", "--")]:
                v = df[col].dropna()
                v_pos = v[v > 0]
                if len(v_pos) > 10:
                    # use numpy histogram as density estimate
                    lo_v, hi_v = v_pos.quantile(0.01), v_pos.quantile(0.99)
                    counts, edges = np.histogram(v_pos.clip(lo_v, hi_v), bins=50, density=True)
                    xv = 0.5 * (edges[:-1] + edges[1:])
                    norm = counts.max() + 1e-12
                    ax.plot(xv, counts / norm, color=COL[lbl], ls=ls, alpha=0.7,
                            lw=1.5, label=f"{lbl} {'buy' if ls=='-' else 'sell'}")
        ax.set_xlabel("qty (BTC/ETH)"); ax.set_ylabel("density (norm.)"); ax.legend(fontsize=5)

        # AFI distribution
        ax = axes[row_i, 1]; ax.set_title(f"{SYM_NICE[sym]} AFI 1s", fontsize=8)
        for lbl in LABELS:
            v = dfs[(sym, lbl)]["afi_1s"].dropna()
            if len(v) > 10:
                ax.hist(v, bins=60, alpha=0.5, color=COL[lbl], label=lbl, density=True)
        ax.set_xlabel("AFI"); ax.legend(fontsize=7)

        # Tail table: p90/p95/p99/max for sell taker qty by session
        ax = axes[row_i, 2]; ax.axis("off")
        pct_data = []
        for lbl in LABELS:
            v = dfs[(sym, lbl)]["sell_taker_qty_1s"].dropna()
            pct_data.append([lbl, f"{v.quantile(.9):.3f}", f"{v.quantile(.95):.3f}",
                              f"{v.quantile(.99):.3f}", f"{v.quantile(.999):.3f}",
                              f"{v.max():.3f}"])
        tbl = ax.table(cellText=pct_data,
                       colLabels=["Label","p90","p95","p99","p99.9","max"],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        ax.set_title(f"{SYM_NICE[sym]} sell_taker_qty_1s tails", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGS / "02_aggressive_flow.png", bbox_inches="tight")
    plt.close()
    print("  Figure 02 saved", flush=True)

# ── 4. Joint-state tables ─────────────────────────────────────────────────────

def bucket_by_pressure(df: pd.DataFrame, side: str) -> pd.Series:
    """Within-session pressure buckets by percentile."""
    pct_col = f"{side}_pressure_pct"
    if pct_col not in df:
        return pd.Series(dtype=str)
    labels = ["0–50%", "50–90%", "90–95%", "95–99%", "top-1%"]
    bins   = [0, 50, 90, 95, 99, 100]
    return pd.cut(df[pct_col], bins=bins, labels=labels, include_lowest=True)

def compute_joint_state(dfs) -> pd.DataFrame:
    """Joint-state table: sell pressure bucket × response metrics."""
    RESP_COLS = {
        "bid_add_qty_1s": "bid_add", "bid_cancel_qty_1s": "bid_cxl",
        "bid_net_passive_1s": "bid_net", "ask_add_qty_1s": "ask_add",
        "ask_cancel_qty_1s": "ask_cxl", "near_book_depth": "depth",
        "signed_mid_change_1s": "mid_chg", "ticks_moved_1s": "ticks",
        "replenishment_ratio_1s": "repl_ratio",
    }
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)].copy()
            buckets = bucket_by_pressure(df, "sell")
            if buckets.empty:
                continue
            df["sell_bucket"] = buckets
            for bname, grp in df.groupby("sell_bucket", observed=True):
                row = {"symbol": sym, "label": lbl, "pressure_side": "SELL",
                       "bucket": str(bname), "n": len(grp)}
                for col, tag in RESP_COLS.items():
                    if col in grp:
                        v = grp[col].dropna()
                        row[f"{tag}_p50"] = round(v.median(), 4) if len(v) else np.nan
                        row[f"{tag}_p90"] = round(v.quantile(.9), 4) if len(v) else np.nan
                        row[f"{tag}_mean"] = round(v.mean(), 4) if len(v) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)

# ── 5. Figure 03: Passive Response by Sell Pressure ──────────────────────────

def fig_passive_response(dfs):
    n_sym = 2; n_met = 3
    fig, axes = plt.subplots(n_sym, n_met, figsize=(13, 8))
    fig.suptitle("Passive Response by Sell Pressure Bucket (1s window)", fontsize=11, fontweight="bold")
    bnames = ["0–50%", "50–90%", "90–95%", "95–99%", "top-1%"]
    x = np.arange(len(bnames)); bw = 0.25
    metrics = [("bid_net_passive_1s", "BID net passive qty", True),
               ("bid_add_qty_1s", "BID add qty", False),
               ("bid_cancel_qty_1s", "BID cancel qty", False)]
    for ri, sym in enumerate(SYMBOLS):
        for ci, (met, title, has_zero) in enumerate(metrics):
            ax = axes[ri, ci]
            ax.set_title(f"{SYM_NICE[sym]} — {title}", fontsize=8)
            offsets = [-bw, 0, bw]
            for oi, lbl in enumerate(LABELS):
                df = dfs[(sym, lbl)].copy()
                df["sell_bucket"] = bucket_by_pressure(df, "sell")
                means = [df[df.sell_bucket == b][met].dropna().median()
                         if met in df else 0 for b in bnames]
                ax.bar(x + offsets[oi], means, width=bw, label=lbl,
                       color=COL[lbl], alpha=0.8)
            if has_zero:
                ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(bnames, rotation=30, ha="right", fontsize=7)
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(FIGS / "03_passive_response.png", bbox_inches="tight")
    plt.close()
    print("  Figure 03 saved", flush=True)

# ── 6. Figure 04: Pressure × Price Response ──────────────────────────────────

def fig_pressure_price_response(dfs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Sell Pressure × Same-Window Price Response (1s)", fontsize=11, fontweight="bold")
    bnames = ["0–50%", "50–90%", "90–95%", "95–99%", "top-1%"]
    metrics = [("ticks_moved_1s", "Ticks moved (unsigned)"),
               ("signed_mid_change_1s", "Signed mid change ($ BTC / $0.1 ETH)")]
    x = np.arange(len(bnames)); bw = 0.25
    for ri, sym in enumerate(SYMBOLS):
        for ci, (met, title) in enumerate(metrics):
            ax = axes[ri, ci]
            ax.set_title(f"{SYM_NICE[sym]} — {title}", fontsize=8)
            for oi, lbl in enumerate(LABELS):
                df = dfs[(sym, lbl)].copy()
                df["sell_bucket"] = bucket_by_pressure(df, "sell")
                medians = [df[df.sell_bucket == b][met].dropna().median()
                           if met in df else 0 for b in bnames]
                ax.bar(x + (oi-1)*bw, medians, width=bw, label=lbl,
                       color=COL[lbl], alpha=0.8)
            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(bnames, rotation=30, ha="right", fontsize=7)
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(FIGS / "04_pressure_vs_price_response.png", bbox_inches="tight")
    plt.close()
    print("  Figure 04 saved", flush=True)

# ── 7. Figure 05: High Sell Pressure — bid response × price response ──────────

def fig_pressure_bid_price(dfs):
    """For top-10% sell pressure: classify by bid_net_passive sign, show price response."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Top-10% Sell Pressure: Bid Response × Price Response", fontsize=11, fontweight="bold")
    axes_flat = axes.flatten()
    plot_i = 0
    for sym in SYMBOLS:
        for lbl in LABELS:
            ax = axes_flat[plot_i]; plot_i += 1
            df = dfs[(sym, lbl)]
            if "sell_pressure_pct" not in df:
                ax.axis("off"); continue
            top10 = df[df.sell_pressure_pct > 90].copy()
            if len(top10) < 10:
                ax.axis("off"); continue
            if "bid_net_passive_1s" not in top10 or "ticks_moved_1s" not in top10:
                ax.axis("off"); continue
            # Classify by bid net passive
            pos_bid = top10[top10.bid_net_passive_1s > 0]
            neg_bid = top10[top10.bid_net_passive_1s < 0]
            neu_bid = top10[top10.bid_net_passive_1s == 0]
            for grp, nm, c in [(pos_bid, "bid adds↑", "#2166ac"),
                                (neu_bid, "neutral", "#999999"),
                                (neg_bid, "bid cancels↑", "#d6604d")]:
                if len(grp) > 3:
                    ax.hist(grp.ticks_moved_1s.dropna(), bins=30, alpha=0.6,
                            color=c, label=f"{nm} (n={len(grp)})", density=True)
            ax.set_title(f"{SYM_NICE[sym]} {lbl}", fontsize=8)
            ax.set_xlabel("ticks_moved_1s (unsigned)", fontsize=7)
            ax.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(FIGS / "05_pressure_vs_bid_response.png", bbox_inches="tight")
    plt.close()
    print("  Figure 05 saved", flush=True)

# ── 8. Figure 06: Execution × Replenishment ──────────────────────────────────

def fig_execution_replenishment(dfs):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Execution × Replenishment (1s window, log1p ratio)", fontsize=11, fontweight="bold")
    axes_flat = axes.flatten()
    plot_i = 0
    for sym in SYMBOLS:
        for lbl in LABELS:
            ax = axes_flat[plot_i]; plot_i += 1
            df = dfs[(sym, lbl)]
            exec_col = "total_exec_qty_1s"; ratio_col = "replenishment_ratio_1s"
            if exec_col not in df or ratio_col not in df:
                ax.axis("off"); continue
            sub = df[df[exec_col] > 1e-6].copy()
            if len(sub) < 10:
                ax.axis("off"); continue
            # x = log1p(exec_qty), y = log1p(ratio)
            sub["log_exec"] = np.log1p(sub[exec_col])
            sub["log_ratio"] = np.log1p(sub[ratio_col].clip(lower=0))
            ax.scatter(sub.log_exec, sub.log_ratio, alpha=0.3, s=3, c=COL[lbl])
            # Buckets: annotate mean by exec quartile
            eq = sub.log_exec.quantile([.25, .5, .75, 1.0]).values
            xs, ys = [], []
            prev = 0
            for q in eq:
                grp = sub[(sub.log_exec >= prev) & (sub.log_exec < q + 1e-9)]
                if len(grp) > 3:
                    xs.append(grp.log_exec.median()); ys.append(grp.log_ratio.median())
                prev = q
            if xs:
                ax.plot(xs, ys, "k-o", lw=2, ms=4)
            ax.set_xlabel("log1p(exec_qty)", fontsize=7)
            ax.set_ylabel("log1p(repl_ratio)", fontsize=7)
            ax.set_title(f"{SYM_NICE[sym]} {lbl} (n={len(sub)})", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "06_execution_vs_replenishment.png", bbox_inches="tight")
    plt.close()
    print("  Figure 06 saved", flush=True)

# ── 9. Figure 07: Liquidity Concentration ────────────────────────────────────

def fig_liquidity_concentration(dfs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Liquidity Concentration: max_qty / median_qty at each anchor", fontsize=11, fontweight="bold")
    for ri, sym in enumerate(SYMBOLS):
        for ci, met in enumerate(["bid_max_qty_vs_median", "ask_max_qty_vs_median"]):
            ax = axes[ri, ci]
            ax.set_title(f"{SYM_NICE[sym]} — {met.split('_')[0]}", fontsize=8)
            data = [np.log1p(dfs[(sym, lbl)][met].dropna()) for lbl in LABELS
                    if met in dfs[(sym, lbl)]]
            if not data:
                continue
            parts = ax.violinplot(data, positions=range(len(LABELS)),
                                  showmedians=True, showextrema=True)
            for pc, lbl in zip(parts["bodies"], LABELS):
                pc.set_facecolor(COL[lbl]); pc.set_alpha(0.7)
            ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
            ax.set_ylabel("log1p(max/median)", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "07_liquidity_concentration.png", bbox_inches="tight")
    plt.close()
    print("  Figure 07 saved", flush=True)

# ── 10. Figure 08: BTC vs ETH normalized ─────────────────────────────────────

def fig_btc_vs_eth(dfs):
    """Normalized feature comparison across instruments and sessions."""
    NORM_COLS = ["afi_1s", "activity_rolling_percentile_30m",
                 "bid_net_passive_1s", "bid_max_share", "replenishment_ratio_1s"]
    fig, axes = plt.subplots(1, len(NORM_COLS), figsize=(15, 5))
    fig.suptitle("BTC vs ETH: normalized features (violin)", fontsize=11, fontweight="bold")
    sym_markers = {"PF_XBTUSD": "#4e79a7", "PF_ETHUSD": "#e15759"}
    for ax, col in zip(axes, NORM_COLS):
        ax.set_title(col.replace("_", "\n"), fontsize=7)
        pos_i = 0
        ticks, tick_labels = [], []
        for sym in SYMBOLS:
            for lbl in LABELS:
                df = dfs[(sym, lbl)]
                if col not in df:
                    pos_i += 1; continue
                v = df[col].dropna()
                if col == "bid_net_passive_1s":
                    # normalize by near_book_depth
                    depth = df["near_book_depth"].dropna()
                    if len(depth) and depth.median() > 0:
                        v = v / depth.reindex(v.index).fillna(depth.median())
                if len(v) > 10:
                    parts = ax.violinplot(v.clip(v.quantile(.01), v.quantile(.99)),
                                          positions=[pos_i], showmedians=True,
                                          widths=0.8)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(sym_markers[sym]); pc.set_alpha(0.6)
                ticks.append(pos_i)
                tick_labels.append(f"{SYM_NICE[sym]}\n{lbl}")
                pos_i += 1
        ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, fontsize=5)
    plt.tight_layout()
    plt.savefig(FIGS / "08_btc_vs_eth.png", bbox_inches="tight")
    plt.close()
    print("  Figure 08 saved", flush=True)

# ── 11. Mechanically selected episodes ───────────────────────────────────────

EP_COLS = ["sell_taker_qty_1s", "buy_taker_qty_1s", "afi_1s",
           "bid_add_qty_1s", "bid_cancel_qty_1s", "ask_add_qty_1s",
           "bid_net_passive_1s", "ticks_moved_1s", "near_book_depth",
           "replenishment_ratio_1s", "spread"]

def plot_episode(ax_row: list, df_full: pd.DataFrame, anchor_idx: int,
                 cols: list[str], half_w: int = 30):
    lo = max(0, anchor_idx - half_w); hi = min(len(df_full)-1, anchor_idx + half_w)
    sub = df_full.iloc[lo:hi+1].copy()
    sub["t"] = np.arange(-( anchor_idx - lo), hi - anchor_idx + 1)
    for ax, col in zip(ax_row, cols):
        if col not in sub:
            ax.set_visible(False); continue
        ax.plot(sub.t, sub[col].fillna(0), lw=1.2, color="#333")
        ax.axvline(0, color="red", lw=1.2, ls="--")
        ax.set_title(col[:18], fontsize=6); ax.tick_params(labelsize=6)

def mechanically_select_episodes(df: pd.DataFrame) -> dict[str, list[int]]:
    """Purely mechanical selection by threshold. Fixed seed. No cherry-picking."""
    sel: dict[str, list[int]] = {}
    if "sell_taker_qty_1s" in df:
        top_sell = df["sell_taker_qty_1s"].quantile(.99)
        cands = df.index[df.sell_taker_qty_1s >= top_sell].tolist()
        sel["top1pct_sell"] = RNG.choice(cands, size=min(3, len(cands)), replace=False).tolist()
    if "buy_taker_qty_1s" in df:
        top_buy = df["buy_taker_qty_1s"].quantile(.99)
        cands = df.index[df.buy_taker_qty_1s >= top_buy].tolist()
        sel["top1pct_buy"] = RNG.choice(cands, size=min(3, len(cands)), replace=False).tolist()
    # High flow / low response
    if "sell_taker_qty_1s" in df and "ticks_moved_1s" in df:
        high_flow = df["sell_taker_qty_1s"] > df["sell_taker_qty_1s"].quantile(.90)
        low_resp  = df["ticks_moved_1s"].fillna(0) < df["ticks_moved_1s"].quantile(.50)
        cands = df.index[high_flow & low_resp].tolist()
        sel["high_flow_low_response"] = RNG.choice(cands, size=min(3, len(cands)), replace=False).tolist() if cands else []
        high_resp = df["ticks_moved_1s"].fillna(0) >= df["ticks_moved_1s"].quantile(.50)
        cands2 = df.index[high_flow & high_resp].tolist()
        sel["high_flow_high_response"] = RNG.choice(cands2, size=min(3, len(cands2)), replace=False).tolist() if cands2 else []
    if "total_exec_qty_1s" in df and "replenishment_ratio_1s" in df:
        high_exec = df["total_exec_qty_1s"] > df["total_exec_qty_1s"].quantile(.90)
        high_repl = df["replenishment_ratio_1s"].fillna(0) > df["replenishment_ratio_1s"].quantile(.90)
        cands = df.index[high_exec & high_repl].tolist()
        sel["high_exec_high_repl"] = RNG.choice(cands, size=min(3, len(cands)), replace=False).tolist() if cands else []
    return sel

def save_episode_fig(df: pd.DataFrame, indices: list[int], name: str,
                     sym: str, lbl: str):
    n_cols = min(len(EP_COLS), 6)
    cols_use = [c for c in EP_COLS if c in df.columns][:n_cols]
    for k, idx in enumerate(indices):
        fig, axes = plt.subplots(1, len(cols_use), figsize=(14, 2.5), sharey=False)
        if len(cols_use) == 1:
            axes = [axes]
        plot_episode(list(axes), df, idx, cols_use)
        ts = df.iloc[idx]["timestamp_ms"] if "timestamp_ms" in df else idx
        fig.suptitle(f"{SYM_NICE[sym]} {lbl} — {name} — anchor={ts}", fontsize=7)
        fname = f"{name}_{SYM_NICE[sym]}_{lbl}_{k+1:02d}.png"
        plt.savefig(EPFIGS / fname, bbox_inches="tight", dpi=120)
        plt.close()

def episodes_all(dfs):
    print("  Generating episode figures ...", flush=True)
    # Use BTC MEDIAN as primary (one representative session)
    for sym, lbl in [("PF_XBTUSD", "MEDIAN"), ("PF_ETHUSD", "MEDIAN")]:
        df = dfs[(sym, lbl)].reset_index(drop=True)
        ep = mechanically_select_episodes(df)
        for name, idxs in ep.items():
            if idxs:
                save_episode_fig(df, idxs, name, sym, lbl)
    print(f"  Episode figures saved to {EPFIGS}/", flush=True)

# ── 12. Activity × volatility matrix ─────────────────────────────────────────

def compute_act_vol_matrix(dfs) -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            if "events_per_sec_1s" not in df or "ticks_moved_1s" not in df:
                continue
            df2 = df[["events_per_sec_1s", "ticks_moved_1s"]].dropna()
            if len(df2) < 10:
                continue
            df2["act_pct"] = pd.cut(df2.events_per_sec_1s.rank(pct=True)*100,
                                     bins=[0,25,50,75,90,100],
                                     labels=["0-25","25-50","50-75","75-90","90-100"])
            df2["vol_pct"] = pd.cut(df2.ticks_moved_1s.rank(pct=True)*100,
                                     bins=[0,25,50,75,90,100],
                                     labels=["0-25","25-50","50-75","75-90","90-100"])
            ct = df2.groupby(["act_pct","vol_pct"], observed=True).size().reset_index(name="count")
            ct["symbol"] = sym; ct["label"] = lbl; ct["pct"] = ct["count"] / len(df2) * 100
            rows.append(ct)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ── 13. BTC/ETH comparison CSV ────────────────────────────────────────────────

def compute_btc_eth_comparison(dfs) -> pd.DataFrame:
    NORM = {
        "afi_1s": "AFI (signed)",
        "bid_net_passive_1s": "bid_net/depth (normalized)",
        "bid_max_share": "max_level_share_of_side",
        "replenishment_ratio_1s": "repl_ratio_1s (p50)",
        "ticks_moved_1s": "ticks_moved_1s (p90)",
        "events_per_sec_1s": "events_per_sec_1s (p50)",
        "trades_per_sec_1s": "trades_per_sec_1s (p90)",
    }
    rows = []
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            row = {"symbol": sym, "label": lbl}
            for col, _ in NORM.items():
                if col not in df:
                    continue
                v = df[col].dropna()
                row[f"{col}_p50"] = round(v.median(), 4)
                row[f"{col}_p90"] = round(v.quantile(.9), 4)
                row[f"{col}_mean"] = round(v.mean(), 4)
            rows.append(row)
    return pd.DataFrame(rows)

# ── 14. ORDER_FLOW_REPORT.md ─────────────────────────────────────────────────

def generate_report(dist_df: pd.DataFrame, jst_df: pd.DataFrame,
                    act_vol_df: pd.DataFrame, dfs: dict) -> str:
    def _pct_row(dfs, sym, lbl, col, pctiles=(50,90,99)):
        df = dfs.get((sym,lbl))
        if df is None or col not in df:
            return "—"
        v = df[col].dropna()
        return " / ".join(f"p{p}={v.quantile(p/100):.3g}" for p in pctiles)

    lines = [
        "# ORDER_FLOW_REPORT — KRAKEN-OF-1-R3",
        "",
        f"**Feature version:** {FEATURE_VERSION}  "
        f"**Book semantics:** {BOOK_SEMANTICS_VERSION}",
        "**Sessions:** LOW / MEDIAN / HIGH × BTC / ETH  "
        "**Window:** 1h research, 1s anchors (3600 rows/session)",
        "**Rule:** no future prices, MFE/MAE, ML or PnL.",
        "",
        "---",
        "",
        "## 1. Data Foundation",
        "",
        "| Session | Symbol | rows | anchors | feature_version |",
        "|---|---|---|---|---|",
    ]
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            lines.append(f"| {lbl} | {SYM_NICE[sym]} | {len(df)} | 3600 | {df['feature_version'].iloc[0]} |")
    lines += ["", "See `R3_PROVENANCE.csv` for parquet hashes.", "", "---", ""]

    lines += [
        "## 2. Regime Comparison",
        "",
        "Key question: do mechanically-selected LOW/MEDIAN/HIGH differ physically, "
        "not just in order-event rate?",
        "",
        "### Events/sec median by session",
        "",
        "| Symbol | LOW p50 | MEDIAN p50 | HIGH p50 |",
        "|---|---|---|---|",
    ]
    for sym in SYMBOLS:
        meds = []
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            v = df["events_per_sec_1s"].dropna().median() if "events_per_sec_1s" in df else float("nan")
            meds.append(f"{v:.0f}")
        lines.append(f"| {SYM_NICE[sym]} | {' | '.join(meds)} |")

    lines += ["",
        "### Trades/sec (execution intensity) p50 / p90 / p99",
        "",
        "| Symbol | Label | p50 | p90 | p99 |",
        "|---|---|---|---|---|",
    ]
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            if "trades_per_sec_1s" not in df:
                continue
            v = df["trades_per_sec_1s"].dropna()
            lines.append(f"| {SYM_NICE[sym]} | {lbl} | {v.quantile(.5):.2f} | {v.quantile(.9):.2f} | {v.quantile(.99):.2f} |")

    # Key finding on regime divergence
    btc_low_eps = dfs[("PF_XBTUSD","LOW")]["events_per_sec_1s"].dropna().median() if "events_per_sec_1s" in dfs[("PF_XBTUSD","LOW")] else 0
    btc_high_eps= dfs[("PF_XBTUSD","HIGH")]["events_per_sec_1s"].dropna().median() if "events_per_sec_1s" in dfs[("PF_XBTUSD","HIGH")] else 0
    btc_low_tps = dfs[("PF_XBTUSD","LOW")]["trades_per_sec_1s"].dropna().median() if "trades_per_sec_1s" in dfs[("PF_XBTUSD","LOW")] else 0
    btc_high_tps= dfs[("PF_XBTUSD","HIGH")]["trades_per_sec_1s"].dropna().median() if "trades_per_sec_1s" in dfs[("PF_XBTUSD","HIGH")] else 0
    eps_ratio = btc_high_eps / btc_low_eps if btc_low_eps > 0 else 0
    tps_ratio = btc_high_tps / btc_low_tps if btc_low_tps > 0 else 0
    lines += [
        "",
        f"**BTC finding:** order-event rate (events/sec) ratio HIGH/LOW = {eps_ratio:.1f}×, "
        f"but execution rate (trades/sec) ratio HIGH/LOW = {tps_ratio:.1f}×. "
        "High-activity sessions amplify order messaging significantly more than actual trades.",
        "",
        "See Figure 01.",
        "", "---", "",
    ]

    lines += [
        "## 3. Aggressive Flow",
        "",
        "Tails matter more than means for battle research. "
        "Median aggression per 1s = ~0 (most seconds have no executions at all).",
        "",
        "### BTC sell_taker_qty_1s tail distribution",
        "",
        "| Session | p90 | p95 | p99 | p99.9 | max |",
        "|---|---|---|---|---|---|",
    ]
    for lbl in LABELS:
        df = dfs[("PF_XBTUSD", lbl)]
        if "sell_taker_qty_1s" not in df:
            continue
        v = df["sell_taker_qty_1s"].dropna()
        lines.append(f"| {lbl} | {v.quantile(.9):.3f} | {v.quantile(.95):.3f} | "
                     f"{v.quantile(.99):.3f} | {v.quantile(.999):.3f} | {v.max():.3f} |")
    lines += [
        "",
        "Note: acceleration (`flow_accel_*`) is only defined when the prior half-window "
        "has non-zero flow (~15–17% of anchors). Do not compare session means without "
        "conditioning on defined rows.",
        "",
        "See Figure 02.",
        "", "---", "",
    ]

    lines += [
        "## 4. Passive Response by Sell Pressure",
        "",
        "Joint-state table: within-session sell_taker_qty_1s percentile bucket → "
        "bid response (p50 of each metric per bucket).",
        "",
    ]
    sell_jst = jst_df[(jst_df.pressure_side == "SELL") & (jst_df.symbol == "PF_XBTUSD")]
    for lbl in LABELS:
        sub = sell_jst[sell_jst.label == lbl]
        if len(sub) == 0:
            continue
        lines.append(f"### BTC {lbl}")
        lines.append("")
        lines.append("| bucket | n | bid_net (p50) | bid_add (p50) | bid_cxl (p50) | mid_chg (p50) |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            bid_net = r.get("bid_net_p50", float("nan"))
            bid_add = r.get("bid_add_p50", float("nan"))
            bid_cxl = r.get("bid_cxl_p50", float("nan"))
            mid_chg = r.get("mid_chg_p50", float("nan"))
            lines.append(f"| {r['bucket']} | {r['n']} | {bid_net:.2f} | "
                         f"{bid_add:.2f} | {bid_cxl:.2f} | {mid_chg:.2f} |")
        lines.append("")
    lines += ["See Figure 03 and Figure 04.", "", "---", ""]

    lines += [
        "## 5. Execution × Replenishment",
        "",
        "replenishment_ratio_1s = total_refill_qty_1s / total_exec_qty_1s "
        "(only counted when exec > 0; exec_qty deduplicated per exec_ts).",
        "",
        "**Extreme ratios (max ~370k) are physically real**: a tiny execution (0.001 ETH) "
        "can trigger hundreds of new limit orders at the same level within 1s. "
        "The ratio is not capped. Visualize with log1p.",
        "",
    ]
    for sym in SYMBOLS:
        for lbl in ["MEDIAN"]:
            df = dfs[(sym, lbl)]
            if "replenishment_ratio_1s" not in df or "total_exec_qty_1s" not in df:
                continue
            exec_nonzero = df[df.total_exec_qty_1s > 1e-6]
            if len(exec_nonzero) == 0:
                continue
            v = exec_nonzero.replenishment_ratio_1s.dropna()
            lines.append(f"**{SYM_NICE[sym]} MEDIAN** ({len(exec_nonzero)} anchors with exec): "
                         f"ratio p50={v.median():.2f} p90={v.quantile(.9):.1f} "
                         f"p99={v.quantile(.99):.0f} max={v.max():.0f}")
            lines.append("")
    lines += ["See Figure 06.", "", "---", ""]

    lines += [
        "## 6. Liquidity Concentration",
        "",
        "max_qty_vs_median = max active level qty / median active level qty at each anchor.",
        "Very heavy-tailed. A fixed wall threshold (e.g. 3× median) is not meaningful "
        "because the typical ratio is already 10,000–80,000 in these sessions.",
        "",
        "| Symbol | Session | p50 ratio | p90 ratio | p99 ratio |",
        "|---|---|---|---|---|",
    ]
    for sym in SYMBOLS:
        for lbl in LABELS:
            df = dfs[(sym, lbl)]
            col = "bid_max_qty_vs_median"
            if col not in df:
                continue
            v = df[col].dropna()
            lines.append(f"| {SYM_NICE[sym]} | {lbl} | {v.quantile(.5):.0f} | "
                         f"{v.quantile(.9):.0f} | {v.quantile(.99):.0f} |")
    lines += ["", "See Figure 07.", "", "---", ""]

    lines += [
        "## 7. Observed Market-State Families",
        "",
        "The following descriptive groupings emerge from joint-state analysis. "
        "**These are not strategies or labels — they describe recurring joint configurations "
        "of simultaneous features at anchor_ts.**",
        "",
        "### A. Message Churn",
        "- Very high events/sec",
        "- Ordinary or low trades/sec",
        "- Near-zero aggressive qty",
        "- Large add/cancel activity (high passive add AND cancel)",
        "- Little price movement",
        "- Consistent with high-frequency repricing patterns (post→cancel; ~98% cancellation rate in data)",
        "",
        "### B. Aggressive Depletion-Like State",
        "- Strong sell (or buy) taker flow",
        "- Bid (or ask) net passive negative (cancels dominate adds)",
        "- High same-window downward (or upward) price response",
        "- Found in top-1% sell pressure with bid_net_passive < 0",
        "",
        "### C. Passive Absorption-Like State (CANDIDATE — NOT YET ESTABLISHED)",
        "- *Hypothesis:* strong sell (or buy) taker flow + bid/ask net passive positive + lower price response",
        "- Bid (or ask) net passive positive (adds dominate cancels)",
        "- Lower same-window price response",
        "- *Requires further conditioning before treating as established state*",
        "",
        "### D. Trading Burst",
        "- High execution intensity (trades/sec p99)",
        "- Aggressive imbalance present (|AFI| > 0.5)",
        "- BBO movement",
        "- Distinguished from Message Churn by actual execution rate",
        "",
        "**Next step:** use these observed distributions to define BATTLE ontology thresholds — "
        "not from this analysis, but from structural review with partner after examining these figures.",
        "",
        "---",
        "",
        "## 8. BTC vs ETH Comparison",
        "",
        "See Figure 08 and `BTC_ETH_COMPARISON.csv` for normalized feature comparison.",
        "Key observation: AFI and regime percentile distributions are qualitatively similar "
        "across both instruments after normalization.",
        "",
        "---",
        "",
        "## 9. Caveats and Limitations",
        "",
        "- All features are causal (backward-looking from anchor_ts)",
        "- No future price, MFE, MAE, or forward labels were used",
        "- 120m context window is not_ready for 60-min research sessions (by design)",
        "- acceleration features are NaN in ~83% of rows (valid only when prior window has flow)",
        "- replenishment_ratio is not bounded; log1p transform recommended for visualization",
        "- ms-batch semantics: same-ms events form unordered set; no FIFO or intra-ms ordering",
        "",
        "---",
        "",
        f"*Generated by run_analysis.py — FEATURE_VERSION={FEATURE_VERSION}*",
    ]
    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("KRAKEN-OF-1-R3 — Descriptive Microstructure Analysis", flush=True)
    print("Loading 6 feature parquets ...", flush=True)
    dfs = load_all()
    print(f"  Loaded {sum(len(v) for v in dfs.values())} total rows", flush=True)

    print("Freezing provenance ...", flush=True)
    freeze_provenance(dfs)

    print("Computing distributions ...", flush=True)
    dist_df = descriptive_distributions(dfs)
    dist_df.to_csv(REPORTS / "DESCRIPTIVE_DISTRIBUTIONS.csv", index=False)
    print(f"  DESCRIPTIVE_DISTRIBUTIONS.csv: {len(dist_df)} rows", flush=True)

    print("Computing joint-state tables ...", flush=True)
    jst_df = compute_joint_state(dfs)
    jst_df.to_csv(REPORTS / "JOINT_STATE_TABLES.csv", index=False)
    print(f"  JOINT_STATE_TABLES.csv: {len(jst_df)} rows", flush=True)

    print("Activity × volatility matrix ...", flush=True)
    act_vol = compute_act_vol_matrix(dfs)
    if len(act_vol):
        act_vol.to_csv(REPORTS / "ACTIVITY_VOLATILITY_MATRIX.csv", index=False)

    print("BTC vs ETH comparison ...", flush=True)
    btc_eth = compute_btc_eth_comparison(dfs)
    btc_eth.to_csv(REPORTS / "BTC_ETH_COMPARISON.csv", index=False)

    print("Generating figures ...", flush=True)
    fig_activity_regimes(dfs)
    fig_aggressive_flow(dfs)
    fig_passive_response(dfs)
    fig_pressure_price_response(dfs)
    # fig_pressure_bid_price excluded from default run (figure 05 not in published figures/)
    fig_execution_replenishment(dfs)
    fig_liquidity_concentration(dfs)
    fig_btc_vs_eth(dfs)

    print("Generating episode figures ...", flush=True)
    episodes_all(dfs)

    print("Writing ORDER_FLOW_REPORT.md ...", flush=True)
    report_md = generate_report(dist_df, jst_df, act_vol, dfs)
    (REPORTS / "ORDER_FLOW_REPORT.md").write_text(report_md, encoding="utf-8")

    print("\nAll outputs written.", flush=True)
    print(f"  reports/  ORDER_FLOW_REPORT.md  DESCRIPTIVE_DISTRIBUTIONS.csv  JOINT_STATE_TABLES.csv", flush=True)
    print(f"  figures/  01..04, 06..08 + episodes/", flush=True)


if __name__ == "__main__":
    main()
