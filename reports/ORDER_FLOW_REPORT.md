# ORDER_FLOW_REPORT — KRAKEN-OF-1-R3 (revised after QA1)

**Feature version:** r2.2  **Book semantics:** ms-batch-v1
**Sessions:** LOW / MEDIAN / HIGH x BTC / ETH  
**Window:** 1h research, 1s anchors (3600 rows/session)
**HARD RULE:** no future prices, MFE/MAE, ML or PnL. All features causal.

---

## 1. Regime Comparison

### Order-message intensity (events/sec p50)

| Symbol | LOW | MEDIAN | HIGH |
|---|---|---|---|
| BTC | 410.0 | 443.0 | 822.0 |
| ETH | 236.0 | 139.0 | 371.0 |

> **Note:** ETH session ordering does not match BTC. Sessions were selected by BTC order-event rate. ETH MEDIAN (eps=139) < ETH LOW (eps=236). BTC and ETH activity regimes are not co-linear on the same hour.

### Execution intensity (trades/sec) — full distribution

| Symbol | Session | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| BTC | LOW | 3.281 | 0.0 | 7.0 | 60.02 | 201 |
| BTC | MEDIAN | 2.3 | 0.0 | 5.0 | 39.0 | 153 |
| BTC | HIGH | 2.856 | 1.0 | 8.0 | 32.01 | 135 |
| ETH | LOW | 1.359 | 0.0 | 3.0 | 18.0 | 200 |
| ETH | MEDIAN | 1.252 | 0.0 | 3.0 | 22.01 | 93 |
| ETH | HIGH | 1.16 | 0.0 | 2.0 | 22.0 | 143 |

> **Corrected finding:** BTC HIGH has ~2× more order-messaging than LOW (eps p50 822 vs 410), but execution intensity does NOT scale proportionally. HIGH/LOW trades/sec: mean=0.87×, p90=1.14×, p99=0.53×. The selected HIGH session has *less* extreme execution tail than LOW. Order-message burst ≠ trading burst ≠ price burst. Underlying features are causal/backward-looking; pressure percentile buckets used in this descriptive analysis are retrospective within-session stratifications.

See Figure 01.

---

## 2. Aggressive Flow

### BTC sell_taker_qty_1s tails

| Session | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| LOW | 0.0263 | 0.1241 | 5.0000 | 10.0016 | 32.444 |
| MEDIAN | 0.0013 | 0.1025 | 2.0000 | 5.0497 | 15.480 |
| HIGH | 0.0293 | 0.1000 | 0.6931 | 5.6333 | 16.341 |

Median is 0 in all sessions — most 1-second windows have no executions. Future battle research focuses on the non-zero tail, not the mean.

See Figure 02.

---

## 3. Passive Response — SELL Pressure

Within-session percentile buckets for sell_taker_qty_1s:

### BTC LOW — SELL pressure

| bucket | n | bid_net p50 | bid_add p50 | bid_cxl p50 | mid_chg p50 | ticks p50 | repl p50 |
|---|---|---|---|---|---|---|---|
| 0-50% | 2854 | 0.26 | 60.41 | 59.81 | 0.00 | 0.00 | 0.00 |
| 50-90% | 385 | 0.40 | 76.76 | 76.90 | 0.00 | 1.00 | 0.00 |
| 90-95% | 181 | -0.74 | 100.53 | 95.26 | -1.00 | 3.00 | 0.46 |
| 95-99% | 145 | -3.00 | 130.96 | 132.40 | -4.00 | 5.00 | 10.64 |
| top-1% | 35 | -0.42 | 136.13 | 157.37 | -5.00 | 7.00 | 86.61 |

### BTC MEDIAN — SELL pressure

| bucket | n | bid_net p50 | bid_add p50 | bid_cxl p50 | mid_chg p50 | ticks p50 | repl p50 |
|---|---|---|---|---|---|---|---|
| 0-50% | 3107 | 0.16 | 59.08 | 58.47 | 0.00 | 0.00 | 0.00 |
| 50-90% | 135 | -0.51 | 87.59 | 85.25 | -2.00 | 2.00 | 0.00 |
| 90-95% | 178 | 0.61 | 105.83 | 108.32 | -3.00 | 4.00 | 0.00 |
| 95-99% | 143 | -4.92 | 149.29 | 154.13 | -7.00 | 7.50 | 0.88 |
| top-1% | 37 | -0.84 | 103.16 | 105.96 | -2.00 | 3.00 | 37.34 |

### BTC HIGH — SELL pressure

| bucket | n | bid_net p50 | bid_add p50 | bid_cxl p50 | mid_chg p50 | ticks p50 | repl p50 |
|---|---|---|---|---|---|---|---|
| 0-50% | 2748 | 0.63 | 79.18 | 76.59 | 0.00 | 1.00 | 0.00 |
| 50-90% | 492 | -2.16 | 118.56 | 123.19 | -3.00 | 4.00 | 0.00 |
| 90-95% | 180 | -0.63 | 111.41 | 111.34 | -0.50 | 4.50 | 4.95 |
| 95-99% | 144 | -1.35 | 108.74 | 104.60 | -2.00 | 4.00 | 3.86 |
| top-1% | 36 | -0.26 | 129.33 | 134.15 | -7.50 | 8.25 | 8.08 |

See Figure 03.

---

## 4. Passive Response — BUY Pressure

### BTC LOW — BUY pressure

| bucket | n | ask_net p50 | ask_add p50 | ask_cxl p50 | mid_chg p50 | ticks p50 |
|---|---|---|---|---|---|---|
| 0-50% | 3014 | 0.17 | 56.10 | 55.55 | 0.00 | 0.00 |
| 50-90% | 226 | 0.97 | 90.56 | 89.38 | 2.00 | 2.50 |
| 90-95% | 180 | -1.43 | 113.91 | 115.10 | 3.00 | 4.00 |
| 95-99% | 145 | -1.87 | 132.99 | 133.70 | 4.00 | 5.00 |
| top-1% | 35 | -5.07 | 123.68 | 131.17 | 4.00 | 5.00 |

### BTC MEDIAN — BUY pressure

| bucket | n | ask_net p50 | ask_add p50 | ask_cxl p50 | mid_chg p50 | ticks p50 |
|---|---|---|---|---|---|---|
| 0-50% | 3006 | 0.20 | 52.53 | 51.49 | 0.00 | 0.00 |
| 50-90% | 234 | -0.83 | 98.25 | 101.28 | 2.00 | 2.00 |
| 90-95% | 160 | -1.39 | 98.43 | 103.16 | 3.00 | 3.75 |
| 95-99% | 164 | -0.61 | 75.31 | 74.47 | 1.00 | 2.00 |
| top-1% | 36 | 0.41 | 155.22 | 164.55 | 7.00 | 7.00 |

### BTC HIGH — BUY pressure

| bucket | n | ask_net p50 | ask_add p50 | ask_cxl p50 | mid_chg p50 | ticks p50 |
|---|---|---|---|---|---|---|
| 0-50% | 2173 | 0.44 | 70.66 | 68.65 | 0.00 | 1.00 |
| 50-90% | 1067 | 0.08 | 70.18 | 69.71 | 0.00 | 2.00 |
| 90-95% | 180 | -2.46 | 130.71 | 132.06 | 4.00 | 5.00 |
| 95-99% | 144 | -0.17 | 122.82 | 123.84 | 4.00 | 5.00 |
| top-1% | 36 | -3.76 | 166.82 | 171.99 | 5.25 | 7.75 |


---

## 5. Conditional Split: Top-10% Flow x Passive Response Sign

Within top-10% sell (or buy) taker flow: split by passive-side net_passive sign.
Tests absorption-candidate hypothesis on same-window data only.

| Symbol | Session | Aggressor | Passive response | n | ticks p50 | ticks p90 | mid_chg p50 | repl p50 |
|---|---|---|---|---|---|---|---|---|
| BTC | LOW | SELL | passive_adds | 154 | 4.00 | 12.35 | -1.00 | 6.28 |
| BTC | LOW | SELL | passive_neutral | 6 | 2.00 | 7.00 | -0.50 | 36.05 |
| BTC | LOW | SELL | passive_cancels | 201 | 4.00 | 15.00 | -3.00 | 1.88 |
| BTC | LOW | BUY | passive_adds | 150 | 3.75 | 11.00 | 2.00 | 9.37 |
| BTC | LOW | BUY | passive_neutral | 6 | 2.50 | 18.00 | 1.00 | 4.81 |
| BTC | LOW | BUY | passive_cancels | 204 | 5.00 | 13.00 | 5.00 | 1.45 |
| BTC | MEDIAN | SELL | passive_adds | 155 | 4.00 | 13.00 | -3.00 | 1.06 |
| BTC | MEDIAN | SELL | passive_cancels | 202 | 6.00 | 15.00 | -5.00 | 0.46 |
| BTC | MEDIAN | BUY | passive_adds | 160 | 2.00 | 10.00 | 0.25 | 7.42 |
| BTC | MEDIAN | BUY | passive_neutral | 3 | 1.00 | 5.00 | 1.00 | 0.00 |
| BTC | MEDIAN | BUY | passive_cancels | 197 | 5.00 | 13.00 | 5.00 | 0.74 |
| BTC | HIGH | SELL | passive_adds | 161 | 4.00 | 11.00 | 0.00 | 6.93 |
| BTC | HIGH | SELL | passive_neutral | 5 | 7.50 | 12.20 | -7.50 | 259.00 |
| BTC | HIGH | SELL | passive_cancels | 194 | 5.00 | 14.00 | -3.00 | 3.62 |
| BTC | HIGH | BUY | passive_adds | 155 | 4.00 | 12.00 | 2.00 | 9.79 |
| BTC | HIGH | BUY | passive_neutral | 4 | 2.00 | 4.40 | 1.00 | 0.00 |
| BTC | HIGH | BUY | passive_cancels | 201 | 6.00 | 15.00 | 6.00 | 1.07 |

### 5.1 Cross-session absorption regularity

For each of 12 symbol × session × aggressor combinations: passive_adds vs passive_cancels median ticks_moved_1s.

| Symbol | Session | Aggressor | adds ticks p50 | cancels ticks p50 | comparison |
|---|---|---|---|---|---|
| ETH | HIGH | BUY | 1.50 | 2.00 | **lower** |
| ETH | HIGH | SELL | 2.00 | 2.50 | **lower** |
| ETH | LOW | BUY | 2.00 | 2.00 | **equal** |
| ETH | LOW | SELL | 1.50 | 3.00 | **lower** |
| ETH | MEDIAN | BUY | 2.00 | 2.00 | **equal** |
| ETH | MEDIAN | SELL | 2.00 | 3.00 | **lower** |
| BTC | HIGH | BUY | 4.00 | 6.00 | **lower** |
| BTC | HIGH | SELL | 4.00 | 5.00 | **lower** |
| BTC | LOW | BUY | 3.75 | 5.00 | **lower** |
| BTC | LOW | SELL | 4.00 | 4.00 | **equal** |
| BTC | MEDIAN | BUY | 2.00 | 5.00 | **lower** |
| BTC | MEDIAN | SELL | 4.00 | 6.00 | **lower** |

**Result: 9/12 lower (passive_adds < cancels), 3/12 equal, 0/12 higher.**
0/12 combinations show passive_adds with higher price response than passive_cancels.
Traceable from `CONDITIONAL_FLOW_SPLIT.csv`.

### 5.2 Reprice flow by passive-response group (top-10% sell pressure, 1s)

SELL aggressor: bid_reprice_away_qty_1s p50; ratio = passive_cancels / passive_adds (dimensionless).

| Symbol | Session | Group | bid_reprice_away p50 | bid_reprice_toward p50 | ratio |
|---|---|---|---|---|---|
| BTC | LOW | passive_adds | 2.9311 | 1.2760 |  |
| BTC | LOW | passive_cancels | 3.4600 | 2.8844 | 1.18× |
| BTC | MEDIAN | passive_adds | 3.1327 | 1.4685 |  |
| BTC | MEDIAN | passive_cancels | 5.3541 | 3.1412 | 1.71× |
| BTC | HIGH | passive_adds | 1.5079 | 2.9750 |  |
| BTC | HIGH | passive_cancels | 2.2234 | 1.5877 | 1.47× |
| BTC | **median** | | | | **1.47×** |
| ETH | LOW | passive_adds | 32.0610 | 34.4220 |  |
| ETH | LOW | passive_cancels | 102.4250 | 41.8270 | 3.19× |
| ETH | MEDIAN | passive_adds | 19.1320 | 5.0530 |  |
| ETH | MEDIAN | passive_cancels | 37.6870 | 8.1040 | 1.97× |
| ETH | HIGH | passive_adds | 94.6170 | 24.7460 |  |
| ETH | HIGH | passive_cancels | 103.6285 | 22.1820 | 1.10× |
| ETH | **median** | | | | **1.97×** |

> **Per-combination ratio (cancels/adds, bid_reprice_away):** all 6 symbol × session SELL ratios > 1. Median per-combination ratio = 1.59×. Traceable in `CONDITIONAL_FLOW_SPLIT.csv`.

### 5.3 Top-1% sell pressure: bid reprice away/toward ratio (BTC, 1s window)

| Session | away qty p50 | toward qty p50 | away/toward ratio |
|---|---|---|---|
| LOW | 3.1078 | 1.2547 | 2.48 |
| MEDIAN | 0.6487 | 0.7486 | 0.87 |
| HIGH | 3.7738 | 3.4267 | 1.10 |

Traceable in `REPRICE_SUMMARY.csv`.

---

## 6. Execution x Replenishment

replenishment_ratio = total_refill_qty / total_exec_qty (exec_qty deduplicated per exec_ts).
Show raw exec_qty and refill_qty alongside ratio — ratio alone is not interpretable.

### BTC — replenishment by execution-size bucket (1s window)

| Session | exec bucket | n | exec_qty p50 | refill_qty p50 | ratio p50 | ratio p90 |
|---|---|---|---|---|---|---|
| LOW | zero | 3067 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| LOW | p0-50 | 265 | 0.0043 | 0.1074 | 54.57 | 2331.2 |
| LOW | p50-90 | 214 | 0.0453 | 0.5477 | 10.64 | 299.9 |
| LOW | p90-95 | 27 | 0.5291 | 14.0729 | 28.20 | 92.1 |
| LOW | p95-99 | 21 | 0.9019 | 14.2288 | 15.32 | 60.7 |
| LOW | top-1% | 6 | 2.1340 | 27.8354 | 16.26 | 22.2 |
| MEDIAN | zero | 3148 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| MEDIAN | p0-50 | 224 | 0.0020 | 0.2052 | 100.10 | 3750.6 |
| MEDIAN | p50-90 | 182 | 0.0350 | 0.2314 | 7.34 | 111.3 |
| MEDIAN | p90-95 | 23 | 0.2954 | 0.9130 | 4.54 | 27.5 |
| MEDIAN | p95-99 | 18 | 0.6407 | 3.3624 | 5.53 | 25.7 |
| MEDIAN | top-1% | 5 | 1.4176 | 8.8244 | 6.59 | 46.1 |
| HIGH | zero | 2719 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| HIGH | p0-50 | 487 | 0.0038 | 0.0701 | 31.15 | 2964.0 |
| HIGH | p50-90 | 305 | 0.0205 | 0.2391 | 12.13 | 251.0 |
| HIGH | p90-95 | 44 | 0.0822 | 0.5674 | 7.06 | 274.0 |
| HIGH | p95-99 | 36 | 0.2687 | 1.2493 | 3.55 | 174.1 |
| HIGH | top-1% | 9 | 1.1892 | 3.4019 | 3.50 | 36.1 |

### ETH — replenishment by execution-size bucket (1s window)

| Session | exec bucket | n | exec_qty p50 | refill_qty p50 | ratio p50 | ratio p90 |
|---|---|---|---|---|---|---|
| LOW | zero | 3074 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| LOW | p0-50 | 262 | 0.0690 | 1.0345 | 26.93 | 1156.3 |
| LOW | p50-90 | 211 | 0.4720 | 2.2660 | 4.09 | 65.7 |
| LOW | p90-95 | 26 | 4.3385 | 45.1835 | 8.63 | 89.1 |
| LOW | p95-99 | 21 | 13.9200 | 34.6510 | 1.88 | 10.0 |
| LOW | top-1% | 6 | 29.0640 | 152.1320 | 1.74 | 17.2 |
| MEDIAN | zero | 3206 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| MEDIAN | p0-50 | 196 | 0.0400 | 2.4020 | 59.27 | 1629.5 |
| MEDIAN | p50-90 | 158 | 0.4345 | 4.7530 | 8.25 | 124.0 |
| MEDIAN | p90-95 | 20 | 2.9610 | 26.9610 | 9.32 | 188.0 |
| MEDIAN | p95-99 | 16 | 10.9685 | 68.7505 | 7.47 | 41.3 |
| MEDIAN | top-1% | 4 | 37.0050 | 660.3520 | 17.87 | 33.3 |
| HIGH | zero | 3252 | 0.0000 | 0.0000 | 0.00 | 0.0 |
| HIGH | p0-50 | 169 | 0.0420 | 1.5450 | 106.35 | 1534.8 |
| HIGH | p50-90 | 144 | 0.6325 | 3.5665 | 4.68 | 110.3 |
| HIGH | p90-95 | 17 | 4.5290 | 30.0770 | 5.20 | 59.9 |
| HIGH | p95-99 | 14 | 19.9010 | 173.7025 | 7.38 | 30.6 |
| HIGH | top-1% | 4 | 93.2095 | 784.2455 | 9.92 | 44.9 |

---

## 7. Near-Market Liquidity Concentration

Near-BBO depth (existing features from BBO replay):

| Symbol | Session | bid_5bps p50 | bid_10bps p50 | bid_25bps p50 | 5of25 conc p50 | wall_dist p50 ticks |
|---|---|---|---|---|---|---|
| BTC | LOW | 36.72 | 147.96 | 292.41 | 0.127 | 247.5 |
| BTC | MEDIAN | 33.59 | 149.57 | 295.55 | 0.114 | 326.5 |
| BTC | HIGH | 49.47 | 197.77 | 363.35 | 0.138 | 98.5 |
| ETH | LOW | 225.29 | 1459.80 | 4020.34 | 0.058 | 162.5 |
| ETH | MEDIAN | 236.06 | 1447.81 | 4355.73 | 0.054 | 174.5 |
| ETH | HIGH | 159.88 | 1222.21 | 4079.08 | 0.039 | 33.5 |

> **Finding:** Median `bid_top1_dist_ticks` is 247–327 ticks from mid in LOW/MEDIAN, and 98 ticks in HIGH. Only ~11% of 25bps bid depth is within 5bps of mid (BTC MEDIAN). The maximum-qty level (the candidate 'wall') is typically far from the current market. Global `max_qty_vs_median` across all active levels is NOT a useful battle-wall detector.

> **Implication:** Wall detection must use local neighborhood (within 5/10/25 bps of mid), not global book statistics. This will be implemented in the next pass.

---

## 8. Observed Market-State Families

**Rules:** only same-window features (t-W, t]. No future prices. No thresholds tuned on outcomes.

### OBSERVED (supported by current joint-state data)

**A. Message Churn**
- Very high order-event rate (events/sec p90-p99)
- Ordinary or low execution intensity (trades/sec near median)
- Near-zero aggressive taker qty
- High simultaneous bid/ask add AND cancel (high passive turnover both sides)
- Little BBO movement
- Consistent with high-frequency repricing patterns (~98% cancellation rate observed in data; participant identity not observable)

**B. Aggressive Depletion-Like State**
- Top 10-1% sell (or buy) taker flow
- bid_net_passive < 0 at p50 across all sessions
- Significant same-window downward (or upward) price movement
- Consistent across LOW/MEDIAN/HIGH BTC sessions
- Ask side shows analogous pattern for BUY pressure in most high-pressure buckets, but not uniformly across all sessions

### CANDIDATE STATES — NOT YET ESTABLISHED

**C. Passive Absorption-Like State**
- *Hypothesis:* strong sell flow + bid_net_passive > 0 + low same-window price response
- *Current evidence:* within top-10% sell flow the passive_adds group has fewer anchors; conditional split is available but p50 ticks requires more conditioning to distinguish from noise
- *Next step:* conditional analysis on `bid_net_passive > 0` AND `replenishment_ratio > threshold` simultaneously vs price response

**D. Trading Burst**
- *Hypothesis:* trades/sec in top percentile + |AFI| imbalance + BBO movement
- *Current evidence:* trades/sec p99 reaches 32-60 in BTC sessions; top-1% execution anchors exist; joint analysis with flow imbalance pending

### Why NOT to name these as established families yet
The conditional split shows some signal difference (e.g. `passive_adds` vs `passive_cancels` within top sell flow), but sample sizes per cell are small (n=3-15 in some buckets) and the within-bucket variance is high. These states need battle episode analysis (Part G of the original R1 plan) with resolved outcomes — which requires the BATTLE ontology freeze and separate R4 pass.

---

## 9. Caveats and Limitations

- All features strictly causal (backward-looking from anchor_ts)
- Pressure percentile buckets used in this descriptive analysis are retrospective within-session stratifications (not causal in the predictive sense)
- ms-batch semantics: same-ms events form unordered set
- 120m context window not_ready for 60-min research sessions (by design)
- acceleration features: NaN in ~83% of 1s rows (valid only when prior window has flow)
- Session selection was by BTC order-event rate; ETH activity ordering differs
- Global max_qty_vs_median includes deep-book levels; for battle wall detection, use near-market zones (5/10/25 bps) — not implemented yet

---

*FEATURE_VERSION=r2.2  |  BOOK_SEMANTICS_VERSION=ms-batch-v1  |  R3-QA1 revised*