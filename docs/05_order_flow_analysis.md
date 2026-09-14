# Order Flow Analysis

## Feature Dataset (FEATURE_VERSION=r2.1)

For each of six sessions (LOW/MEDIAN/HIGH × BTC/ETH), a 1-second anchor grid
over the 60-minute research window produces 3,600 rows × 475 causal features.

All features are **strictly backward-looking**: a feature at anchor `t` uses only
events in `(t−W, t]` for its window `W`. No future prices, MFE, MAE, or labels.

Feature families:

| Family | Examples | Windows |
|---|---|---|
| Regime | events/sec, trades/sec, spread, depth | 100ms – 300s |
| Aggressive flow | buy/sell_taker_qty, AFI, flow_acceleration | 100ms – 300s |
| Passive flow | bid/ask_add_qty, bid/ask_cancel_qty, net_passive | 100ms – 300s |
| Price response | signed_mid_change, ticks_moved (unsigned), levels_crossed | 100ms – 10s |
| Liquidity | max_qty_vs_median, max_share, top1_dist_ticks | per anchor |
| Replenishment | exec_qty, refill_qty, replenishment_ratio by window | 100ms – 5s |

Feature semantics: see `reports/SEMANTICS_SANITY_REPORT.txt`.  
QA: `feature_qa.py` — all 6 sessions pass (NaN 6–9%, inf=0, range violations=0).

## Regime Comparison

Sessions were selected mechanically by BTC order-event rate. ETH activity does NOT
follow the same ordering (ETH MEDIAN < ETH LOW): regimes are instrument-local,
not a shared market clock.

**BTC order-message intensity vs execution intensity:**

| Session | Events/sec p50 | Trades/sec mean | Trades/sec p99 |
|---|---|---|---|
| LOW | 410 | 3.28 | 60 |
| MEDIAN | 443 | 2.30 | 39 |
| HIGH | 822 | 2.86 | 32 |

HIGH has ~2× order-message rate but *smaller* execution p99 than LOW.
Order-message burst ≠ trading burst.

## Joint-State Analysis

**Sell pressure × bid passive response (BTC, within-session percentile buckets):**

The key finding: for top-1% sell taker flow, `bid_net_passive < 0` at p50
in all sessions — passive bids cancel more than they add during strong aggressive pressure.
Same-window price response is negative (downward).

For BUY pressure: qualitatively similar in most sessions (ask_net and price direction reversed),
though not uniform across every bucket.

**Conditional split:** within top-10% sell flow, anchors where `bid_net_passive > 0`
(passive bids adding) show ~30% lower same-window ticks_moved than anchors where
`bid_net_passive < 0`. This cross-session regularity (9/12 cases) is the basis for the
absorption-candidate state.

## Observed Market-State Families

### Established (descriptive evidence in current data)

**A. Message Churn:** Very high order-event rate, ordinary execution intensity,
near-zero aggressive qty, high simultaneous bid/ask add+cancel turnover, little BBO movement.
Consistent with HFT repricing cycles (~98% of orders are cancelled within the same session).

**B. Aggressive Depletion-Like:** Top 10–1% sell or buy taker flow,
negative bid (or ask) net passive, significant same-window price response.
Consistent across LOW/MEDIAN/HIGH. For BUY: qualitatively similar in most cases.

### Candidate (signal present; episode-level outcome analysis needed)

**C. Passive Absorption-Like:** Strong sell flow + `bid_net_passive > 0` →
lower same-window price response. Cross-session regularity observed (9/12 cells),
but within-bucket variance is high and per-episode outcomes have not been examined.

**D. Trading Burst:** High execution intensity + AFI imbalance + BBO movement.
Anchors exist (trades/sec p99 = 32–60); joint episode structure pending.

## Next: BATTLE-1

The battle detector will define episode boundaries, classify states
(APPROACH → CONTACT → ATTACK → RESOLUTION), and measure resolution outcomes
(HOLD / BREAK / FALSE_BREAK / CHOP) — on the feature dataset already built here,
with battle ontology thresholds derived from the distributions above, not from
future-return optimization.
