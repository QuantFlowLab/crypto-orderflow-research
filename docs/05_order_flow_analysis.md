# Order Flow Analysis

## Feature Dataset (`FEATURE_VERSION=r2.2`)

For each of the six research sessions (LOW / MEDIAN / HIGH × BTC / ETH), a 1-second anchor grid over the 60-minute research window produces:

* **3,600 rows per session**
* **615 columns**
* **608 causal numeric features**
* **1 timestamp column**
* **6 metadata / provenance columns**

All features are **strictly backward-looking**. A feature evaluated at anchor `t` uses only events in `(t−W, t]` for its lookback window `W`.

No future prices, MFE, MAE, future returns, or outcome labels are used in feature construction.

### Feature families

| Family          | Examples                                                | Windows      |
| --------------- | ------------------------------------------------------- | ------------ |
| Regime          | events/sec, trades/sec, spread, depth                   | 100ms – 300s |
| Aggressive flow | buy/sell taker qty, AFI, flow acceleration              | 100ms – 300s |
| Passive flow    | bid/ask add qty, bid/ask cancel qty, net passive flow   | 100ms – 300s |
| Price response  | signed mid change, unsigned ticks moved, levels crossed | 100ms – 10s  |
| Liquidity       | max qty vs median, max share, top-1 distance in ticks   | per anchor   |
| Replenishment   | execution qty, refill qty, replenishment ratio          | 100ms – 5s   |

Feature semantics are implemented primarily in `src/passive_flow.py` and covered by tests including `tests/test_reprice_flow.py`.

Feature QA is performed by `feature_qa.py`.

Across all six sessions:

* all sessions pass QA;
* NaN share is approximately **5–7%**;
* infinite values: **0**;
* range violations: **0**.

The complete feature schema is recorded in `reports/FEATURE_COLUMNS_r2.2.txt`, with session-level QA results in `reports/FEATURE_QA_REPORT.csv`.

---

## Regime Comparison

Sessions were selected mechanically using BTC order-event rate.

ETH activity does **not** follow the same LOW → MEDIAN → HIGH ordering. In particular, ETH activity during the BTC-selected MEDIAN session is lower than during the BTC-selected LOW session.

This is an important constraint on interpretation:

> The LOW / MEDIAN / HIGH labels describe the BTC-selected market regime and should not be treated as a universal market-wide activity clock.

### BTC order-message intensity vs execution intensity

| Session | Events/sec p50 | Trades/sec mean | Trades/sec p99 |
| ------- | -------------: | --------------: | -------------: |
| LOW     |            410 |            3.28 |             60 |
| MEDIAN  |            443 |            2.30 |             39 |
| HIGH    |            822 |            2.86 |             32 |

The HIGH session has roughly twice the median order-message rate of LOW, while its execution-intensity tail is smaller:

* LOW trades/sec p99: **60**
* HIGH trades/sec p99: **32**

Therefore:

> **Order-message burst ≠ execution burst.**

A high rate of order lifecycle activity does not necessarily imply a proportionally high rate of executed trading.

---

## Joint-State Analysis

### Strong SELL pressure and passive bid response

For top-1% sell taker-flow anchors, `bid_net_passive < 0` at the median in **all six symbol × session combinations**.

In other words, during the strongest observed aggressive SELL pressure, passive bid liquidity is typically being removed faster than it is being added.

The same-window median price response is also negative in all six combinations.

This joint state is consistent with an **aggressive depletion-like regime**:

> strong aggressive selling + net passive bid withdrawal + downward same-window price response.

### Strong BUY pressure

BUY-side behavior is qualitatively similar in most sessions:

* aggressive BUY pressure is evaluated against ask-side passive response;
* the relevant passive variable is `ask_net_passive`;
* price direction is reversed relative to the SELL case.

The pattern is not uniform across every percentile bucket, so BUY and SELL states should continue to be evaluated separately rather than assumed to be perfectly symmetric.

### Conditional signal for an absorption-like state

Across top-10% aggressive-flow anchors, the sample is split by the passive response on the side being attacked:

* **SELL:** `bid_net_passive`
* **BUY:** `ask_net_passive`

The passive-adds group (`net_passive > 0`) shows lower same-window median `ticks_moved_1s` in:

* **9/12** symbol × session × aggressor combinations;
* **3/12** are equal;
* **0/12** are higher.

This cross-session regularity is the basis for the **passive absorption-like candidate state**.

It should not yet be interpreted as a predictive trading signal. The current result is descriptive and conditional on contemporaneous order-flow state.

Episode-level outcome analysis is required to determine whether the state has predictive value beyond the same-window relationship observed here.

*(The exact 9 / 3 / 0 count is reproducible from `reports/CONDITIONAL_FLOW_SPLIT.csv` via `run_r3_qa1.py`.)*

---

## Observed Market-State Families

### Established descriptive states

#### A. Message Churn

Characteristics:

* very high order-event rate;
* ordinary or only moderately elevated execution intensity;
* near-zero aggressive quantity during many anchors;
* simultaneous bid/ask add-and-cancel turnover;
* limited BBO displacement.

This state is consistent with high-frequency repricing and order-lifecycle churn.

Approximately 98% of observed orders are cancelled within the same session, reinforcing the importance of treating order lifecycle activity separately from executed flow.

The current evidence establishes this as a **descriptive market state**, not as a trading signal.

---

#### B. Aggressive Depletion-Like

Characteristics:

* top 10–1% aggressive SELL or BUY taker flow;
* negative passive response on the attacked side:

  * SELL → negative `bid_net_passive`;
  * BUY → negative `ask_net_passive`;
* material same-window price response in the aggressive direction.

The SELL-side relationship is observed consistently across LOW / MEDIAN / HIGH sessions for both BTC and ETH.

BUY-side behavior is qualitatively similar in most cases, although not uniformly across every bucket.

This is therefore treated as an established **descriptive order-flow state**.

---

### Candidate states

#### C. Passive Absorption-Like

Characteristics:

* strong aggressive flow;
* passive liquidity on the attacked side is added rather than withdrawn:

  * SELL → `bid_net_passive > 0`;
  * BUY → `ask_net_passive > 0`;
* lower same-window median price displacement relative to the corresponding passive-withdrawal group.

Across the 12 symbol × session × aggressor combinations:

* 9 show lower median `ticks_moved_1s`;
* 3 are equal;
* none are higher.

This is a **candidate state**, not an established predictive signal.

Within-bucket variance remains high, and episode-level outcomes have not yet been evaluated.

---

#### D. Trading Burst

Characteristics:

* elevated execution intensity;
* directional AFI imbalance;
* observable BBO movement.

Trading-burst anchors clearly exist in the current dataset, with session-level `trades/sec` p99 values ranging from approximately **32 to 60**.

However, the joint temporal structure of these bursts has not yet been formalized into episodes.

This state therefore remains a candidate for subsequent event-based analysis.

---

## Interpretation Boundaries

The results in this stage are **descriptive order-flow findings**.

They establish that several distinct microstructure states can be separated using causal features available at the observation anchor.

They do **not** establish:

* predictive alpha;
* profitable entry or exit rules;
* executable strategy economics;
* causal impact of individual order-flow variables;
* persistence of the observed states beyond the analyzed window;
* episode-level HOLD / BREAK probabilities.

In particular, same-window price response is used here to characterize the contemporaneous state. It must not be interpreted as an out-of-sample future-return target.

---

## Next: BATTLE-1

The next stage is **BATTLE-1**, which moves from anchor-level descriptive analysis to explicit market episodes.

The battle detector will:

1. define episode boundaries;

2. classify the evolving state sequence;

3. identify stages such as:

   `APPROACH → CONTACT → ATTACK → RESOLUTION`

4. measure episode outcomes such as:

   `HOLD / BREAK / FALSE_BREAK / CHOP`

The detector will operate on the causal feature dataset already constructed in this stage.

Battle ontology and thresholds should be derived from the observed feature and state distributions — **not from future-return optimization**.

This separation is intentional:

> the current stage establishes observable market-state structure;
> BATTLE-1 will test whether those states organize into economically meaningful episodes.
