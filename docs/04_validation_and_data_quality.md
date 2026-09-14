# Validation and Data Quality

## QA1 — Ghost Order Audit

The first QA pass tested the **lit-eligibility hypothesis**: the hypothesis that
ghost orders (never-placed orders kept active by a replay bug) were Post/Limit
orders whose `OrderCancelled` arrived before their `OrderPlaced` in the same ms.

**Result:** 100% of crossing orders were Post/Limit type — and the event-level
evidence was conclusive:

| Evidence | ETH | BTC |
|---|---|---|
| Total crossing orders | 5,871 | 8,998 |
| cancel_ts == place_ts (same ms) | 5,524 / 5,526 = **99.96%** | confirms pattern |
| File order in API response | `OrderCancelled` before `OrderPlaced` | systematic |

The fix produced a regression test: `tests/test_same_ms_batch_invariance.py` shuffles
events within each ms-batch and verifies that post-batch book state is invariant
(30 random permutations, 100% match).

## QA2 — Independent L2 Validation

After fixing the ghost-order bug, the reconstructed book was validated against
**independent L2 market data** (sampled snapshots from a separate collector,
~2.6s cadence).

| Metric | BTC | ETH |
|---|---|---|
| BBO exact match (stable book) | 97.35% | 98.47% |
| 1-tick tolerance match | 99.12% | 99.24% |
| Any 2-tick+ mismatch explanation | TIMESTAMP_ALIGNMENT 100% | TIMESTAMP_ALIGNMENT 100% |
| UNKNOWN mismatch | 0.0% | 0.0% |

"Stable book" = anchors where book had not changed for >1s before the L2 snapshot.
All remaining mismatches are attributed to the local-receive timestamp skew of the
L2 collector (not a reconstruction error).

**Hidden events between L2 snapshots:**

| Metric | BTC | ETH |
|---|---|---|
| Events between snapshots p50 | ~1,822 | ~1,048 |
| Events between snapshots max | ~13,741 | — |

Sampled L2 data misses the vast majority of individual events.

## Six-Session Hard QA

Before the feature pipeline, all six (symbol × session) combinations passed a
two-stage gated QA:

**Stage 1 — Download Truth:**  
Manifest interval verification, duplicate event UID check, timestamp monotonicity,
same-ms statistics.

**Stage 2 — Market Truth:**  
Full book replay, crossed_book=0, negative_qty=0, duplicate_active_uid=0,
near-BBO completeness ≥99.9% at L1, maker/taker execution linkage.

All six sessions passed. Results: `reports/SESSION_QA_SUMMARY.txt`.

## Known Limitations

| Limitation | Impact |
|---|---|
| No `sequence_id` | Sub-ms event ordering not recoverable; 82% events share a ts |
| Locked market (best_bid == best_ask) | Observable millisecond-batch state; may reflect cross-stream timestamp alignment rather than a persistent matching-engine lock |
| Warmup floor (1.7% BTC, 2.9% ETH unknown-origin) | Residual within tested warmup lengths; near-BBO completeness was unaffected in all six audited sessions |
| L2 collector uses local receive timestamps | Not exchange server time; explains timing mismatches in QA2 |
| REST API, not WebSocket | Historical data only; live stream would require different endpoint |
