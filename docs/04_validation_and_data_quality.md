Validation and Data Quality
QA1 — Ghost Order Audit

The first QA pass investigated a replay failure mode in which cancelled orders could remain active in the reconstructed book.

The working hypothesis was that the affected orders were Post/Limit orders whose OrderCancelled event appeared before the corresponding OrderPlaced event within the same millisecond.

The audit confirmed this pattern.

Evidence	ETH	BTC
Total crossing orders	5,871	8,998
cancel_ts == place_ts	5,524 / 5,526 = 99.96%	same-ms pattern confirmed
API response ordering	OrderCancelled before OrderPlaced	systematic same-ms ordering pattern

The key issue was therefore not the existence of the events themselves, but the attempt to impose an arbitrary within-millisecond sequence on events whose true sub-ms ordering is not recoverable from the source data.

The reconstruction logic was changed to use millisecond-batch semantics rather than relying on response order inside the same timestamp.

A dedicated regression test was added:

tests/test_same_ms_batch_invariance.py

The test randomly permutes events within each millisecond batch and verifies that the resulting post-batch book state is invariant.

Current regression result:

30 random permutations
100% post-batch state match

This invariant is now part of the reconstruction QA.

QA2 — Independent L2 Validation

After the same-millisecond replay issue was corrected, the reconstructed order book was validated against an independent L2 market-data source.

The comparison used sampled L2 snapshots collected separately at approximately 2.6-second cadence.

BBO validation
Metric	BTC	ETH
BBO exact match on stable-book observations	97.35%	98.47%
Match within 1 tick	99.12%	99.24%
Explained 2-tick+ mismatches	TIMESTAMP_ALIGNMENT 100%	TIMESTAMP_ALIGNMENT 100%
UNKNOWN mismatch	0.0%	0.0%

The remaining larger mismatches were attributable to timestamp alignment between the reconstructed historical event stream and the independently sampled L2 collector.

No unexplained mismatch category remained in the audited sample.

Reproducibility boundary

The raw L2 snapshot dataset used for this validation is not redistributed because it was collected separately from the public historical-event dataset used by the main pipeline.

The validation results are summarized in this repository, but:

the raw L2 snapshots are not included;
the original QA2 reconciliation artifact is not included;
the exact BBO-validation percentages therefore cannot be independently recomputed from the public repository alone.

These results should be interpreted as a documented external validation of the reconstruction, not as a fully self-contained public replication package for the separate L2 collector.

Hidden Events Between L2 Snapshots

The independent L2 collector sampled the market at approximately 2.6-second intervals, while the historical event stream contains the individual order-lifecycle events occurring between those snapshots.

The number of events hidden between adjacent L2 observations was substantial:

Metric	BTC	ETH
Events between snapshots, p50	~1,822	~1,048
Events between snapshots, max	~13,741	—

This demonstrates an important limitation of snapshot-only market data:

A sampled L2 book can validate observable book state, but it does not preserve the majority of individual order-lifecycle events occurring between snapshots.

For this reason, the L2 collector was used as an independent state-validation source rather than as the primary source for lifecycle reconstruction.

Six-Session Hard QA

Before entering the feature-generation stage, all six symbol × session combinations were required to pass a two-stage gated QA process.

The six combinations are:

BTC × LOW
BTC × MEDIAN
BTC × HIGH
ETH × LOW
ETH × MEDIAN
ETH × HIGH

No session was allowed into the feature pipeline without passing both stages.

Stage 1 — Download Truth

The first stage validates the downloaded historical event stream itself.

Checks include:

manifest interval verification;
expected research-window coverage;
duplicate event UID checks;
timestamp consistency;
same-millisecond event statistics;
source-file integrity.

The purpose of this stage is to establish that the downloaded event set represents the intended time interval without obvious duplication or truncation artifacts.

Stage 2 — Market Truth

The second stage validates the reconstructed market state.

Checks include:

full order-book replay;
crossed_book = 0;
negative_qty = 0;
duplicate_active_uid = 0;
near-BBO completeness of at least 99.9% at L1;
maker/taker execution linkage;
reconstruction consistency after same-ms batching.

All six sessions passed the hard QA gate.

Summary results are available in:

reports/SESSION_QA_SUMMARY.txt

Session-level QA can be reproduced by running:

session_qa.py

when the corresponding raw historical session files are available locally.

Known Limitations
Limitation	Impact
No sequence_id	True sub-millisecond event ordering cannot be recovered; approximately 82% of events share a timestamp with at least one other event
Millisecond timestamp granularity	Events sharing a timestamp must be reconstructed using batch semantics rather than an assumed response-order sequence
Locked market (best_bid == best_ask)	Can appear as an observable millisecond-batch state and may reflect cross-stream timestamp alignment rather than a persistent matching-engine lock
Warmup floor	Approximately 1.7% of BTC and 2.9% of ETH orders remain unknown-origin under the tested warmup lengths
Unknown-origin orders	Residual uncertainty remains in deep historical lifecycle state, but audited near-BBO completeness was unaffected across all six sessions
L2 collector uses local receive timestamps	L2 timestamps are not exchange-server timestamps and therefore introduce alignment uncertainty in QA2
L2 sampling cadence	Snapshot data cannot observe the full order-event sequence between samples
Raw L2 validation data not redistributed	QA2 percentages are documented but cannot be independently recomputed from the public repository alone
REST historical API	The research pipeline is designed for historical reconstruction; a live implementation would require a separate real-time endpoint and live-state synchronization
Validation Scope

The QA process establishes that the reconstructed book and derived feature dataset are internally consistent under the available source semantics.

It supports the following claims:

event intervals are correctly identified;
duplicate and malformed lifecycle states are controlled;
same-millisecond ambiguity is handled through batch semantics;
reconstructed near-BBO state is highly consistent with an independent L2 source;
the six frozen research sessions satisfy the required reconstruction invariants;
the resulting feature dataset can be generated from the validated event stream.

It does not establish:

true exchange-internal sub-millisecond ordering;
exact queue position;
exchange matching-engine sequence;
live-trading latency behavior;
passive fill probability;
strategy profitability;
predictive alpha.

Those questions require separate execution, episode-level, or live-data studies.

QA Interpretation

The reconstruction should therefore be interpreted as:

a causally valid millisecond-resolution historical market-state reconstruction under explicit batch semantics,

rather than as:

an exact reproduction of the exchange's hidden sub-millisecond matching-engine sequence.

This distinction is intentional and is carried forward into all subsequent feature and market-state analysis.