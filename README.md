# Crypto Order Flow Research

A reproducible study of market microstructure from the **public Kraken Futures**
order-level REST API (BTC and ETH perpetuals, 2026).

> **What this project studies:** individual order placement, cancellation, modification,
> and execution — observed at millisecond resolution from a public endpoint with no
> authentication — and whether that data is sufficient to reconstruct a meaningful
> market-state timeseries.

> **This release makes no claim of predictive alpha or executable profitability; it
> establishes and studies the observable market-state representation on which later
> trading research will be based.**

## Research Pipeline

```
Public Kraken Futures REST API
          |
          v
  Individual Order Events          (orders: OrderPlaced / Cancelled / Updated)
  Individual Execution Events      (executions: maker + taker UIDs, price, qty)
          |
          v
  Order Lifecycle Reconstruction   (96.9% BTC complete in 35-min window)
          |
          v
  Millisecond-Batch Market State   (ms-batch semantics; no sub-ms ordering assumed)
          |
          v
  Independent L2 Validation        (sampled L2 snapshots vs reconstructed BBO)
          |
          v
  Order Flow Feature Dataset        (FEATURE_VERSION=r2.2, 615 columns: 608 causal features + timestamp + 6 metadata)
          |
          v
  Descriptive Microstructure Analysis   <- baseline release: v0.2.2-orderflow
          |
          v
  Market Battle Detection           <- next: BATTLE-1
          |
          v
  Impulse / Pullback / Hook
          |
          v
  Maker Entry Research
```

## Data Source

**Kraken Futures historical REST API (v3)** — no authentication required.

| Endpoint | Content |
|---|---|
| `/api/history/v3/market/{symbol}/orders` | Individual order events |
| `/api/history/v3/market/{symbol}/executions` | Matched trade executions |

Symbols: `PF_XBTUSD` (BTC perpetual), `PF_ETHUSD` (ETH perpetual).
Peak activity: 492–1124 order events/sec, 2–3 executions/sec.
Storage: 3–18 GB/session raw JSON; see `download_raw.py`.

## Validation

Before any analysis, the reconstruction was independently validated:

| Check | Result |
|---|---|
| BBO exact match (stable-book observations) | 97–98% (BTC), 98–99% (ETH) |
| 1-tick tolerance match (stable-book) | 99%+ both instruments |
| Six-Session Hard QA: crossed_book=0 | ✓ All sessions |
| Six-Session Hard QA: near-BBO completeness | 100% L1/L5/L10 all sessions |
| Same-ms ghost-order bug (pre-fix) | Found and fixed — see `docs/04_validation_and_data_quality.md` |

## Key Findings

1. **Order-message activity ≠ execution activity.**
   The HIGH session (selected by peak order-event rate) has ~2× more order events than LOW,
   but execution intensity does not scale proportionally — p99 trades/sec is smaller in HIGH (32)
   than LOW (60). Message burst ≠ trading burst ≠ price burst.

2. **At strong aggressive pressure, passive bid side cancels outweigh adds.**
   For top-1% sell-taker flow: `bid_net_passive < 0` at p50 in all six sessions.
   Same-window price response is negative (downward), confirming directional consistency.

3. Conditional signal for absorption-like state.

Across top-10% aggressive-flow anchors, splitting by the same-side passive response
(SELL: bid_net_passive; BUY: ask_net_passive), the passive-adds group
(net_passive > 0) shows lower same-window median ticks_moved_1s in
9/12 symbol × session × aggressor combinations; 3/12 are equal; 0/12 are higher.

This is a candidate state — episode-level outcome analysis is needed to establish it.

(Exact count reproducible from CONDITIONAL_FLOW_SPLIT.csv via run_r3_qa1.py.)

4. **Replenishment is heavy-tailed and physically real.**
   Per-session `replenishment_ratio_1s` (exec > 0 anchors): p50 ranges from 9 to 26,
   p90 from 393 to 1,287, observed maxima up to 466,447.
   Extreme values arise when tiny executions are followed by large bursts of same-level
   limit additions; consistent with rapid liquidity replenishment.
   *(Full distribution traceable from `reports/REPLENISHMENT_GLOBAL_SUMMARY.csv`
   and `reports/REPLENISHMENT_BY_EXEC_SIZE.csv` via `run_r3_qa1.py`.)*

5. **Passive repricing under attack varies across market regimes.**
   Under top-1% sell pressure, bid_reprice_away_qty / bid_reprice_toward_qty varies by session
   (0.87×–2.48× BTC; traceable in `REPRICE_SUMMARY.csv`).
   The `passive_cancels` group (net negative) shows higher median reprice-away than the
   `passive_adds` group in all 6 symbol × session SELL combinations; median per-combination
   ratio 1.59× (BTC median 1.47×, ETH median 1.97×), plus larger price response —
   consistent with retreat vs protection.
   *(Reprice ratio reproducible from `CONDITIONAL_FLOW_SPLIT.csv` via `run_r3_qa1.py`.
   The fraction of OrderUpdated events that reprice vs change quantity is a raw-event
   statistic not captured in anchor-level parquets.)*

6. **Near-market liquidity concentration requires local measurement.**
   The largest bid level is typically 100–330 ticks from mid — global `max_qty_vs_median`
   is not a useful battle-wall detector. Near-market zones (5/10/25 bps) needed.

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (no data required — all synthetic)
python -m pytest tests/ -q      # 85 tests PASS

# Run end-to-end demo on synthetic sample data (no download needed)
python run_sample.py
```

### Quick API Example

Download a single session from the public Kraken Futures API (~2h ≈ 3–18 GB raw):

```bash
python download_raw.py \
    --symbol PF_XBTUSD \
    --start  2026-09-06T13:00:00Z \
    --end    2026-09-06T15:00:00Z
```

Output: `data/raw/PF_XBTUSD_orders.jsonl` and `data/raw/PF_XBTUSD_executions.jsonl`

### Reproduce the Six-Session Study

The exact three sessions (LOW/MEDIAN/HIGH) are frozen in `reports/SESSION_MANIFEST.csv`.
To reproduce the full pipeline from scratch:

```bash
python download_sessions.py   # downloads all 6 symbol×session files using SESSION_MANIFEST
python session_qa.py          # validates downloads; requires SESSION_MANIFEST
python run_r2.py              # builds feature parquets (608 causal features × 3600 rows × 6 sessions)
python feature_qa.py
python run_analysis.py        # R3 initial figures and report
python run_r3_qa1.py          # QA1 revision → canonical ORDER_FLOW_REPORT.md
```

**Note:** Raw data (~3–18 GB per session) and feature parquets (~5–7 MB each) are
not included in this repository. The frozen `SESSION_MANIFEST.csv` ensures the same
six 2-hour windows are used.

The `data/sample/` directory contains a minimal synthetic subset of the Kraken Futures
API v3 event schema (the fields consumed by this pipeline). Used by tests and
`run_sample.py` to demonstrate the pipeline without downloading real data.

## Limitations

- No `sequence_id` in API: sub-millisecond event ordering not recoverable
- 82% of events share a timestamp_ms with others (up to 120 events/ms)
- Sampled L2 timestamps are local receive time, not exchange server time
- Warmup floor: 1.7% (BTC) / 2.9% (ETH) unknown-origin orders — residual within tested warmup lengths; near-BBO completeness unaffected in audited sessions
- Raw data: 3–18 GB/session; parquets: 5–7 MB/session (not included in this repo)

## Project Status

| Stage | Status |
|---|---|
| KRAKEN-HIST-0: Feasibility | ✓ Complete |
| KRAKEN-OF-1: QA1 (ghost orders) | ✓ Complete |
| KRAKEN-OF-1: QA2 (L2 reconciliation) | ✓ Complete |
| KRAKEN-OF-1: R1 Session selection | ✓ Complete |
| KRAKEN-OF-1: R2 Feature pipeline | ✓ Complete (FEATURE_VERSION=r2.2) |
| KRAKEN-OF-1: R3 Descriptive analysis | ✓ Complete |
| BATTLE-1: Episode detection | Next |

## License

MIT — see `LICENSE`.
Market-data notice: see `DATA_NOTICE.md`.

## Citation

If you use this code or methodology, please reference the repository URL and version tag `v0.2.2-orderflow`.
