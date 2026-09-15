# Market Data Access

## API Overview

Kraken Futures exposes individual order and execution history via an unauthenticated
public REST API (v3):

| Stream | Endpoint |
|---|---|
| Orders | `GET https://futures.kraken.com/api/history/v3/market/{symbol}/orders` |
| Executions | `GET https://futures.kraken.com/api/history/v3/market/{symbol}/executions` |

**No API key required.** Historical depth observed during this study was sufficient to cover
all research windows used; retention depth may change and should not be treated as guaranteed.

Query parameters: `since` (epoch ms), `before` (epoch ms), `count` (1000 used in this study; server-side upper bound is API-defined), `sort` (asc/desc),
`continuation_token` (request cursor, snake_case), `continuationToken` (response cursor, camelCase).

## Symbols Studied

| Symbol | Instrument | Tick size |
|---|---|---|
| `PF_XBTUSD` | BTC/USD perpetual | 1.0 USD |
| `PF_ETHUSD` | ETH/USD perpetual | 0.1 USD |

## Cardinality

**Phase A (initial measurement):** 16.7 events/sec — flagged as unexpectedly low.
**Phase A2 (corrected):** 239 events/sec off-peak — API endpoint was wrong in Phase A.
**Phase B (peak US-close session):** 492 events/sec BTC, 490 events/sec ETH.

The ~2× difference between off-peak and peak is purely session-time: the corrected
figures are consistent, not contradictory.

| Metric | BTC (peak) | ETH (peak) |
|---|---|---|
| Order events/sec | 492.6 | 489.6 |
| Execution events/sec | 2.60 | 2.49 |
| Estimated GB/day (raw) | ~17.6 | ~17.5 |

## Pagination

`continuation_token` — request query parameter (snake_case, per Kraken API docs).
`continuationToken` — response field (camelCase, returned in each API response).
`sort=desc` is sent explicitly in each request; events arrive newest-first within the window.
Downstream code sorts by `timestamp_ms`.

`MAX_PAGES = 50,000` is enforced as a fatal safety fuse in `download_raw.py`.
If reached, the download fails and cleans up the partial file — silent truncation
is a harder class of bug than a visible error (see changelog).

## Storage Architecture

```
Kraken REST API
      |
      v
  *.jsonl.part          (write-in-progress; renamed atomically on success)
      |
      v
  *.jsonl               (canonical raw; one JSON object per line)
      |
      v
  *.manifest            (completion sentinel: first/last ts, n_rows, status=COMPLETE)
      |
      v
  data/derived/features/*.parquet   (feature dataset; ~5-7 MB per session)
```

Raw JSON at 35 GB/day is not practical for long-term storage. The pipeline
retains raw files for QA windows and selected research sessions only.
